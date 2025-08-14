"""
LangGraph node handlers for AskuraAgent, with optimized routing and safeguards.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.graph.message import add_messages

from cogents.common.logging import get_logger
from cogents.common.utils import get_enum_value

from .prompts import get_unified_next_question_prompt
from .schemas import AskuraConfig, AskuraState, InformationSlot

logger = get_logger(__name__)


class AskuraNodes:
    """Encapsulates node implementations and routing policies for AskuraAgent."""

    def __init__(
        self,
        *,
        config: AskuraConfig,
        conversation_manager,
        information_extractor,
        question_generator,
        llm_client=None,
    ) -> None:
        self.config = config
        self.conversation_manager = conversation_manager
        self.information_extractor = information_extractor
        self.question_generator = question_generator
        self.llm = llm_client

    # --- Helpers -----------------------------------------------------------------

    def _is_slot_complete(self, slot: InformationSlot, value: Any) -> bool:
        if value in (None, "", [], {}):
            return False
        if slot.extraction_model and isinstance(value, dict):
            try:
                # Pydantic v2: check required fields on the model
                required_fields = [
                    name for name, field in slot.extraction_model.model_fields.items() if field.is_required
                ]
                for field_name in required_fields:
                    if value.get(field_name) in (None, "", [], {}):
                        return False
            except Exception:
                # If introspection fails, fall back to non-empty check
                return True
        return True

    def _missing_required_slots(self, state: AskuraState) -> List[InformationSlot]:
        info = state.extracted_information_slots
        missing: List[InformationSlot] = []
        for slot in self.config.information_slots:
            if slot.required and not self._is_slot_complete(slot, info.get(slot.name)):
                missing.append(slot)
        # Highest priority first (larger number means higher priority)
        missing.sort(key=lambda s: s.priority, reverse=True)
        return missing

    def _ready_to_summarize(self, state: AskuraState) -> bool:
        # Summarize only when all required slots are complete
        return len(self._missing_required_slots(state)) == 0 and state.turns > 1

    def _is_stalled(self, state: AskuraState) -> bool:
        attempts: Dict[str, int] = state.custom_data.get("slot_question_attempts", {})
        max_attempts_per_slot = 2
        too_many_attempts = any(count >= max_attempts_per_slot for count in attempts.values())
        near_turn_limit = state.turns >= max(self.config.max_conversation_turns - 1, 1)
        return too_many_attempts or near_turn_limit

    def _choose_next_slot_to_ask(self, state: AskuraState) -> Optional[str]:
        missing = self._missing_required_slots(state)
        if not missing:
            return None
        # Choose the highest priority missing slot whose dependencies are satisfied
        info = state.extracted_information_slots
        for slot in missing:
            if not slot.dependencies:
                return slot.name
            if all(info.get(dep) for dep in slot.dependencies):
                return slot.name
        return missing[0].name

    def _increment_slot_attempt(self, state: AskuraState, slot_name: Optional[str]) -> None:
        if not slot_name:
            return
        attempts: Dict[str, int] = state.custom_data.get("slot_question_attempts", {})
        attempts[slot_name] = attempts.get(slot_name, 0) + 1
        state.custom_data["slot_question_attempts"] = attempts

    # --- Nodes -------------------------------------------------------------------

    def conversation_context_analysis_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("ConversationContextAnalysis: Analyzing conversation context")
        conversation_context = self.conversation_manager.analyze_conversation_context(state)
        state.conversation_context = conversation_context

        recent_user_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)][-3:]
        # Store for potential downstream use
        state.custom_data["recent_user_messages"] = recent_user_messages
        return state

    def determine_next_action_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("DetermineNextAction: Selecting next action")
        conversation_context = state.conversation_context
        # Always extract fresh recent user messages to avoid stale data - optimize for token efficiency
        recent_user_messages = self._format_recent_user_messages(state.messages)

        action_result = self.conversation_manager.determine_next_action(
            state=state,
            context=conversation_context,
            recent_messages=recent_user_messages,
            ready_to_summarize=self._ready_to_summarize(state),
        )
        state.next_action_ayalysis = action_result
        state.turns += 1
        logger.info(
            f"Next action: {action_result.next_action} "
            f"(intent: {action_result.intent_type}, confidence: {action_result.confidence})"
        )
        return state

    def completeness_evaluator_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("CompletenessEvaluator: Evaluating info completeness and gap")
        missing_required_slots = [s.name for s in self._missing_required_slots(state)]
        target_slot = self._choose_next_slot_to_ask(state)
        gap_summary = ", ".join(missing_required_slots) if missing_required_slots else "none"
        state.custom_data.update(
            {
                "missing_required_slots": missing_required_slots,
                "target_slot": target_slot,
                "gap_summary": gap_summary,
                "all_required_complete": len(missing_required_slots) == 0,
            }
        )
        return state

    def information_extractor_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("InformationExtractor: Extracting information from user message")

        if not state.messages:
            logger.warning("InformationExtractor: No messages to extract information from")
            return state

        last_user_msg = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)
        if not last_user_msg:
            logger.warning("InformationExtractor: No last user message to extract information from")
            return state

        extracted_info = self.information_extractor.extract_all_information(last_user_msg.content, state)
        state = self.information_extractor.update_state_with_extracted_info(state, extracted_info)

        # Enhanced context enrichment for specific interests
        self._enrich_context_with_suggestions(state, last_user_msg.content)

        state.pending_extraction = False
        return state

    def _enrich_context_with_suggestions(self, state: AskuraState, user_message: str) -> None:
        """Enrich context with specific suggestions when user shows interest but lacks knowledge."""
        trip_context = state.extracted_information_slots.get("trip_plan_context", {})
        must_see = trip_context.get("must_see", [])

        # Check if user is interested in anime locations but needs suggestions
        anime_keywords = ["小丸子", "柯南", "动漫", "电影中出现的地点", "movie locations", "anime", "访问下电影中"]
        contains_anime_interest = any(keyword in user_message for keyword in anime_keywords)

        if contains_anime_interest and "Japan" in str(trip_context.get("destination", "")):
            # Add specific anime location suggestions to the context
            anime_suggestions = [
                "Shizuoka (小丸子的故乡 - Chibi Maruko-chan's hometown)",
                "Tokyo (Detective Conan crime scenes in Shibuya, Harajuku)",
                "Osaka (Detective Conan's Kansai locations)",
                "Tottori (Detective Conan author's hometown with museum)",
            ]

            # Update must_see with concrete suggestions
            if isinstance(must_see, list):
                for suggestion in anime_suggestions:
                    if suggestion not in must_see:
                        must_see.append(suggestion)

            # Update custom_data with enriched suggestions for question generation
            if "enriched_suggestions" not in state.custom_data:
                state.custom_data["enriched_suggestions"] = {}

            state.custom_data["enriched_suggestions"]["anime_locations"] = anime_suggestions

            # Update the extracted slots
            if "trip_plan_context" in state.extracted_information_slots:
                state.extracted_information_slots["trip_plan_context"]["must_see"] = must_see
                # Increase confidence since we're adding helpful information
                current_confidence = state.extracted_information_slots["trip_plan_context"].get("confidence", 0.5)
                state.extracted_information_slots["trip_plan_context"]["confidence"] = min(
                    current_confidence + 0.1, 1.0
                )

                logger.info(f"Enhanced context with anime location suggestions: {anime_suggestions}")

    def _format_recent_user_messages(self, messages) -> List[str]:
        """Format recent user messages while preserving important context."""
        # Take last 3 user messages, but keep more content for context
        user_messages = []
        recent_messages = [m for m in messages if isinstance(m, HumanMessage)][-3:]

        for msg in recent_messages:
            # Preserve full short messages, only truncate very long ones
            content = msg.content if len(msg.content) <= 200 else msg.content[:200] + "..."
            user_messages.append(content)

        return user_messages

    def question_generator_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("QuestionGenerator: Generating contextual question")

        next_action = state.next_action_ayalysis.next_action
        conversation_context = state.conversation_context

        # Unified next-question generation for both smalltalk and task
        utterance = None
        try:
            if self.llm is not None:
                info = state.extracted_information_slots
                missing_required_slots = [s.name for s in self._missing_required_slots(state)]
                target_slot = self._choose_next_slot_to_ask(state)
                gap_summary = ", ".join(missing_required_slots) if missing_required_slots else "none"
                prompt = get_unified_next_question_prompt(
                    conversation_purposes=self.config.conversation_purposes,
                    intent_type=state.next_action_ayalysis.intent_type,
                    next_action_label=next_action,
                    next_action_reasoning=state.next_action_ayalysis.reasoning,
                    conversation_style=get_enum_value(conversation_context.conversation_style),
                    conversation_on_track_confidence=conversation_context.conversation_on_track_confidence,
                    momentum=get_enum_value(conversation_context.conversation_momentum),
                    sentiment=get_enum_value(conversation_context.last_message_sentiment),
                    known_slots=info,
                    missing_required_slots=missing_required_slots,
                    missing_info=conversation_context.missing_info,
                    suggested_next_topics=conversation_context.suggested_next_topics,
                    target_slot=target_slot or "",
                    gap_summary=gap_summary,
                    enriched_suggestions=state.custom_data.get("enriched_suggestions", {}),
                )
                utterance = self.llm.chat_completion(
                    messages=[{"role": "system", "content": prompt}],
                    temperature=0.6,
                    max_tokens=64,
                )
                if isinstance(utterance, str):
                    utterance = utterance.strip()
                if target_slot:
                    self._increment_slot_attempt(state, target_slot)
        except Exception:
            utterance = None

        if not utterance or len(utterance) < 3:
            # Fallbacks
            if next_action == "reply_smalltalk":
                utterance = "What destination are you considering, and when are you hoping to travel?"
            else:
                fallback = self.question_generator.generate_contextual_question(
                    next_action or "ask_info", state, conversation_context
                )
                utterance = fallback

        ai_message = AIMessage(content=utterance)
        state.messages = add_messages(state.messages, [ai_message])
        state.requires_user_input = True
        return state

    def summarizer_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("Summarizer: Generating conversation summary")

        if self._ready_to_summarize(state) or state.turns >= self.config.max_conversation_turns:
            summary = self._generate_summary(state)
            summary_message = AIMessage(content=summary)
            state.messages = add_messages(state.messages, [summary_message])
            state.is_complete = True
            state.requires_user_input = False
        else:
            # Not ready yet; keep asking
            state.requires_user_input = True
        return state

    def human_review_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        logger.info("HumanReview: Awaiting human input")
        # When resumed, mark extraction needed
        state.requires_user_input = False
        state.pending_extraction = True
        return state

    # --- Routers -----------------------------------------------------------------

    def determine_next_action_router(self, state: AskuraState) -> str:
        logger.info("DetermineNextActionRouter: Routing after next action decision")
        next_action = state.next_action_ayalysis.next_action
        if state.pending_extraction:
            return "information_extractor"
        if next_action == "reply_smalltalk" or (next_action and next_action.startswith("ask_")):
            return "question_generator"
        if self._missing_required_slots(state):
            return "question_generator"
        if self._ready_to_summarize(state) or state.turns >= self.config.max_conversation_turns:
            return "summarizer"
        return "question_generator"

    def human_review_router(self, state: AskuraState) -> str:
        logger.info("HumanReviewRouter: Routing human review")
        if state.is_complete:
            return "end"
        return "continue"

    # --- Summary helpers ---------------------------------------------------------

    def _generate_summary(self, state: AskuraState) -> str:
        information_slots = state.extracted_information_slots
        summary_parts: List[str] = []
        for slot in self.config.information_slots:
            if information_slots.get(slot.name):
                summary_parts.append(f"{slot.name}: {information_slots[slot.name]}")
        return "Summary: " + " | ".join(summary_parts) if summary_parts else "Conversation completed."
