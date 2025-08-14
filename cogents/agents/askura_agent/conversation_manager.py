"""
Conversation Manager for AskuraAgent - Handles dynamic conversation analysis and flow control.
"""

import random
from typing import Any, Dict, List, Optional

from langchain_core.messages import HumanMessage

from cogents.common.llm import BaseLLMClient
from cogents.common.logging import get_logger
from cogents.common.utils import get_enum_value

from .prompts import get_conversation_analysis_prompt
from .schemas import (
    AskuraConfig,
    AskuraState,
    ConversationContext,
    ConversationDepth,
    ConversationMomentum,
    ConversationSentiment,
    ConversationStyle,
    NextActionResponse,
    UserConfidence,
)

logger = get_logger(__name__)


class ConversationManager:
    """Manages dynamic conversation analysis and flow control."""

    def __init__(self, config: AskuraConfig, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the conversation manager."""
        self.config = config
        self.llm = llm_client

    def analyze_conversation_context(self, state: AskuraState) -> ConversationContext:
        """Analyze conversation context to understand user preferences and conversation flow."""
        context = ConversationContext(conversation_purpose=self.config.conversation_purposes[0])

        # Analyze recent messages for context
        recent_messages = state.messages[-8:]

        if not recent_messages:
            logger.warning("No recent messages found")
            return context

        # Analyze user engagement and style
        user_messages = [msg for msg in recent_messages if isinstance(msg, HumanMessage)]
        if not user_messages:
            logger.warning("No user messages found")
            context.missing_info = self._get_missing_information_prioritized(state)
            return context

        last_user_text = user_messages[-1].content

        try:
            if not self.llm or not isinstance(last_user_text, str):
                raise ValueError("LLM client or last user text is not valid")

            # Prepare recent messages for analysis
            recent_messages_text = "\n".join(m.content for m in user_messages[-5:] if isinstance(m.content, str))[
                -1200:
            ]  # Limit to last 1200 chars

            # Get structured prompt for conversation analysis
            prompt = get_conversation_analysis_prompt(
                "conversation_context",
                conversation_purpose=context.conversation_purpose,
                recent_messages=recent_messages_text,
            )

            # Use structured completion with retry for reliable analysis
            context = self.llm.structured_completion(
                messages=[{"role": "user", "content": prompt}],
                response_model=ConversationContext,
                temperature=0.3,
                max_tokens=500,
            )

            logger.info(f"Conversation context (from LLM): {context}")

        except Exception as e:
            logger.warning(f"Error analyzing conversation context using LLM: {e}")
            # Fallback to heuristics
            last_user_msg = last_user_text.lower()
            style_info = self._detect_conversation_style(last_user_msg, user_messages)
            context.conversation_style = style_info["conversation_style"]
            context.information_density = style_info["information_density"]

            sentiment_info = self._analyze_sentiment_and_momentum(last_user_msg, user_messages)
            context.last_message_sentiment = sentiment_info["last_message_sentiment"]
            context.conversation_momentum = sentiment_info["conversation_momentum"]

            depth_info = self._analyze_conversation_depth(last_user_msg, user_messages)
            context.conversation_depth = depth_info["conversation_depth"]
            context.user_confidence = depth_info["user_confidence"]

            logger.warning(f"Conversation context (fallback to heuristics): {context.model_dump()}")

        # Analyze what information we have and what's missing
        context.missing_info = self._get_missing_information_prioritized(state)

        return context

    def determine_next_action(
        self,
        state: AskuraState,
        context: ConversationContext,
        recent_messages: List[str],
        ready_to_summarize: bool = False,
    ) -> NextActionResponse:
        """
        Unified method to determine next action with intent classification.

        This method combines intent classification and next action determination
        into a single LLM call for better consistency and efficiency.
        """
        try:
            # Prepare available actions
            missing_info = context.missing_info or []
            allowed = [f"ask_{s}" for s in [m.replace("ask_", "") for m in missing_info]]
            if ready_to_summarize:
                allowed.append("summarize")
            allowed.extend(["redirect_conversation", "reply_smalltalk"])

            # Get structured prompt for unified next action determination
            prompt = get_conversation_analysis_prompt(
                "determine_next_action",
                conversation_purpose=context.conversation_purpose,
                conversation_on_track_confidence=context.conversation_on_track_confidence,
                conversation_style=get_enum_value(context.conversation_style),
                information_density=context.information_density,
                conversation_depth=get_enum_value(context.conversation_depth),
                user_confidence=get_enum_value(context.user_confidence),
                conversation_flow=get_enum_value(context.conversation_flow),
                sentiment=get_enum_value(context.last_message_sentiment),
                momentum=get_enum_value(context.conversation_momentum),
                missing_info=missing_info,
                available_actions=allowed,
                ready_to_summarize=ready_to_summarize,
                recent_messages="\n".join(recent_messages[-3:]),  # Last 3 messages
            )

            # Use structured completion with retry for reliable unified analysis
            result: NextActionResponse = self.llm.structured_completion(
                messages=[{"role": "user", "content": prompt}],
                response_model=NextActionResponse,
                temperature=0.3,
                max_tokens=300,
            )

            # Validate the response
            if result.next_action not in allowed:
                raise ValueError(f"LLM returned invalid action: {result.next_action}")
            return result

        except Exception as e:
            logger.warning(f"Unified next action determination failed: {e}, falling back to heuristics")
            # Fallback to heuristic approach
            next_action = self._get_heuristic_next_action(context, context.missing_info)
            return NextActionResponse(
                intent_type="task",
                next_action=next_action or "summarize",
                reasoning=f"Heuristic fallback - error: {str(e)}",
                confidence=0.5,
                is_smalltalk=False,
            )

    def _get_heuristic_next_action(self, context: ConversationContext, missing_info: List[str]) -> Optional[str]:
        """Get next action using heuristic approach as fallback."""

        if not missing_info:
            return "summarize"

        # If conversation is off-track, prioritize redirecting
        if context.conversation_on_track_confidence < 0.4:
            return "redirect_conversation"

        # If conversation is highly on-track, focus on gathering missing info
        if context.conversation_on_track_confidence > 0.7:
            # Prioritize based on conversation context
            if context.conversation_style == ConversationStyle.DIRECT:
                # Pick randomly from missing info instead of always first
                return random.choice(missing_info) if missing_info else None
            elif context.conversation_style == ConversationStyle.EXPLORATORY:
                # For exploratory users, suggest topics they might be interested in
                return random.choice(missing_info) if missing_info else None
            elif context.conversation_style == ConversationStyle.CASUAL:
                # For casual users, ask easy questions first
                easy_questions = self._get_easy_questions()
                for question in easy_questions:
                    if question in missing_info:
                        return question
                # If no easy questions found, pick randomly from missing info
                return random.choice(missing_info) if missing_info else None

        # For moderate alignment, balance between staying on track and gathering info
        # Pick randomly from missing info
        return random.choice(missing_info) if missing_info else None

    def _detect_conversation_style(self, last_user_msg: str, user_messages: List[HumanMessage]) -> Dict[str, Any]:
        """Detect user's conversation style based on message patterns."""
        style_info = {"conversation_style": ConversationStyle.DIRECT, "information_density": 0.0}

        # Analyze message length and complexity
        avg_length = sum(len(msg.content.split()) for msg in user_messages) / len(user_messages)

        # Direct style indicators
        direct_indicators = ["yes", "no", "ok", "sure", "fine", "whatever", "i guess"]
        if any(word in last_user_msg for word in direct_indicators):
            style_info["conversation_style"] = ConversationStyle.DIRECT
            style_info["information_density"] = 0.3
        # Exploratory style indicators
        elif avg_length > 25 or any(
            word in last_user_msg for word in ["tell me more", "what about", "how about", "i wonder"]
        ):
            style_info["conversation_style"] = ConversationStyle.EXPLORATORY
            style_info["information_density"] = 0.8
        # Casual style indicators
        elif any(word in last_user_msg for word in ["maybe", "not sure", "i think", "probably", "kind of"]):
            style_info["conversation_style"] = ConversationStyle.CASUAL
            style_info["information_density"] = 0.5

        return style_info

    def _analyze_sentiment_and_momentum(self, last_user_msg: str, user_messages: List[HumanMessage]) -> Dict[str, Any]:
        """Analyze sentiment and conversation momentum."""
        sentiment_info = {
            "last_message_sentiment": ConversationSentiment.NEUTRAL,
            "conversation_momentum": ConversationMomentum.POSITIVE,
        }

        # Enhanced sentiment analysis
        positive_words = [
            "love",
            "excited",
            "great",
            "perfect",
            "amazing",
            "wonderful",
            "fantastic",
            "awesome",
            "brilliant",
        ]
        negative_words = [
            "hate",
            "dislike",
            "boring",
            "expensive",
            "difficult",
            "problem",
            "annoying",
            "frustrated",
            "confused",
        ]
        uncertainty_words = ["maybe", "not sure", "i think", "probably", "possibly", "perhaps"]

        # Count sentiment indicators in recent messages
        positive_count = sum(1 for msg in user_messages[-3:] for word in positive_words if word in msg.content.lower())
        negative_count = sum(1 for msg in user_messages[-3:] for word in negative_words if word in msg.content.lower())
        uncertainty_count = sum(
            1 for msg in user_messages[-3:] for word in uncertainty_words if word in msg.content.lower()
        )

        if positive_count > negative_count:
            sentiment_info["last_message_sentiment"] = ConversationSentiment.POSITIVE
            sentiment_info["conversation_momentum"] = ConversationMomentum.POSITIVE
        elif negative_count > positive_count:
            sentiment_info["last_message_sentiment"] = ConversationSentiment.NEGATIVE
            sentiment_info["conversation_momentum"] = ConversationMomentum.NEGATIVE
        elif uncertainty_count > 0:
            sentiment_info["last_message_sentiment"] = ConversationSentiment.UNCERTAIN
            sentiment_info["conversation_momentum"] = ConversationMomentum.NEUTRAL

        return sentiment_info

    def _analyze_conversation_depth(self, last_user_msg: str, user_messages: List[HumanMessage]) -> Dict[str, Any]:
        """Analyze conversation depth and user confidence."""
        depth_info = {"conversation_depth": ConversationDepth.SURFACE, "user_confidence": UserConfidence.MEDIUM}

        # Analyze for deep conversation indicators
        deep_indicators = ["because", "since", "although", "however", "but", "actually", "really", "specifically"]
        confidence_indicators = ["definitely", "absolutely", "certainly", "for sure", "without a doubt"]
        uncertainty_indicators = ["maybe", "i think", "probably", "not sure", "i guess", "perhaps"]

        deep_count = sum(1 for msg in user_messages[-3:] for word in deep_indicators if word in msg.content.lower())
        confidence_count = sum(
            1 for msg in user_messages[-3:] for word in confidence_indicators if word in msg.content.lower()
        )
        uncertainty_count = sum(
            1 for msg in user_messages[-3:] for word in uncertainty_indicators if word in msg.content.lower()
        )

        if deep_count > 2:
            depth_info["conversation_depth"] = ConversationDepth.DEEP
        elif deep_count > 0:
            depth_info["conversation_depth"] = ConversationDepth.MODERATE

        if confidence_count > uncertainty_count:
            depth_info["user_confidence"] = UserConfidence.HIGH
        elif uncertainty_count > confidence_count:
            depth_info["user_confidence"] = UserConfidence.LOW

        return depth_info

    def _get_missing_information_prioritized(self, state: AskuraState) -> List[str]:
        """Get prioritized list of missing information based on importance and context."""
        missing = []
        information_slots = state.information_slots

        # Sort slots by priority (higher priority first)
        sorted_slots = sorted(self.config.information_slots, key=lambda slot: slot.priority, reverse=True)

        for slot in sorted_slots:
            if slot.required and not information_slots.get(slot.name):
                missing.append(f"ask_{slot.name}")

        return missing

    def _get_confidence_boosting_action(self, context: ConversationContext) -> str:
        """Get an action that will boost user confidence."""
        # Ask easy, non-threatening questions first
        easy_questions = self._get_easy_questions()
        missing_info = context.missing_info

        for question in easy_questions:
            if question in missing_info:
                return question

        return missing_info[0] if missing_info else "redirect_conversation"

    def _get_easy_questions(self) -> List[str]:
        """Get list of easy questions that boost confidence."""
        easy_questions = []
        for slot in self.config.information_slots:
            # Consider questions about preferences and interests as "easy"
            if any(word in slot.name.lower() for word in ["interest", "preference", "like", "favorite"]):
                easy_questions.append(f"ask_{slot.name}")
        return easy_questions
