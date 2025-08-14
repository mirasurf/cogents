"""
AskuraAgent - A general-purpose dynamic conversation agent.

AskuraAgent provides a flexible, configurable framework for human-in-the-loop
conversations that adapt to different user communication styles and dynamically
collect required information through natural conversation flow.
"""
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph
from langgraph.graph.message import add_messages

from cogents.common.lg_hooks import NodeLoggingCallback, TokenUsageCallback
from cogents.common.llm import get_llm_client_instructor
from cogents.common.logging import get_logger
from cogents.common.utils import get_enum_value

from .conversation_manager import ConversationManager
from .information_extractor import InformationExtractor
from .question_generator import QuestionGenerator
from .schemas import AskuraConfig, AskuraResponse, AskuraState

logger = get_logger(__name__)


class AskuraAgent:
    """
    A general-purpose dynamic conversation agent.

    AskuraAgent provides a flexible, configurable framework for human-in-the-loop
    conversations that adapt to different user communication styles and dynamically
    collect required information through natural conversation flow.
    """

    def __init__(self, config: AskuraConfig, extraction_tools: Optional[Dict[str, Any]] = None):
        """Initialize the AskuraAgent."""
        self.config = config
        self.extraction_tools = extraction_tools or {}
        self.memory = MemorySaver()

        # Initialize LLM client (optional)
        self.llm = get_llm_client_instructor(provider=config.llm_api_provider, chat_model=config.model_name)

        # Initialize components (pass LLM client to enable intelligent behavior)
        self.conversation_manager = ConversationManager(config, llm_client=self.llm)
        self.information_extractor = InformationExtractor(config, self.extraction_tools, llm_client=self.llm)
        self.question_generator = QuestionGenerator(config, llm_client=self.llm)

        # Build the conversation graph
        self.graph = self._build_conversation_graph()
        self._export_graph()

        # Session storage
        self._session_states: Dict[str, AskuraState] = {}

    def start_conversation(self, user_id: str, initial_message: Optional[str] = None) -> AskuraResponse:
        """Start a new conversation with a user."""
        session_id = str(uuid.uuid4())
        now = self._now_iso()

        # Create initial state
        state = AskuraState(
            user_id=user_id,
            session_id=session_id,
            messages=[],
            conversation_context={},
            extracted_information_slots={},
            turns=0,
            created_at=now,
            updated_at=now,
            next_action=None,
            requires_user_input=False,
            is_complete=False,
            custom_data={},
        )

        # Add initial message if provided
        if initial_message:
            user_msg = HumanMessage(content=initial_message)
            state.messages = add_messages(state.messages, [user_msg])

        # Store state
        self._session_states[session_id] = state

        # Run the graph to get initial response
        response, updated_state = self._run_graph(state)

        # Update stored state with the updated state from graph execution
        self._session_states[session_id] = updated_state

        logger.info(f"Started conversation for user {user_id}, session {session_id}")
        return response

    def process_user_message(self, user_id: str, session_id: str, message: str) -> AskuraResponse:
        """Process a user message and return the agent's response."""

        # Get the current state
        state = self._session_states.get(session_id)
        if not state:
            raise ValueError(f"Session {session_id} not found")

        # Add user message to state
        user_msg = HumanMessage(content=message)
        state.messages = add_messages(state.messages, [user_msg])
        state.updated_at = self._now_iso()
        # Ensure we prioritize extraction on the next turn to avoid loops
        state.pending_extraction = True

        # Run the graph to process the message
        response, updated_state = self._run_graph(state)

        # Update stored state with the updated state from graph execution
        self._session_states[session_id] = updated_state

        return response

    def _run_graph(self, state: AskuraState) -> tuple[AskuraResponse, AskuraState]:
        """Run the conversation graph with the given state."""
        try:
            # Run the graph with per-session thread_id for checkpoints
            config = RunnableConfig(
                configurable={"thread_id": state.session_id},
                recursion_limit=15,
                callbacks=[NodeLoggingCallback(), TokenUsageCallback()],
            )
            result = self.graph.invoke(state, config)

            # Convert result back to AskuraState if it's a dict
            if isinstance(result, dict):
                result = AskuraState(**result)

            # Create response from final state
            return self._create_response(result), result

        except Exception as e:
            logger.error(f"Error running AskuraAgent graph: {e}")
            return self._create_error_response(state, str(e)), state

    def _build_conversation_graph(self) -> StateGraph:
        """Build the conversation graph."""
        builder = StateGraph(AskuraState)

        # Add nodes
        builder.add_node("conversation_manager", self._conversation_manager_node)
        builder.add_node("information_extractor", self._information_extractor_node)
        builder.add_node("question_generator", self._question_generator_node)
        builder.add_node("human_review", self._human_review_node)
        builder.add_node("summarizer", self._summarizer_node)

        # Entry
        builder.add_edge(START, "conversation_manager")

        # Conditional routing from conversation manager
        builder.add_conditional_edges(
            "conversation_manager",
            self._conversation_router,
            {
                "question_generator": "question_generator",
                "information_extractor": "information_extractor",
                "human_review": "human_review",
                "summarizer": "summarizer",
                "end": END,
            },
        )

        # After question generation, go to human review (HITL)
        builder.add_edge("question_generator", "human_review")

        # After information extraction, loop back to conversation manager
        builder.add_edge("information_extractor", "conversation_manager")

        # Human review routing
        builder.add_conditional_edges(
            "human_review",
            self._human_review_router,
            {
                "continue": "conversation_manager",
                "end": END,
            },
        )

        # Summarizer ends the conversation
        builder.add_edge("summarizer", END)

        return builder.compile(checkpointer=self.memory, interrupt_before=["human_review"])

    def _export_graph(self):
        """Export the agent graph visualization to PNG format."""
        try:
            pass
        except ImportError:
            logger.debug("pygraphviz is not installed, skipping graph export")
            return

        try:
            graph_structure = self.graph.get_graph()
            graph_structure.draw_png("askura_agent_graph.png")
            logger.info("Graph exported successfully to askura_agent_graph.png")
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            try:
                graph_structure = self.graph.get_graph()
                graph_structure.draw_mermaid_png("askura_agent_graph.png")
                logger.info("Graph exported successfully using Mermaid fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to export graph with fallback: {fallback_error}")

    def _conversation_manager_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        """Manage conversation flow and determine next actions dynamically using unified LLM approach."""

        # Analyze current conversation context
        # TODO (xmingc): conversation context analysis could be async and occasional.
        logger.info("ConversationManager: Analyzing conversation context")
        conversation_context = self.conversation_manager.analyze_conversation_context(state)
        state.conversation_context = conversation_context

        # Get recent user messages for unified analysis
        # TODO (xmingc): combine context and memory information.
        recent_user_messages = [m.content for m in state.messages if isinstance(m, HumanMessage)][-3:]

        # Use unified method for intent classification and next action determination
        logger.info("ConversationManager: Determining next action")
        action_result = self.conversation_manager.determine_next_action(
            state=state,
            context=conversation_context,
            recent_messages=recent_user_messages,
            ready_to_summarize=self._ready_to_summarize(state),
        )

        # Set the next action based on unified result
        state.next_action_response = action_result
        state.turns += 1
        logger.info(
            f"Next action: {action_result.next_action} "
            f"(intent: {action_result.intent_type}, confidence: {action_result.confidence})"
        )
        return state

    def _information_extractor_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        """Extract structured information from user messages."""
        logger.info("InformationExtractor: Processing user input dynamically")

        if not state.messages:
            logger.warning("InformationExtractor: No messages to extract information from")
            return state

        # Get the last user message
        last_user_msg = next((msg for msg in reversed(state.messages) if isinstance(msg, HumanMessage)), None)

        if not last_user_msg:
            logger.warning("InformationExtractor: No last user message to extract information from")
            return state

        extracted_info = self.information_extractor.extract_all_information(last_user_msg.content, state)
        # TODO (xmingc): polish extracted info with LLM.
        state = self.information_extractor.update_state_with_extracted_info(state, extracted_info)

        # Clear the pending_extraction flag after processing
        state.pending_extraction = False

        return state

    def _question_generator_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        """Generate contextual questions based on conversation state."""
        logger.info("QuestionGenerator: Generating contextual question")

        next_action = state.next_action_response.next_action
        conversation_context = state.conversation_context

        # Handle smalltalk reply entirely via LLM (no template)
        if next_action == "reply_smalltalk":
            reply = None
            try:
                if self.llm is not None:
                    # Use last user message as context
                    last_user_msg = next(
                        (m for m in reversed(state.messages) if isinstance(m, HumanMessage)),
                        None,
                    )
                    content = last_user_msg.content if last_user_msg else "how are you"
                    reply = self.llm.chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": "Be friendly and concise. Briefly answer the greeting and pivot to travel planning with one short question.",
                            },
                            {"role": "user", "content": content},
                        ],
                        temperature=0.5,
                        max_tokens=64,
                    )
            except Exception:
                reply = None

            # If LLM fails, fallback softly with a minimal pivot only
            if not reply or not isinstance(reply, str) or len(reply.strip()) < 3:
                reply = "What kind of trip are you thinking about right now?"

            ai_message = AIMessage(content=reply)
            state.messages = add_messages(state.messages, [ai_message])
            state.requires_user_input = True
            return state

        # Generate contextual question if needed
        if next_action and next_action.startswith("ask_"):
            # LLM-first: generate the question creatively with guidance; fallback to template generator
            question = None
            try:
                if self.llm is not None:
                    # Compose a brief instruction with state context to avoid rigid repetition
                    info = state.extracted_information_slots
                    guidance = (
                        "Ask one natural, varied question for the missing info. Avoid repeating previous phrasing."
                    )

                    user_ctx = {
                        "next_action": next_action,
                        "known": info,
                        "style": get_enum_value(conversation_context.conversation_style),
                    }
                    msg = [
                        {"role": "system", "content": guidance},
                        {"role": "user", "content": str(user_ctx)},
                    ]
                    candidate = self.llm.chat_completion(messages=msg, temperature=0.7, max_tokens=64)
                    if isinstance(candidate, str) and len(candidate.strip()) > 3:
                        question = candidate.strip()
            except Exception:
                question = None

            if not question:
                question = self.question_generator.generate_contextual_question(
                    next_action, state, conversation_context
                )

            # Optional LLM polishing
            try:
                if self.llm is not None and question:
                    refine = self.llm.chat_completion(
                        messages=[
                            {
                                "role": "system",
                                "content": "Refine the user-facing question to be concise and friendly.",
                            },
                            {"role": "user", "content": question},
                        ],
                        temperature=0.3,
                        max_tokens=64,
                    )
                    if isinstance(refine, str) and len(refine.strip()) >= 5:
                        question = refine.strip()
            except Exception:
                pass

            if question:
                ai_message = AIMessage(content=question)
                state.messages = add_messages(state.messages, [ai_message])
                state.requires_user_input = True
        else:
            # If no specific action, generate a general follow-up question via generator
            general_question = self.question_generator.generate_contextual_question(
                "redirect_conversation",
                state,
                conversation_context,
            )
            ai_message = AIMessage(content=general_question)
            state.messages = add_messages(state.messages, [ai_message])
            state.requires_user_input = True

        return state

    def _summarizer_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        """Generate final summary and mark conversation as complete."""
        logger.info("Summarizer: Generating conversation summary")

        # Check if we have enough information to complete
        if self._ready_to_summarize(state):
            summary = self._generate_summary(state)
            summary_message = AIMessage(content=summary)
            state.messages = add_messages(state.messages, [summary_message])
            state.is_complete = True
            state.requires_user_input = False
        else:
            # If not ready to summarize, continue the conversation
            state.requires_user_input = True

        return state

    # TODO (xmingc): conversation router is important for a fluent chat.
    # Optimize the router to be more logical, more efficient, and not too rigid.
    def _conversation_router(self, state: AskuraState) -> str:
        """Route from conversation manager based on next action and flags."""
        next_action = state.next_action_response.next_action

        # If a user message just arrived, prioritize extraction before asking again
        if state.pending_extraction:
            return "information_extractor"

        # Check if we need to summarize or if conversation is complete
        if next_action == "summarize" or state.is_complete:
            return "summarizer"

        # If we need to ask for more information, route to question generator
        if next_action and (next_action == "reply_smalltalk" or next_action.startswith("ask_")):
            return "question_generator"

        # If we need user input, route to human review
        if state.requires_user_input:
            return "human_review"

        # Check if we've reached max turns
        if state.turns >= self.config.max_conversation_turns:
            return "summarizer"

        # Default: if we have incomplete information, route to question generator
        # This prevents infinite loops and ensures we ask for missing information
        if not self._ready_to_summarize(state):
            return "question_generator"

        # If all information is complete, route to summarizer
        return "summarizer"

    def _human_review_node(self, state: AskuraState, config: RunnableConfig) -> AskuraState:
        """Human-in-the-loop review node (interrupted before execution)."""
        logger.info("HumanReview: Awaiting human input")

        # Mark waiting state; interrupt occurs before execution
        if state.requires_user_input:
            state.requires_user_input = True

        # When resumed, clear flag and mark extraction needed
        state.requires_user_input = False
        state.pending_extraction = True
        state.updated_at = self._now_iso()
        return state

    def _human_review_router(self, state: AskuraState) -> str:
        """Route after human review."""
        if state.is_complete:
            return "end"
        return "continue"

    def _ready_to_summarize(self, state: AskuraState) -> bool:
        """Check if we have enough information to summarize."""
        information_slots = state.extracted_information_slots

        # Check if all required slots are filled
        required_slots_filled = True
        for slot in self.config.information_slots:
            if slot.required and not information_slots.get(slot.name):
                required_slots_filled = False
                break

        # Also check if we have at least some information and the conversation has progressed
        has_some_info = len(information_slots) > 0
        conversation_progressed = state.turns > 1

        return required_slots_filled and has_some_info and conversation_progressed

    def _generate_summary(self, state: AskuraState) -> str:
        """Generate a summary of the collected information."""
        information_slots = state.extracted_information_slots

        summary_parts = []
        for slot in self.config.information_slots:
            if information_slots.get(slot.name):
                summary_parts.append(f"{slot.name}: {information_slots[slot.name]}")

        if summary_parts:
            return "Summary: " + " | ".join(summary_parts)
        else:
            return "Conversation completed."

    def _create_response(self, state: AskuraState) -> AskuraResponse:
        """Create response from final state."""
        # Get last assistant message
        last_message = None
        for msg in reversed(state.messages):
            if isinstance(msg, AIMessage):
                last_message = msg.content
                break

        return AskuraResponse(
            message=last_message or "I'm here to help!",
            session_id=state.session_id,
            is_complete=state.is_complete,
            confidence=self._calculate_confidence(state),
            next_actions=[state.next_action_response.next_action] if state.next_action_response else [],
            requires_user_input=state.requires_user_input,
            metadata={
                "turns": state.turns,
                "conversation_context": state.conversation_context,
                "information_slots": state.extracted_information_slots,
            },
            custom_data=state.custom_data,
        )

    def _create_error_response(self, state: AskuraState, error_message: str) -> AskuraResponse:
        """Create error response."""
        return AskuraResponse(
            message=f"I encountered an issue while processing your request. Please try again. Error: {error_message}",
            session_id=state.session_id,
            is_complete=False,
            confidence=0.0,
            metadata={"error": error_message},
            requires_user_input=True,
        )

    def _calculate_confidence(self, state: AskuraState) -> float:
        """Calculate confidence score based on gathered information."""
        information_slots = state.extracted_information_slots

        # Count filled slots
        filled_slots = sum(1 for slot in self.config.information_slots if information_slots.get(slot.name))
        total_slots = len(self.config.information_slots)

        if total_slots == 0:
            return 1.0

        return min(filled_slots / total_slots, 1.0)

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        return datetime.utcnow().isoformat()

    def get_session_state(self, session_id: str) -> Optional[AskuraState]:
        """Get the state for a specific session."""
        return self._session_states.get(session_id)

    def list_sessions(self) -> List[str]:
        """List all active session IDs."""
        return list(self._session_states.keys())

    def clear_session(self, session_id: str) -> bool:
        """Clear a specific session."""
        if session_id in self._session_states:
            del self._session_states[session_id]
            return True
        return False
