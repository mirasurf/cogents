"""
AskuraAgent - A general-purpose dynamic conversation agent.

AskuraAgent provides a flexible, configurable framework for human-in-the-loop
conversations that adapt to different user communication styles and dynamically
collect required information through natural conversation flow.
"""
import os
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

from .conversation_manager import ConversationManager
from .information_extractor import InformationExtractor
from .models import AskuraConfig, AskuraResponse, AskuraState
from .nodes import AskuraNodes

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
        self.nodes = AskuraNodes(
            config=config,
            conversation_manager=self.conversation_manager,
            information_extractor=self.information_extractor,
            llm_client=self.llm,
        )

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
            chat_context={},
            extracted_slots={},
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

    def _run_graph(self, state: AskuraState) -> tuple[AskuraResponse, AskuraState]:
        """Run the conversation graph with the given state."""
        try:
            # Run the graph with per-session thread_id for checkpoints
            config = RunnableConfig(
                configurable={"thread_id": state.session_id},
                recursion_limit=15,
                callbacks=[NodeLoggingCallback(node_id="graph"), TokenUsageCallback()],
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

        # Add nodes (delegated to AskuraNodes)
        builder.add_node("conversation_context_analysis", self.nodes.conversation_context_analysis_node)
        builder.add_node("determine_next_action", self.nodes.determine_next_action_node)
        builder.add_node("information_extractor", self.nodes.information_extractor_node)
        builder.add_node("question_generator", self.nodes.question_generator_node)
        builder.add_node("completeness_evaluator", self.nodes.completeness_evaluator_node)
        builder.add_node("human_review", self.nodes.human_review_node)
        builder.add_node("summarizer", self.nodes.summarizer_node)

        # Entry: analyze context first, then decide action
        builder.add_edge(START, "conversation_context_analysis")
        builder.add_edge("conversation_context_analysis", "determine_next_action")

        # Conditional routing from determining next action
        builder.add_conditional_edges(
            "determine_next_action",
            self.nodes.determine_next_action_router,
            {
                "question_generator": "question_generator",
                "information_extractor": "information_extractor",
                "completeness_evaluator": "completeness_evaluator",
                "human_review": "human_review",
                "summarizer": "summarizer",
                "end": END,
            },
        )

        # After question generation, go to human review (wait for user)
        builder.add_edge("question_generator", "human_review")

        # After information extraction, evaluate completeness, then decide next action
        builder.add_edge("information_extractor", "completeness_evaluator")
        builder.add_edge("completeness_evaluator", "determine_next_action")

        # Human review routing
        builder.add_conditional_edges(
            "human_review",
            self.nodes.human_review_router,
            {
                "continue": "conversation_context_analysis",
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
            graph_structure.draw_png(os.path.join(os.path.dirname(__file__), "askura_agent_graph.png"))
            logger.info("Graph exported successfully to askura_agent_graph.png")
        except Exception as e:
            logger.error(f"Failed to export graph: {e}")
            try:
                graph_structure = self.graph.get_graph()
                graph_structure.draw_mermaid_png("askura_agent_graph.png")
                logger.info("Graph exported successfully using Mermaid fallback")
            except Exception as fallback_error:
                logger.error(f"Failed to export graph with fallback: {fallback_error}")

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
            next_actions=[state.next_action_plan.next_action] if state.next_action_plan else [],
            requires_user_input=state.requires_user_input,
            metadata={
                "turns": state.turns,
                "conversation_context": state.chat_context,
                "information_slots": state.extracted_slots,
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
        information_slots = state.extracted_slots

        # Count filled slots
        filled_slots = sum(1 for slot in self.config.information_slots if information_slots.get(slot.name))
        total_slots = len(self.config.information_slots)

        if total_slots == 0:
            return 1.0

        return min(filled_slots / total_slots, 1.0)

    def _now_iso(self) -> str:
        """Get current time in ISO format."""
        return datetime.utcnow().isoformat()
