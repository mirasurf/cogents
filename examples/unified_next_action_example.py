#!/usr/bin/env python3
"""
Example demonstrating unified next action determination in AskuraAgent.

This example shows how the conversation manager uses a unified LLM approach
to classify user intent and determine the optimal next action in a single call.
"""

import os
import sys

# Add the project root to the path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from langchain_core.messages import AIMessage, HumanMessage

from cogents.agents.askura_agent.conversation_manager import ConversationManager
from cogents.agents.askura_agent.schemas import AskuraConfig, AskuraState, ConversationContext


def create_sample_conversation_state(purpose: str, messages: list) -> AskuraState:
    """Create a sample conversation state for testing."""
    return AskuraState(
        session_id="test_session",
        user_id="test_user",
        messages=messages,
        information_slots={},
        conversation_purposes=[purpose],
        metadata={},
    )


def demonstrate_determine_next_action():
    """Demonstrate unified next action determination."""

    # Create a sample configuration
    config = AskuraConfig(
        conversation_purposes=["travel planning"],
        information_slots=[],
        enable_style_adaptation=True,
        enable_sentiment_analysis=True,
    )

    # Create conversation manager (without LLM for this example)
    manager = ConversationManager(config=config, llm_client=None)

    # Test case 1: Smalltalk intent
    print("=== Test Case 1: Smalltalk Intent ===")
    smalltalk_messages = [
        HumanMessage(content="Hello! How are you today?"),
        AIMessage(content="Hello! I'm doing well, thank you for asking. How can I help you today?"),
        HumanMessage(content="I'm good too, thanks!"),
    ]

    state1 = create_sample_conversation_state("travel planning", smalltalk_messages)
    context1 = ConversationContext(
        conversation_purpose="travel planning",
        conversation_on_track_confidence=0.2,  # Low confidence for smalltalk
        conversation_style="casual",
        information_density=0.3,
        missing_info=["ask_destination", "ask_dates"],
    )

    recent_messages1 = [msg.content for msg in smalltalk_messages if isinstance(msg, HumanMessage)]
    result1 = manager.determine_next_action(
        state=state1, context=context1, recent_messages=recent_messages1, ready_to_summarize=False
    )

    print(f"Intent type: {result1.intent_type}")
    print(f"Is smalltalk: {result1.is_smalltalk}")
    print(f"Next action: {result1.next_action}")
    print(f"Reasoning: {result1.reasoning}")
    print(f"Confidence: {result1.confidence}")
    print()

    # Test case 2: Task intent with high confidence
    print("=== Test Case 2: Task Intent (High Confidence) ===")
    task_messages = [
        HumanMessage(content="I want to plan a trip to Japan"),
        AIMessage(
            content="Great! I'd be happy to help you plan your trip to Japan. When are you thinking of traveling?"
        ),
        HumanMessage(
            content="I'm thinking of going in March for cherry blossom season, and I want to stay for about 10 days"
        ),
    ]

    state2 = create_sample_conversation_state("travel planning", task_messages)
    context2 = ConversationContext(
        conversation_purpose="travel planning",
        conversation_on_track_confidence=0.9,  # High confidence for on-track task
        conversation_style="direct",
        information_density=0.8,
        missing_info=["ask_budget", "ask_interests"],
    )

    recent_messages2 = [msg.content for msg in task_messages if isinstance(msg, HumanMessage)]
    result2 = manager.determine_next_action(
        state=state2, context=context2, recent_messages=recent_messages2, ready_to_summarize=False
    )

    print(f"Intent type: {result2.intent_type}")
    print(f"Is smalltalk: {result2.is_smalltalk}")
    print(f"Next action: {result2.next_action}")
    print(f"Reasoning: {result2.reasoning}")
    print(f"Confidence: {result2.confidence}")
    print()

    # Test case 3: Ready to summarize
    print("=== Test Case 3: Ready to Summarize ===")
    summary_messages = [
        HumanMessage(content="I want to plan a trip to Japan"),
        AIMessage(content="Great! When are you thinking of traveling?"),
        HumanMessage(content="March for cherry blossoms"),
        AIMessage(content="Perfect! How long would you like to stay?"),
        HumanMessage(content="10 days"),
        AIMessage(content="What's your budget range?"),
        HumanMessage(content="Around $3000"),
        AIMessage(content="What are your main interests?"),
        HumanMessage(content="Culture, food, and nature"),
    ]

    state3 = create_sample_conversation_state("travel planning", summary_messages)
    context3 = ConversationContext(
        conversation_purpose="travel planning",
        conversation_on_track_confidence=0.95,  # Very high confidence
        conversation_style="direct",
        information_density=0.9,
        missing_info=[],  # No missing info
    )

    recent_messages3 = [msg.content for msg in summary_messages if isinstance(msg, HumanMessage)]
    result3 = manager.determine_next_action(
        state=state3, context=context3, recent_messages=recent_messages3, ready_to_summarize=True
    )

    print(f"Intent type: {result3.intent_type}")
    print(f"Is smalltalk: {result3.is_smalltalk}")
    print(f"Next action: {result3.next_action}")
    print(f"Reasoning: {result3.reasoning}")
    print(f"Confidence: {result3.confidence}")
    print()


def demonstrate_benefits():
    """Demonstrate the benefits of unified next action determination."""

    print("=== Benefits of Unified Next Action Determination ===")
    print()

    print("1. **Consistency**:")
    print("   - Single LLM call ensures intent classification and action selection are consistent")
    print("   - No conflicts between separate intent and action decisions")
    print()

    print("2. **Efficiency**:")
    print("   - Reduces from 2-3 LLM calls to 1 unified call")
    print("   - Faster response times and lower costs")
    print()

    print("3. **Better Reasoning**:")
    print("   - LLM can consider intent and action together")
    print("   - More nuanced decision-making with full context")
    print()

    print("4. **Structured Output**:")
    print("   - Type-safe response with validation")
    print("   - Clear reasoning and confidence scores")
    print("   - Easy to debug and monitor")
    print()

    print("5. **Fallback Handling**:")
    print("   - Graceful degradation when LLM is unavailable")
    print("   - Heuristic fallbacks maintain functionality")
    print("   - Comprehensive error handling")
    print()


if __name__ == "__main__":
    print("Unified Next Action Determination Example")
    print("=" * 50)
    print()

    demonstrate_determine_next_action()
    demonstrate_benefits()

    print("Example completed!")
    print("\nKey improvements:")
    print("- Unified intent classification and action selection")
    print("- Single LLM call for better consistency")
    print("- Structured output with reasoning and confidence")
    print("- Comprehensive fallback handling")
    print("- Better integration with conversation purpose alignment")
