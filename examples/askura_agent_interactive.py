#!/usr/bin/env python3
"""
Interactive AskuraAgent demo (LLM-enhanced, HITL).

Run this script to chat with Askura as a trip planner.
Type 'exit' or 'quit' to stop.
"""
import sys
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


from tools.trip_extraction import (
    extract_budget_info,
    extract_date_info,
    extract_destination_info,
    extract_group_info,
    extract_interests_info,
)

from cogents.agents.askura_agent import AskuraAgent
from cogents.agents.askura_agent.schemas import AskuraConfig, InformationSlot


def create_travel_planning_config() -> AskuraConfig:
    """Create configuration for a simple trip planning scenario."""
    information_slots = [
        InformationSlot(
            name="destination",
            description="Travel destination",
            priority=5,
            required=True,
            extraction_tools=["extract_destination_info"],
        ),
        InformationSlot(
            name="dates",
            description="Travel dates",
            priority=4,
            required=True,
            extraction_tools=["extract_date_info"],
        ),
        InformationSlot(
            name="budget",
            description="Travel budget",
            priority=3,
            required=True,
            extraction_tools=["extract_budget_info"],
        ),
        InformationSlot(
            name="interests",
            description="Travel interests",
            priority=2,
            required=False,
            extraction_tools=["extract_interests_info"],
        ),
        InformationSlot(
            name="group",
            description="Group info (size, composition)",
            priority=1,
            required=False,
            extraction_tools=["extract_group_info"],
        ),
    ]

    return AskuraConfig(
        model_name="gpt-4o-mini",
        temperature=0.7,
        max_tokens=1000,
        max_conversation_turns=12,
        information_slots=information_slots,
        conversation_purposes=["collect user information about the next planned trip"],
        custom_config={"use_case": "travel_planning"},
    )


def interactive_loop() -> None:
    """Run a blocking interactive chat loop with Askura."""
    print("\n" + "=" * 80)
    print("AskuraAgent Interactive Demo (Trip Planner)")
    print("Type 'exit' or 'quit' to end the session.")
    print("=" * 80)

    config = create_travel_planning_config()
    extraction_tools = {
        "extract_destination_info": extract_destination_info,
        "extract_date_info": extract_date_info,
        "extract_budget_info": extract_budget_info,
        "extract_interests_info": extract_interests_info,
        "extract_group_info": extract_group_info,
    }
    agent = AskuraAgent(config=config, extraction_tools=extraction_tools)

    # Start the conversation
    try:
        initial_user_text = input("You (initial): ").strip()
    except (EOFError, KeyboardInterrupt):
        print("\nGoodbye!")
        return

    if initial_user_text.lower() in {"exit", "quit", "q"}:
        print("Goodbye!")
        return

    response = agent.start_conversation(user_id="interactive_user", initial_message=initial_user_text)
    session_id = response.session_id

    print(f"\nAskura: {response.message}")
    if response.requires_user_input:
        print("(Narrator) Awaiting your input...")

    while not response.is_complete:
        try:
            user_text = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nEnding session.")
            break

        if user_text.lower() in {"exit", "quit", "q"}:
            print("Session ended by user.")
            break

        response = agent.process_user_message(
            user_id="interactive_user",
            session_id=session_id,
            message=user_text,
        )

        print(f"Askura: {response.message}")
        if response.requires_user_input and not response.is_complete:
            print("(Narrator) Awaiting your input...")

        if response.is_complete:
            print(f"\n✅ Conversation completed. Confidence: {response.confidence:.2f}")
            break

    # Optionally print a compact summary of gathered info
    slots = response.metadata.get("information_slots", {}) if response.metadata else {}
    if slots:
        print("\nCollected information:")
        for slot_name, value in slots.items():
            print(f"- {slot_name}: {value}")


def main() -> None:
    interactive_loop()


if __name__ == "__main__":
    main()
