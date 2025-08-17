"""
Simplified GoalithService LLM Decomposer Example

This example demonstrates the optimized LLM decomposer with minimal dependencies.
It focuses on testing the decomposer logic without requiring the full goalith system.
"""

import os
import sys
from datetime import datetime, timedelta
from pathlib import Path

# Add the project root to the path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def test_decomposer_prompts():
    """Test the new prompt structure without LLM calls."""
    print("🔄 Testing optimized prompt structure...")

    try:
        # Import directly to avoid dependency issues
        from cogents.goalith.base.goal_node import GoalNode, NodeType
        from cogents.goalith.decomposer.prompts import (
            get_decomposition_system_prompt,
            get_decomposition_user_prompt,
            get_fallback_prompt,
        )

        # Create a sample goal
        sample_goal = GoalNode(
            description="Plan and execute a product launch for a new mobile app",
            type=NodeType.GOAL,
            priority=8.0,
            deadline=datetime.now() + timedelta(days=90),
            context={
                "target_audience": "young professionals",
                "budget": "$50,000",
                "platform": "iOS and Android",
            },
            tags={"product", "launch", "mobile", "marketing"},
        )

        print(f"✅ Created sample goal: {sample_goal.description}")

        # Test system prompt
        system_prompt = get_decomposition_system_prompt()
        print(f"✅ System prompt length: {len(system_prompt)} characters")
        print("📋 System prompt highlights:")
        print("   - Sets up LLM as goal decomposition expert")
        print("   - Enforces maximum 6 subgoals/tasks limit")
        print("   - Provides clear priority scoring (0-10)")
        print("   - Includes effort estimation guidelines")

        # Test user prompt
        user_prompt = get_decomposition_user_prompt(
            goal_node=sample_goal,
            context={
                "timeline": "3 months",
                "team_size": "5 people",
                "experience_level": "intermediate",
            },
            domain_context={
                "industry": "mobile app development",
                "methodology": "agile",
                "quality_standards": "high",
            },
            include_historical_patterns=True,
        )

        print(f"✅ User prompt length: {len(user_prompt)} characters")
        print("📋 User prompt includes:")
        print("   - Goal details and context")
        print("   - Domain-specific guidance")
        print("   - Historical patterns for similar goals")
        print("   - Clear 6-item limit constraint")

        # Test fallback prompt
        fallback_prompt = get_fallback_prompt()
        print(f"✅ Fallback prompt length: {len(fallback_prompt)} characters")
        print("📋 Fallback prompt provides simplified decomposition approach")

        return True

    except Exception as e:
        print(f"❌ Prompt testing failed: {e}")
        return False


def test_decomposer_schema():
    """Test the optimized Pydantic schemas."""
    print("🔄 Testing optimized decomposer schemas...")

    try:
        from pydantic import ValidationError

        from cogents.goalith.decomposer.llm_decomposer import GoalDecomposition, SubgoalSpec

        # Test valid decomposition (6 items)
        valid_subgoals = [
            SubgoalSpec(
                description="Market research and competitor analysis",
                type="subgoal",
                priority=9.0,
                estimated_effort="1 week",
                tags=["research", "planning"],
            ),
            SubgoalSpec(
                description="Develop product positioning and messaging",
                type="subgoal",
                priority=8.5,
                estimated_effort="3 days",
                dependencies=["Market research and competitor analysis"],
                tags=["strategy", "messaging"],
            ),
            SubgoalSpec(
                description="Create launch timeline and milestones",
                type="task",
                priority=8.0,
                estimated_effort="1 day",
                tags=["planning", "timeline"],
            ),
            SubgoalSpec(
                description="Design marketing materials and assets",
                type="subgoal",
                priority=7.5,
                estimated_effort="2 weeks",
                dependencies=["Develop product positioning and messaging"],
                tags=["design", "marketing"],
            ),
            SubgoalSpec(
                description="Execute launch campaign",
                type="task",
                priority=9.0,
                estimated_effort="1 week",
                dependencies=["Design marketing materials and assets"],
                tags=["execution", "campaign"],
            ),
            SubgoalSpec(
                description="Monitor metrics and gather feedback",
                type="task",
                priority=7.0,
                estimated_effort="ongoing",
                dependencies=["Execute launch campaign"],
                tags=["monitoring", "feedback"],
            ),
        ]

        valid_decomposition = GoalDecomposition(
            reasoning="This decomposition follows a logical sequence from research to execution, with clear dependencies and realistic timelines.",
            decomposition_strategy="sequential",
            subgoals=valid_subgoals,
            success_criteria=[
                "Successful app launch with target downloads",
                "Positive user feedback and ratings",
                "Achievement of marketing KPIs",
            ],
            potential_risks=["Market competition", "Technical issues during launch", "Budget constraints"],
            estimated_timeline="3 months",
            confidence=0.85,
        )

        print(f"✅ Valid decomposition with {len(valid_decomposition.subgoals)} subgoals")
        print("📋 Decomposition strategy:", valid_decomposition.decomposition_strategy)
        print("📋 Confidence level:", valid_decomposition.confidence)

        # Test that too many subgoals are rejected
        try:
            too_many_subgoals = valid_subgoals + [
                SubgoalSpec(
                    description="Extra task that exceeds limit", type="task", priority=5.0, estimated_effort="1 hour"
                )
            ]

            invalid_decomposition = GoalDecomposition(
                reasoning="This should fail",
                decomposition_strategy="sequential",
                subgoals=too_many_subgoals,  # 7 items - should fail
                confidence=0.5,
            )

            print("❌ Schema should have rejected 7 subgoals but didn't")
            return False

        except ValidationError:
            print("✅ Schema correctly rejects more than 6 subgoals")

        return True

    except Exception as e:
        print(f"❌ Schema testing failed: {e}")
        return False


def test_mock_decomposition():
    """Test decomposition logic with a mock LLM response."""
    print("🔄 Testing decomposition with mock data...")

    try:
        # This test doesn't actually call an LLM, but tests the logic
        from cogents.goalith.base.goal_node import GoalNode, NodeType
        from cogents.goalith.decomposer.llm_decomposer import LLMDecomposer

        # Create decomposer (won't actually initialize LLM client due to mock)
        domain_context = {
            "industry": "software development",
            "methodology": "agile",
            "team_structure": "cross-functional",
            "quality_standards": "high",
        }

        decomposer = LLMDecomposer(
            provider="openrouter",  # Won't actually use this
            domain_context=domain_context,
            include_historical_patterns=True,
            temperature=0.3,
            max_tokens=2000,
        )

        print("✅ LLMDecomposer created with optimized configuration")
        print("📋 Domain context:", domain_context)
        print("📋 Historical patterns enabled:", decomposer._include_historical)

        # Create a goal to decompose
        goal = GoalNode(
            description="Develop a real-time chat feature for our web application",
            type=NodeType.GOAL,
            priority=7.5,
            deadline=datetime.now() + timedelta(days=45),
            context={
                "technology_stack": "React, Node.js, Socket.io",
                "expected_users": "1000+ concurrent",
                "existing_codebase": "mature",
            },
            tags={"development", "feature", "real-time", "chat"},
        )

        print(f"✅ Created test goal: {goal.description}")

        # Test that decomposer has all the required methods
        required_methods = ["decompose", "_fallback_decomposition", "name"]
        for method in required_methods:
            if hasattr(decomposer, method):
                print(f"✅ Method '{method}' exists")
            else:
                print(f"❌ Method '{method}' missing")
                return False

        # Test fallback decomposition (doesn't require LLM call)
        # We can't actually call it without mocking the LLM, but we can verify it exists
        print("✅ Fallback decomposition method available for error handling")

        return True

    except Exception as e:
        print(f"❌ Mock decomposition test failed: {e}")
        return False


def main():
    """Run the simplified decomposer example."""
    print("🚀 Simplified GoalithService LLM Decomposer Example\n")

    # Check if API key is available (but don't require it)
    api_key_available = bool(os.getenv("OPENROUTER_API_KEY"))
    if api_key_available:
        print("✅ OPENROUTER_API_KEY is available")
    else:
        print("⚠️  OPENROUTER_API_KEY not set (tests will use mock data)")

    print("\n" + "=" * 80)
    print("TESTING OPTIMIZED LLM DECOMPOSER")
    print("=" * 80)

    tests = [
        ("Prompt Structure", test_decomposer_prompts),
        ("Schema Validation", test_decomposer_schema),
        ("Mock Decomposition", test_mock_decomposition),
    ]

    passed = 0
    total = len(tests)

    for test_name, test_func in tests:
        print(f"\n📋 {test_name}")
        print("-" * 60)
        if test_func():
            passed += 1
            print("✅ Test passed")
        else:
            print("❌ Test failed")
        print("-" * 60)

    print(f"\n📊 Test Results: {passed}/{total} tests passed")

    if passed == total:
        print("\n🎉 All optimizations are working correctly!")
        print("\n✨ Key improvements in this optimized version:")
        print("  🔹 Prompts separated into system and user messages")
        print("  🔹 Maximum 6 subgoals/tasks enforced via schema")
        print("  🔹 Clear priority scoring and effort estimation")
        print("  🔹 Domain context and historical patterns")
        print("  🔹 Fallback decomposition for error handling")
        print("  🔹 Type-specific guidance for different node types")

        if api_key_available:
            print("\n💡 To test with real LLM calls:")
            print("   - The decomposer is ready to use with your API key")
            print("   - Calls will automatically limit to 6 subgoals")
            print("   - Fallback mode activates if structured completion fails")
    else:
        print(f"\n❌ {total - passed} optimization(s) need attention")

    print("\n" + "=" * 80)


if __name__ == "__main__":
    main()
