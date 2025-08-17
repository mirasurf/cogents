"""
Prompts for the LLM-based goal decomposer.

Separated into system and user messages for easier maintenance and optimization.
"""

from typing import Any, Dict, Optional

from ..base.goal_node import GoalNode, NodeType


def get_decomposition_system_prompt() -> str:
    """
    Get the system prompt for goal decomposition.

    Returns:
        System prompt that sets up the LLM as a goal decomposition expert
    """
    return """You are an expert goal decomposition assistant. Your role is to break down goals into actionable subgoals and tasks.

**KEY PRINCIPLES:**
1. **Clarity**: Each subgoal should be clear, specific, and actionable
2. **Limited Scope**: Create a maximum of 6 subgoals/tasks to maintain focus and avoid overwhelm
3. **Appropriate Granularity**: Break down into manageable chunks (not too big, not too small)
4. **Clear Dependencies**: Identify which subgoals depend on others
5. **Realistic Effort**: Estimate effort accurately based on complexity

**NODE TYPES:**
- 'task': Concrete, actionable items that can be completed in a single session
- 'subgoal': Intermediate objectives that may require multiple tasks
- 'goal': Major milestones or complex objectives

**DECOMPOSITION STRATEGIES:**
- **Sequential**: Tasks must be done in order (waterfall approach)
- **Parallel**: Tasks can be done simultaneously (parallel execution)
- **Hybrid**: Mix of sequential and parallel work streams
- **Milestone-based**: Organized around key milestones and deliverables

**PRIORITY SCORING (0.0-10.0):**
- 9-10: Critical/urgent - must be done first, blocks other work
- 7-8: High priority - important for success, should be done early
- 5-6: Medium priority - supports the goal, normal scheduling
- 3-4: Low priority - nice to have, can be done later
- 1-2: Optional - can be deferred or skipped if needed

**EFFORT ESTIMATION:**
Use specific terms like: "30 minutes", "2 hours", "half day", "1 day", "1 week" 
Or qualitative terms: "low", "medium", "high", "very high"

**CONSTRAINTS:**
- Maximum 6 subgoals/tasks to ensure focus and manageability
- Each subgoal must be actionable and measurable
- Include realistic timelines and effort estimates
- Consider dependencies and sequencing
- Provide confidence score (0.0-1.0) for the decomposition

Your response must be a structured JSON following the GoalDecomposition schema."""


def get_decomposition_user_prompt(
    goal_node: GoalNode,
    context: Optional[Dict[str, Any]] = None,
    domain_context: Optional[Dict[str, Any]] = None,
    include_historical_patterns: bool = True,
) -> str:
    """
    Generate the user prompt for goal decomposition.

    Args:
        goal_node: The goal node to decompose
        context: Optional additional context for decomposition
        domain_context: Domain-specific context
        include_historical_patterns: Whether to include historical patterns

    Returns:
        Formatted user prompt with goal details and context
    """

    prompt_parts = []

    # Main goal information
    prompt_parts.append("**GOAL TO DECOMPOSE:**")
    prompt_parts.append(goal_node.description)
    prompt_parts.append("")

    # Goal details
    prompt_parts.append("**GOAL DETAILS:**")
    prompt_parts.append(f"- Type: {goal_node.type}")
    prompt_parts.append(f"- Priority: {goal_node.priority}")
    prompt_parts.append(f"- Current Status: {goal_node.status}")

    # Add deadline if present
    if goal_node.deadline:
        prompt_parts.append(f"- Deadline: {goal_node.deadline.isoformat()}")

    # Add existing context
    if goal_node.context:
        prompt_parts.append("- Existing Context:")
        for key, value in goal_node.context.items():
            prompt_parts.append(f"  - {key}: {value}")

    # Add tags if present
    if goal_node.tags:
        prompt_parts.append(f"- Tags: {', '.join(goal_node.tags)}")

    prompt_parts.append("")

    # Add additional context
    if context:
        prompt_parts.append("**ADDITIONAL CONTEXT:**")
        for key, value in context.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Add domain-specific context
    if domain_context:
        prompt_parts.append("**DOMAIN CONTEXT:**")
        for key, value in domain_context.items():
            prompt_parts.append(f"- {key}: {value}")
        prompt_parts.append("")

    # Add goal type specific guidance
    type_guidance = _get_type_specific_guidance(goal_node.type)
    if type_guidance:
        prompt_parts.append("**TYPE-SPECIFIC GUIDANCE:**")
        prompt_parts.append(type_guidance)
        prompt_parts.append("")

    # Add historical patterns if enabled
    if include_historical_patterns:
        patterns = _get_historical_patterns(goal_node)
        if patterns:
            prompt_parts.append("**HISTORICAL PATTERNS:**")
            prompt_parts.append(patterns)
            prompt_parts.append("")

    # Add specific requirements for this decomposition
    prompt_parts.append("**DECOMPOSITION REQUIREMENTS:**")
    prompt_parts.append("- Break down into maximum 6 subgoals/tasks (this is critical for focus)")
    prompt_parts.append("- Each subgoal must be specific and actionable")
    prompt_parts.append("- Assign realistic priorities and effort estimates")
    prompt_parts.append("- Identify clear dependencies between subgoals")
    prompt_parts.append("- Choose the most appropriate decomposition strategy")
    prompt_parts.append("- Provide success criteria for the overall goal")
    prompt_parts.append("- List potential risks and mitigation strategies")
    prompt_parts.append("- Estimate realistic timeline for completion")
    prompt_parts.append("- Include your confidence level (0.0-1.0) in this decomposition")
    prompt_parts.append("")

    prompt_parts.append(
        "**IMPORTANT:** Limit your decomposition to a maximum of 6 subgoals/tasks. Quality over quantity - focus on the most essential steps needed to achieve the goal."
    )

    return "\n".join(prompt_parts)


def _get_type_specific_guidance(node_type: NodeType) -> str:
    """Get guidance specific to the node type."""

    guidance = {
        NodeType.GOAL: """For GOAL decomposition:
- Break into major phases or milestones (max 6)
- Consider resource allocation and timeline
- Include planning, execution, and review phases
- Ensure measurable outcomes and success criteria
- Focus on strategic deliverables rather than tactical tasks""",
        NodeType.SUBGOAL: """For SUBGOAL decomposition:
- Focus on concrete deliverables (max 6)
- Keep tasks specific and actionable
- Consider dependencies and sequencing
- Include validation and quality assurance steps
- Ensure each item moves toward the subgoal completion""",
        NodeType.TASK: """For TASK decomposition:
- Break into atomic actions (max 6)
- Each subtask should be completable in one session
- Include preparation and cleanup steps
- Consider error handling and rollback procedures
- Focus on the specific steps needed to complete the task""",
    }

    return guidance.get(node_type, "")


def _get_historical_patterns(goal_node: GoalNode) -> str:
    """Get historical decomposition patterns for similar goals."""

    # This is a placeholder for actual historical pattern analysis
    # In a real implementation, this would query the memory system
    # for similar goals and their successful decomposition patterns

    patterns = []

    # Add pattern based on goal tags
    if "planning" in goal_node.tags:
        patterns.append(
            "- Planning goals typically benefit from: research → analysis → strategy → implementation → review (5 phases)"
        )

    if "project" in goal_node.tags:
        patterns.append(
            "- Project goals often follow: requirements → design → development → testing → deployment → maintenance (6 phases)"
        )

    if "learning" in goal_node.tags:
        patterns.append(
            "- Learning goals work well with: assessment → planning → study → practice → evaluation → reflection (6 phases)"
        )

    if "research" in goal_node.tags:
        patterns.append(
            "- Research goals typically involve: literature review → methodology → data collection → analysis → documentation (5 phases)"
        )

    if "development" in goal_node.tags:
        patterns.append(
            "- Development goals usually include: requirements → design → implementation → testing → deployment (5 phases)"
        )

    if "marketing" in goal_node.tags:
        patterns.append(
            "- Marketing goals often involve: market research → strategy → content creation → campaign execution → analysis (5 phases)"
        )

    if patterns:
        return "\n".join(patterns)

    return ""


def get_fallback_prompt() -> str:
    """
    Get a fallback prompt when structured generation fails.

    Returns:
        Simple fallback prompt for basic decomposition
    """
    return """Please break down the given goal into a maximum of 6 actionable subgoals or tasks. 

For each item, provide:
1. Clear description of what needs to be done
2. Type (goal, subgoal, or task)  
3. Priority (1-10 scale)
4. Estimated effort
5. Any dependencies

Focus on the most essential steps needed to achieve the goal. Keep it practical and actionable."""
