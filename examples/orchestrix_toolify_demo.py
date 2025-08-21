from __future__ import annotations

from cogents.goalith import GoalithService, NodeType
from cogents.orchestrix import OrchestrixService
from cogents.toolify import ToolCapability, ToolCard, ToolifyService


def main() -> None:
    # Setup Goalith
    goalith = GoalithService()

    # Create a simple task
    task_id = goalith.create_goal("Echo task", goal_type=NodeType.TASK, context={"capabilities": ["echo.run"]})

    # Setup Toolify with a local echo tool
    toolify = ToolifyService()
    toolify.register_tool(
        ToolCard(
            tool_id="echo_local",
            name="Echo",
            type="local",
            capabilities=[ToolCapability(name="echo.run")],
        )
    )

    # Setup Orchestrix
    orchestrix = OrchestrixService(goalith=goalith, toolify=toolify)

    # Run one orchestration cycle
    orchestrated = orchestrix.run_once(limit=10)

    # Process pending updates
    goalith.process_pending_updates()

    # Show result
    goal = goalith.get_goal(task_id)
    print("Task status:", goal.status)


if __name__ == "__main__":
    main()
