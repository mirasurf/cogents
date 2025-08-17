"""
LLM-based goal decomposer for the GoalithService.

Provides structured goal decomposition using LLM clients with instructor integration.
"""

from typing import Any, Dict, List, Optional

from cogent_nano.utils.llm_utils import get_llm_instructor
from pydantic import BaseModel, Field

from cogents.common.logging import get_logger

from ..base.decomposer import GoalDecomposer
from ..base.errors import DecompositionError
from ..base.goal_node import GoalNode, NodeType

logger = get_logger(__name__)


class SubgoalSpec(BaseModel):
    """Specification for a single subgoal or task."""

    description: str = Field(description="Clear, actionable description of the subgoal or task")
    type: str = Field(
        description="Type of node: 'goal', 'subgoal', or 'task'",
        pattern="^(goal|subgoal|task)$",
    )
    priority: float = Field(
        description="Priority score between 0.0 and 10.0 (higher = more important)",
        ge=0.0,
        le=10.0,
    )
    estimated_effort: Optional[str] = Field(
        description="Estimated effort or duration (e.g., '2 hours', '3 days', 'low', 'medium', 'high')",
        default=None,
    )
    dependencies: List[str] = Field(
        description="List of descriptions of other subgoals this depends on (will be matched by description)",
        default_factory=list,
    )
    tags: List[str] = Field(
        description="Tags for categorization (e.g., 'research', 'planning', 'execution')",
        default_factory=list,
    )
    notes: Optional[str] = Field(description="Additional notes or context about this subgoal", default=None)


class GoalDecomposition(BaseModel):
    """Complete decomposition of a goal into subgoals and tasks."""

    reasoning: str = Field(description="Explanation of the decomposition approach and rationale")
    decomposition_strategy: str = Field(
        description="Strategy used: 'sequential', 'parallel', 'hybrid', or 'milestone-based'"
    )
    subgoals: List[SubgoalSpec] = Field(description="List of subgoals and tasks, in logical order", min_length=1)
    success_criteria: List[str] = Field(
        description="Criteria that indicate successful completion of the overall goal",
        default_factory=list,
    )
    potential_risks: List[str] = Field(
        description="Potential risks or challenges in executing this plan",
        default_factory=list,
    )
    estimated_timeline: Optional[str] = Field(
        description="Overall estimated timeline for goal completion", default=None
    )
    confidence: float = Field(description="Confidence in this decomposition (0.0 to 1.0)", ge=0.0, le=1.0)


class LLMDecomposer(GoalDecomposer):
    """
    Enhanced LLM-based goal decomposer using structured completion.

    Uses the project's LLM infrastructure with instructor for structured output
    to decompose goals into subgoals and tasks. Includes contextual features
    for domain-specific knowledge and historical patterns.
    """

    def __init__(
        self,
        model_name: Optional[str] = None,
        temperature: float = 0.3,
        max_tokens: int = 2000,
        name: str = "llm_decomposer",
        domain_context: Optional[Dict[str, Any]] = None,
        include_historical_patterns: bool = True,
    ):
        """
        Initialize LLM decomposer.

        Args:
            model_name: LLM model to use (uses default if None)
            temperature: Sampling temperature for LLM
            max_tokens: Maximum tokens for response
            name: Name of this decomposer
            domain_context: Domain-specific context to include
            include_historical_patterns: Whether to include historical decomposition patterns
        """
        self._model_name = model_name
        self._temperature = temperature
        self._max_tokens = max_tokens
        self._name = name
        self._domain_context = domain_context or {}
        self._include_historical = include_historical_patterns

        # Initialize LLM client lazily - will be set when first needed
        self._llm_client = None

    def _ensure_llm_client(self):
        """Ensure LLM client is initialized (lazy initialization)."""
        if self._llm_client is None:
            try:
                self._llm_client = get_llm_instructor()
                logger.info(f"Initialized LLM decomposer with model: {self._model_name or 'default'}")
            except Exception as e:
                logger.error(f"Failed to initialize LLM client: {e}")
                raise DecompositionError(f"LLM client initialization failed: {e}")

    @property
    def name(self) -> str:
        """Get the name of this decomposer."""
        return self._name

    def decompose(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> List[GoalNode]:
        """
        Decompose a goal using LLM structured completion.

        Args:
            goal_node: The goal node to decompose
            context: Optional context for decomposition

        Returns:
            List of subgoal/task nodes

        Raises:
            DecompositionError: If decomposition fails
        """
        # Ensure LLM client is initialized before use
        self._ensure_llm_client()

        try:
            logger.info(f"Decomposing goal: {goal_node.description}")

            # Build the decomposition prompt
            prompt = self._build_decomposition_prompt(goal_node, context)

            # Get structured decomposition from LLM
            messages = [{"role": "user", "content": prompt}]

            decomposition: GoalDecomposition = self._llm_client.structured_completion(
                messages=messages,
                response_model=GoalDecomposition,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )

            logger.info(f"LLM decomposition completed with {len(decomposition.subgoals)} subgoals")
            logger.debug(f"Decomposition reasoning: {decomposition.reasoning}")

            # Convert to GoalNode objects
            subgoal_nodes = self._convert_to_goal_nodes(decomposition, goal_node, context)

            # Set up dependencies
            self._setup_dependencies(subgoal_nodes, decomposition.subgoals)

            # Add decomposition metadata to parent goal
            goal_node.update_context(
                "llm_decomposition",
                {
                    "reasoning": decomposition.reasoning,
                    "strategy": decomposition.decomposition_strategy,
                    "success_criteria": decomposition.success_criteria,
                    "potential_risks": decomposition.potential_risks,
                    "estimated_timeline": decomposition.estimated_timeline,
                    "confidence": decomposition.confidence,
                    "subgoal_count": len(decomposition.subgoals),
                },
            )

            return subgoal_nodes

        except Exception as e:
            logger.error(f"LLM decomposition failed for goal {goal_node.id}: {e}")
            raise DecompositionError(f"LLM decomposition failed: {e}")

    def _build_decomposition_prompt(self, goal_node: GoalNode, context: Optional[Dict[str, Any]] = None) -> str:
        """Build the decomposition prompt for the LLM."""

        # Base prompt
        prompt = f"""You are an expert goal decomposition assistant. Your task is to break down a goal into actionable subgoals and tasks.

**GOAL TO DECOMPOSE:**
{goal_node.description}

**GOAL DETAILS:**
- Type: {goal_node.type}
- Priority: {goal_node.priority}
- Current Status: {goal_node.status}"""

        # Add deadline if present
        if goal_node.deadline:
            prompt += f"\n- Deadline: {goal_node.deadline.isoformat()}"

        # Add existing context
        if goal_node.context:
            prompt += "\n- Existing Context:"
            for key, value in goal_node.context.items():
                prompt += f"\n  - {key}: {value}"

        # Add tags if present
        if goal_node.tags:
            prompt += f"\n- Tags: {', '.join(goal_node.tags)}"

        # Add additional context
        if context:
            prompt += "\n\n**ADDITIONAL CONTEXT:**"
            for key, value in context.items():
                prompt += f"\n- {key}: {value}"

        # Add domain-specific context
        if self._domain_context:
            prompt += "\n\n**DOMAIN CONTEXT:**"
            for key, value in self._domain_context.items():
                prompt += f"\n- {key}: {value}"

        # Add goal type specific guidance
        type_guidance = self._get_type_specific_guidance(goal_node.type)
        if type_guidance:
            prompt += f"\n\n**TYPE-SPECIFIC GUIDANCE:**\n{type_guidance}"

        # Add historical patterns if enabled
        if self._include_historical:
            patterns = self._get_historical_patterns(goal_node)
            if patterns:
                prompt += f"\n\n**HISTORICAL PATTERNS:**\n{patterns}"

        # Add decomposition guidelines
        prompt += """

**DECOMPOSITION GUIDELINES:**

1. **Clarity**: Each subgoal should be clear, specific, and actionable
2. **Granularity**: Break down into appropriate-sized chunks (not too big, not too small)
3. **Dependencies**: Identify which subgoals depend on others
4. **Types**: Use 'task' for concrete actions, 'subgoal' for intermediate objectives, 'goal' for major milestones
5. **Priority**: Assign priorities based on importance and urgency
6. **Effort**: Estimate effort realistically
7. **Strategy**: Choose the best decomposition approach:
   - Sequential: Tasks must be done in order
   - Parallel: Tasks can be done simultaneously
   - Hybrid: Mix of sequential and parallel work
   - Milestone-based: Organized around key milestones

**PRIORITY SCORING:**
- 9-10: Critical/urgent tasks that must be done first
- 7-8: High priority tasks that are important for success
- 5-6: Medium priority tasks that support the goal
- 3-4: Low priority tasks that are nice to have
- 1-2: Optional tasks that can be deferred

**EFFORT ESTIMATION:**
Use terms like: "15 minutes", "1 hour", "half day", "1 day", "1 week", or qualitative terms like "low", "medium", "high"

**OUTPUT REQUIREMENTS:**
Provide a structured decomposition with clear reasoning and practical subgoals that can be executed."""

        return prompt

    def _get_type_specific_guidance(self, node_type: NodeType) -> str:
        """Get guidance specific to the node type."""

        guidance = {
            NodeType.GOAL: """
For GOAL decomposition:
- Break into major phases or milestones
- Consider resource allocation and timeline
- Include planning, execution, and review phases
- Ensure measurable outcomes""",
            NodeType.SUBGOAL: """
For SUBGOAL decomposition:
- Focus on concrete deliverables
- Keep tasks specific and actionable
- Consider dependencies and sequencing
- Include validation steps""",
            NodeType.TASK: """
For TASK decomposition:
- Break into atomic actions
- Each subtask should be completable in one session
- Include preparation and cleanup steps
- Consider error handling and rollback""",
        }

        return guidance.get(node_type, "")

    def _get_historical_patterns(self, goal_node: GoalNode) -> str:
        """Get historical decomposition patterns for similar goals."""

        # This is a placeholder for actual historical pattern analysis
        # In a real implementation, this would query the memory system
        # for similar goals and their successful decomposition patterns

        patterns = []

        # Add pattern based on goal tags
        if "planning" in goal_node.tags:
            patterns.append(
                "- Planning goals typically benefit from research → analysis → strategy → implementation phases"
            )

        if "project" in goal_node.tags:
            patterns.append("- Project goals often follow: requirements → design → development → testing → deployment")

        if "learning" in goal_node.tags:
            patterns.append("- Learning goals work well with: assessment → planning → study → practice → evaluation")

        if patterns:
            return "\n".join(patterns)

        return ""

    def _convert_to_goal_nodes(
        self,
        decomposition: GoalDecomposition,
        parent_goal: GoalNode,
        context: Optional[Dict[str, Any]] = None,
    ) -> List[GoalNode]:
        """Convert LLM decomposition to GoalNode objects."""

        nodes = []

        for spec in decomposition.subgoals:
            # Create node context
            node_context = {}
            if context:
                node_context.update(context)

            # Add LLM-specific context
            node_context.update(
                {
                    "llm_generated": True,
                    "decomposition_strategy": decomposition.decomposition_strategy,
                    "estimated_effort": spec.estimated_effort,
                    "notes": spec.notes,
                    "parent_goal_id": parent_goal.id,
                }
            )

            # Create the node
            node = GoalNode(
                description=spec.description,
                type=NodeType(spec.type),
                priority=spec.priority,
                parent=parent_goal.id,
                context=node_context,
                tags=set(spec.tags),
                decomposer_name=self.name,
            )

            # Copy deadline from parent if not specified and it's a task
            if parent_goal.deadline and node.type == NodeType.TASK:
                node.deadline = parent_goal.deadline

            nodes.append(node)

        return nodes

    def _setup_dependencies(self, nodes: List[GoalNode], specs: List[SubgoalSpec]) -> None:
        """Set up dependencies between nodes based on LLM specifications."""

        # Create a mapping from description to node ID
        desc_to_id = {node.description: node.id for node in nodes}

        for i, spec in enumerate(specs):
            if not spec.dependencies:
                continue

            current_node = nodes[i]

            for dep_desc in spec.dependencies:
                # Find the dependency by description (fuzzy matching)
                dep_node_id = self._find_dependency_by_description(dep_desc, desc_to_id)

                if dep_node_id:
                    current_node.add_dependency(dep_node_id)
                    logger.debug(f"Added dependency: {dep_node_id} -> {current_node.id}")
                else:
                    logger.warning(f"Could not find dependency '{dep_desc}' for node '{current_node.description}'")

    def _find_dependency_by_description(self, dep_desc: str, desc_to_id: Dict[str, str]) -> Optional[str]:
        """Find a dependency node by description with fuzzy matching."""

        dep_desc_lower = dep_desc.lower().strip()

        # Exact match first
        for desc, node_id in desc_to_id.items():
            if desc.lower().strip() == dep_desc_lower:
                return node_id

        # Partial match (dependency description is contained in node description)
        for desc, node_id in desc_to_id.items():
            if dep_desc_lower in desc.lower() or desc.lower() in dep_desc_lower:
                return node_id

        # Word overlap matching
        dep_words = set(dep_desc_lower.split())
        best_match = None
        best_overlap = 0

        for desc, node_id in desc_to_id.items():
            desc_words = set(desc.lower().split())
            overlap = len(dep_words & desc_words)

            if overlap > best_overlap and overlap >= 2:  # At least 2 words overlap
                best_match = node_id
                best_overlap = overlap

        return best_match
