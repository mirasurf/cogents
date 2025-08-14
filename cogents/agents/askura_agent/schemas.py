"""
Schemas for AskuraAgent - Flexible data structures for dynamic conversations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Type

from langchain_core.messages import BaseMessage
from pydantic import BaseModel, Field

from cogents.common.consts import GEMINI_FLASH


class ConversationStyle(str, Enum):
    """User conversation styles."""

    DIRECT = "direct"
    EXPLORATORY = "exploratory"
    CASUAL = "casual"


class ConversationDepth(str, Enum):
    """Conversation depth levels."""

    SURFACE = "surface"
    MODERATE = "moderate"
    DEEP = "deep"


class UserConfidence(str, Enum):
    """User confidence levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class ConversationFlow(str, Enum):
    """Conversation flow patterns."""

    NATURAL = "natural"
    GUIDED = "guided"
    USER_LED = "user_led"


class ConversationSentiment(str, Enum):
    """Conversation sentiment states."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"
    UNCERTAIN = "uncertain"


class ConversationMomentum(str, Enum):
    """Conversation momentum states."""

    POSITIVE = "positive"
    NEUTRAL = "neutral"
    NEGATIVE = "negative"


class InformationSlot(BaseModel):
    """Configuration for an information slot to be collected."""

    name: str
    description: str
    priority: int = Field(default=1, description="Higher number = higher priority")
    required: bool = Field(default=True)
    extraction_tools: List[str] = Field(default_factory=list, description="Names of extraction tools to use")
    extraction_model: Optional[Type[BaseModel]] = Field(default=None, description="Pydantic model class for extraction")
    question_templates: Dict[str, Dict[str, Dict[str, str]]] = Field(default_factory=dict)
    validation_rules: List[str] = Field(default_factory=list)
    dependencies: List[str] = Field(default_factory=list, description="Other slots this depends on")

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __str__(self) -> str:
        """Return JSON string representation for f-string compatibility."""
        return self.model_dump_json()

    def __repr__(self) -> str:
        """Return JSON string representation for debugging."""
        return self.model_dump_json()


class NextActionAnalysis(BaseModel):
    """Response for intent classification and next action determination."""

    next_action: str = Field(description="The selected next action from available options")
    intent_type: str = Field(description="Intent classification: 'smalltalk' or 'task'")
    is_smalltalk: bool = Field(description="Whether the user's intent is smalltalk")
    reasoning: str = Field(description="Brief explanation of why this action was chosen")
    confidence: float = Field(default=0.0, description="Confidence score (0.0-1.0) in the action choice")


class ConversationContext(BaseModel):
    """Analysis of conversation context."""

    conversation_purpose: str = Field(default="")
    conversation_on_track_confidence: float = Field(default=0.0)
    conversation_style: ConversationStyle = Field(default=ConversationStyle.DIRECT)
    information_density: float = Field(default=0.0)
    conversation_depth: ConversationDepth = Field(default=ConversationDepth.SURFACE)
    user_confidence: UserConfidence = Field(default=UserConfidence.MEDIUM)
    conversation_flow: ConversationFlow = Field(default=ConversationFlow.NATURAL)
    conversation_momentum: ConversationMomentum = Field(default=ConversationMomentum.POSITIVE)
    last_message_sentiment: ConversationSentiment = Field(default=ConversationSentiment.NEUTRAL)
    response_patterns: List[str] = Field(default_factory=list)
    topic_transitions: List[str] = Field(default_factory=list)
    missing_info: List[str] = Field(default_factory=list)
    suggested_next_topics: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __str__(self) -> str:
        """Return concise string representation for f-string compatibility."""
        return f"ConversationContext(confidence={self.conversation_on_track_confidence:.1f}, style={self.conversation_style})"

    def __repr__(self) -> str:
        """Return concise representation for debugging."""
        return f"ConversationContext(confidence={self.conversation_on_track_confidence:.1f}, style={self.conversation_style}, missing={len(self.missing_info)})"


class AskuraState(BaseModel):
    """Core state for AskuraAgent conversations."""

    # User identification
    user_id: str
    session_id: str

    # Conversation state
    messages: Sequence[BaseMessage] = Field(default_factory=list)
    conversation_context: ConversationContext = Field(default_factory=ConversationContext)

    # Information slots (dynamic based on configuration)
    extracted_information_slots: Dict[str, Any] = Field(default_factory=dict)

    # Metadata
    turns: int = Field(default=0)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

    # Agent control
    requires_user_input: bool = Field(default=True)
    is_complete: bool = Field(default=False)
    pending_extraction: bool = Field(default=False)

    # Next action analysis results
    next_action_ayalysis: Optional[NextActionAnalysis] = Field(default=None)

    # Custom fields (for specific agents)
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def __str__(self) -> str:
        """Return concise string representation for f-string compatibility."""
        return f"AskuraState(session={self.session_id[:8]}, turns={self.turns}, complete={self.is_complete})"

    def __repr__(self) -> str:
        """Return concise representation for debugging."""
        return f"AskuraState(user={self.user_id}, session={self.session_id[:8]}, turns={self.turns}, slots={len(self.extracted_information_slots)})"

    def __getitem__(self, key: str) -> Any:
        """Allow dictionary-like access to model fields."""
        return getattr(self, key)

    def __setitem__(self, key: str, value: Any) -> None:
        """Allow dictionary-like setting of model fields."""
        setattr(self, key, value)

    def get(self, key: str, default: Any = None) -> Any:
        """Allow dictionary-like get method."""
        return getattr(self, key, default)


class AskuraConfig(BaseModel):
    """Configuration for AskuraAgent."""

    # LLM configuration
    llm_api_provider: str = "openrouter"
    model_name: str = GEMINI_FLASH
    temperature: float = 0.7
    max_tokens: int = 1000

    # Conversation limits
    max_conversation_turns: int = 10
    max_conversation_time: Optional[int] = None  # seconds

    # Purposes of the conversation
    conversation_purposes: List[str] = Field(default_factory=list)

    # Information slots configuration
    information_slots: List[InformationSlot] = Field(default_factory=list)

    # Conversation style preferences
    preferred_conversation_style: Optional[ConversationStyle] = None
    enable_style_adaptation: bool = True
    enable_sentiment_analysis: bool = True
    enable_confidence_boosting: bool = True

    # Extraction configuration
    extraction_retry_attempts: int = 2
    extraction_timeout: float = 5.0

    # Question generation
    enable_contextual_questions: bool = True
    enable_confidence_boosting_questions: bool = True
    enable_momentum_maintenance: bool = True

    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)


class AskuraResponse(BaseModel):
    """Response from AskuraAgent."""

    message: str
    session_id: str
    is_complete: bool = False
    confidence: float = 0.0
    next_actions: List[str] = Field(default_factory=list)
    requires_user_input: bool = True
    metadata: Dict[str, Any] = Field(default_factory=dict)

    # Custom response data
    custom_data: Dict[str, Any] = Field(default_factory=dict)


class ExtractionResult(BaseModel):
    """Result of information extraction."""

    slot_name: str
    extracted_value: Any
    confidence: float
    extraction_method: str
    raw_text: str
    metadata: Dict[str, Any] = Field(default_factory=dict)


class QuestionTemplate(BaseModel):
    """Template for generating questions."""

    action: str
    style: ConversationStyle
    depth: ConversationDepth
    template: str
    contextual_elements: List[str] = Field(default_factory=list)
    conditions: Dict[str, Any] = Field(default_factory=dict)
