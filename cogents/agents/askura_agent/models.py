"""
Schemas for AskuraAgent - Flexible data structures for dynamic conversations.
"""

from enum import Enum
from typing import Any, Dict, List, Optional, Sequence, Type, Union

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


class NextActionPlan(BaseModel):
    """Response for intent classification and next action determination."""

    next_action: str = Field(description="The selected next action from available options")
    intent_type: str = Field(description="Intent classification: 'smalltalk' or 'task'")
    is_smalltalk: bool = Field(description="Whether the user's intent is smalltalk")
    reasoning: str = Field(description="Brief explanation of why this action was chosen")
    confidence: float = Field(default=0.0, description="Confidence score (0.0-1.0) in the action choice")


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


class ConversationContext(BaseModel):
    """Analysis of conversation context."""

    # Conversation purpose
    conversation_purpose: str = Field(default="")
    conversation_on_track_confidence: float = Field(default=0.0)

    # Conversation vibe
    conversation_style: ConversationStyle = Field(default=ConversationStyle.DIRECT)
    information_density: float = Field(default=0.0)
    conversation_depth: ConversationDepth = Field(default=ConversationDepth.SURFACE)
    user_confidence: UserConfidence = Field(default=UserConfidence.MEDIUM)
    conversation_flow: ConversationFlow = Field(default=ConversationFlow.NATURAL)
    conversation_momentum: ConversationMomentum = Field(default=ConversationMomentum.POSITIVE)
    last_message_sentiment: ConversationSentiment = Field(default=ConversationSentiment.NEUTRAL)

    # Complementary information
    missing_info: List[str] = Field(default_factory=list)
    suggested_next_topics: List[str] = Field(default_factory=list)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "conversation_purpose": self.conversation_purpose,
            "conversation_on_track_confidence": self.conversation_on_track_confidence,
            "conversation_style": self.conversation_style.value,
            "information_density": self.information_density,
            "conversation_depth": self.conversation_depth.value,
            "user_confidence": self.user_confidence.value,
            "conversation_flow": self.conversation_flow.value,
            "sentiment": self.last_message_sentiment.value,
            "momentum": self.conversation_momentum.value,
            "missing_info": self.missing_info,
            "suggested_next_topics": self.suggested_next_topics,
        }


class AskuraState(BaseModel):
    """Core state for AskuraAgent conversations."""

    # Metadata
    user_id: str
    session_id: str
    turns: int = Field(default=0)
    created_at: str = Field(default="")
    updated_at: str = Field(default="")

    # Conversation state
    messages: Sequence[BaseMessage] = Field(default_factory=list)
    chat_context: ConversationContext = Field(default_factory=ConversationContext)

    # Information slots (dynamic based on configuration)
    extracted_slots: Union[str, Dict[str, str]] = Field(default_factory=dict)
    # Next action analysis results
    next_action_plan: Optional[NextActionPlan] = Field(default=None)

    # Agent control
    requires_user_input: bool = Field(default=True)
    is_complete: bool = Field(default=False)
    pending_extraction: bool = Field(default=False)

    # Custom fields (for specific agents)
    custom_data: Dict[str, Any] = Field(default_factory=dict)

    class Config:
        """Pydantic configuration."""

        use_enum_values = True


class AskuraConfig(BaseModel):
    """Configuration for AskuraAgent."""

    # LLM configuration
    llm_api_provider: str = "openrouter"
    model_name: str = GEMINI_FLASH
    temperature: float = 0.7
    max_tokens: int = 1000

    # Conversation limits
    max_conversation_turns: int = 10

    # Purposes of the conversation
    conversation_purposes: List[str] = Field(default_factory=list)

    # Information slots configuration
    information_slots: List[InformationSlot] = Field(default_factory=list)

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
