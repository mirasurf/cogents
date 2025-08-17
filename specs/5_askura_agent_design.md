# AskuraAgent - General-Purpose Dynamic Conversation Agent

## Overview

**AskuraAgent** is a general-purpose, configurable framework for human-in-the-loop conversations that adapts to different user communication styles and dynamically collects required information through natural conversation flow. It serves as a base agent that can be extended for specific use cases like TripSpark.

## Key Features

### 🎯 **General-Purpose Design**
- **Configurable information slots**: Define what information needs to be collected
- **Flexible extraction tools**: Plug in any extraction tools for different domains
- **Adaptable conversation flow**: Works for any conversation scenario
- **Extensible architecture**: Easy to inherit and customize for specific use cases

### 🧠 **Dynamic Conversation System**
- **Style detection**: Automatically detects user conversation style (direct, exploratory, casual)
- **Context-aware questioning**: Questions adapt based on conversation context
- **Multi-topic extraction**: Extracts all possible information from each user message
- **Intelligent prioritization**: Determines the most appropriate next action

### 💬 **Natural Conversation Flow**
- **No rigid order**: Information collection follows natural conversation patterns
- **Contextual questions**: Questions include relevant context based on what's already known
- **Confidence boosting**: Special handling for users with low confidence
- **Momentum maintenance**: Questions designed to maintain positive conversation flow

## Architecture

### Core Components

```
AskuraAgent
├── ConversationManager     # Analyzes conversation context and determines next actions
├── InformationExtractor    # Extracts information from user messages
├── QuestionGenerator       # Generates contextual questions
└── AskuraAgent            # Main orchestrator
```

### Data Flow

```
User Message → ConversationManager → InformationExtractor → QuestionGenerator → Response
     ↓              ↓                      ↓                      ↓
Context Analysis → Action Determination → Info Extraction → Question Generation
```

## Configuration

### AskuraConfig

The main configuration class that defines how the agent behaves:

```python
@dataclass
class AskuraConfig:
    # LLM configuration
    model_name: str = "gpt-4o-mini"
    temperature: float = 0.7
    max_tokens: int = 1000
    
    # Conversation limits
    max_conversation_turns: int = 15
    max_conversation_time: Optional[int] = None
    
    # Information slots configuration
    information_slots: List[InformationSlot] = Field(default_factory=list)
    
    # Conversation style preferences
    enable_style_adaptation: bool = True
    enable_sentiment_analysis: bool = True
    enable_confidence_boosting: bool = True
    
    # Extraction configuration
    enable_multi_topic_extraction: bool = True
    extraction_retry_attempts: int = 2
    
    # Question generation
    enable_contextual_questions: bool = True
    enable_confidence_boosting_questions: bool = True
    enable_momentum_maintenance: bool = True
    
    # Custom configuration
    custom_config: Dict[str, Any] = Field(default_factory=dict)
```

### InformationSlot

Defines what information needs to be collected:

```python
@dataclass
class InformationSlot:
    name: str                    # Unique identifier for the slot
    description: str             # Human-readable description
    priority: int = 1            # Higher number = higher priority
    required: bool = True        # Whether this information is required
    extraction_tools: List[str]  # Tools to use for extraction
    question_templates: Dict     # Question templates for different styles
    validation_rules: List[str]  # Validation rules for the information
    dependencies: List[str]      # Other slots this depends on
```

## Usage Examples

### Basic Usage

```python
from cogents.agents.askura_agent import AskuraAgent, AskuraConfig, InformationSlot

# Define information slots
information_slots = [
    InformationSlot(
        name="name",
        description="User's name",
        priority=5,
        required=True,
        extraction_tools=["name_extractor"]
    ),
    InformationSlot(
        name="email",
        description="User's email",
        priority=4,
        required=True,
        extraction_tools=["email_extractor"]
    )
]

# Create configuration
config = AskuraConfig(
    model_name="gpt-4o-mini",
    information_slots=information_slots
)

# Create extraction tools
extraction_tools = {
    "name_extractor": name_extraction_tool,
    "email_extractor": email_extraction_tool
}

# Initialize agent
agent = AskuraAgent(config, extraction_tools)

# Start conversation
response = agent.start_conversation("user_123", "Hi, I need help")
print(response.message)

# Process user message
response = agent.process_user_message("user_123", response.session_id, "My name is John")
print(response.message)
```

### Travel Planning Example

```python
# Define travel-specific information slots
travel_slots = [
    InformationSlot(
        name="destination",
        description="Travel destination",
        priority=5,
        required=True,
        extraction_tools=["destination_extractor"]
    ),
    InformationSlot(
        name="dates",
        description="Travel dates",
        priority=4,
        required=True,
        extraction_tools=["dates_extractor"]
    ),
    InformationSlot(
        name="budget",
        description="Travel budget",
        priority=3,
        required=True,
        extraction_tools=["budget_extractor"]
    )
]

# Create travel planning agent
travel_config = AskuraConfig(
    model_name="gpt-4o-mini",
    information_slots=travel_slots,
    custom_config={"use_case": "travel_planning"}
)

travel_agent = AskuraAgent(travel_config, travel_extraction_tools)
```

### Job Interview Preparation Example

```python
# Define interview-specific information slots
interview_slots = [
    InformationSlot(
        name="position",
        description="Job position",
        priority=5,
        required=True,
        extraction_tools=["position_extractor"]
    ),
    InformationSlot(
        name="company",
        description="Company name",
        priority=4,
        required=True,
        extraction_tools=["company_extractor"]
    ),
    InformationSlot(
        name="experience",
        description="Relevant experience",
        priority=3,
        required=True,
        extraction_tools=["experience_extractor"]
    )
]

# Create interview preparation agent
interview_config = AskuraConfig(
    model_name="gpt-4o-mini",
    information_slots=interview_slots,
    custom_config={"use_case": "job_interview"}
)

interview_agent = AskuraAgent(interview_config, interview_extraction_tools)
```

## Extending AskuraAgent

### Creating a Custom Agent

```python
class CustomAgent(AskuraAgent):
    """Custom agent that extends AskuraAgent for specific use case."""
    
    def __init__(self, config: CustomConfig):
        # Convert custom config to AskuraConfig
        askura_config = self._convert_config(config)
        
        # Define custom extraction tools
        extraction_tools = self._create_extraction_tools()
        
        # Initialize parent class
        super().__init__(askura_config, extraction_tools)
        
        # Store custom config
        self.custom_config = config
        
    def _convert_config(self, config: CustomConfig) -> AskuraConfig:
        """Convert custom config to AskuraConfig."""
        # Define information slots for your use case
        information_slots = [
            InformationSlot(
                name="custom_field_1",
                description="Custom field 1",
                priority=5,
                required=True,
                extraction_tools=["custom_extractor_1"]
            ),
            # ... more slots
        ]
        
        return AskuraConfig(
            model_name=config.model_name,
            information_slots=information_slots,
            custom_config={"use_case": "custom_use_case"}
        )
        
    def _create_extraction_tools(self) -> Dict[str, Any]:
        """Create custom extraction tools."""
        return {
            "custom_extractor_1": self._custom_extraction_tool_1,
            # ... more tools
        }
        
    def _custom_extraction_tool_1(self, user_message: str) -> Dict[str, Any]:
        """Custom extraction tool implementation."""
        # Implement your extraction logic
        return {"custom_field_1": "extracted_value"}
        
    def start_conversation(self, user_id: str, initial_message: Optional[str] = None) -> CustomResponse:
        """Override to return custom response type."""
        askura_response = super().start_conversation(user_id, initial_message)
        return self._convert_response(askura_response)
        
    def _convert_response(self, askura_response: AskuraResponse) -> CustomResponse:
        """Convert AskuraResponse to CustomResponse."""
        # Implement conversion logic
        return CustomResponse(
            message=askura_response.message,
            session_id=askura_response.session_id,
            # ... other fields
        )
```

## Conversation Styles

### Direct Users
- **Characteristics**: Short, concise answers, clear preferences
- **Strategy**: Efficient, straightforward questions
- **Example**: "What's your name?" → "John" → "What's your email?"

### Exploratory Users
- **Characteristics**: Detailed responses, curious, open-ended
- **Strategy**: Engaging, detailed questions that encourage exploration
- **Example**: "I'm curious about your background! Tell me about yourself." → Detailed response → "What experiences are most important to you?"

### Casual Users
- **Characteristics**: Uncertain, hesitant, needs encouragement
- **Strategy**: Supportive, confidence-boosting questions
- **Example**: "So, what brings you here today?" → "I'm not sure..." → "That's totally fine! Let's start with something simple - what's your name?"

## Question Generation System

### Multi-Dimensional Templates

Questions are generated using a three-dimensional template system:

```python
templates = {
    "ask_name": {
        ConversationStyle.DIRECT: {
            ConversationDepth.SURFACE: "What's your name?",
            ConversationDepth.MODERATE: "What should I call you?",
            ConversationDepth.DEEP: "How would you like to be addressed?"
        },
        ConversationStyle.EXPLORATORY: {
            ConversationDepth.SURFACE: "I'd love to know your name!",
            ConversationDepth.MODERATE: "What name do you go by?",
            ConversationDepth.DEEP: "What name feels most authentic to you?"
        },
        ConversationStyle.CASUAL: {
            ConversationDepth.SURFACE: "What's your name?",
            ConversationDepth.MODERATE: "What do people call you?",
            ConversationDepth.DEEP: "What name resonates with you?"
        }
    }
}
```

### Contextual Elements

Questions are enhanced with contextual elements:

```python
def _generate_contextual_elements(self, action: str, state: AskuraState, context: ConversationContext) -> List[str]:
    contextual_elements = []
    
    # Confidence-boosting context
    if context.user_confidence == UserConfidence.LOW:
        contextual_elements.append("There are no wrong answers")
        
    # Momentum-boosting context
    if context.conversation_momentum == ConversationMomentum.POSITIVE:
        contextual_elements.append("I'm excited to help you")
        
    return contextual_elements
```

## Benefits

### For Developers
- **Reusable framework**: Build multiple conversation agents from one base
- **Consistent behavior**: All agents share the same conversation logic
- **Easy customization**: Simple configuration for different use cases
- **Maintainable code**: Clear separation of concerns

### For Users
- **Natural conversations**: No rigid question sequences
- **Personalized experience**: Adapts to individual communication styles
- **Efficient information sharing**: Can provide multiple pieces of information at once
- **Confidence building**: Supportive interaction for uncertain users

## Integration with TripSpark

TripSpark V2 inherits from AskuraAgent to leverage the dynamic conversation system:

```python
class TripSparkV2(AskuraAgent):
    """TripSpark V2 - Enhanced travel planning agent."""
    
    def __init__(self, config: TripSparkConfig):
        # Convert TripSparkConfig to AskuraConfig
        askura_config = self._convert_config(config)
        
        # Define travel-specific extraction tools
        extraction_tools = {
            "extract_destination_info": extract_destination_info,
            "extract_date_info": extract_date_info,
            # ... more tools
        }
        
        # Initialize parent class
        super().__init__(askura_config, extraction_tools)
        
    def start_conversation(self, user_id: str, initial_message: Optional[str] = None) -> TripSparkResponse:
        """Override to return TripSpark-compatible response."""
        askura_response = super().start_conversation(user_id, initial_message)
        return self._convert_response(askura_response)
```

## Future Enhancements

### Planned Features
- **Learning capabilities**: System learns from successful conversations
- **Emotion detection**: More sophisticated sentiment analysis
- **Cultural adaptation**: Questions adapted to user's cultural background
- **Multi-language support**: Dynamic question generation in multiple languages

### Extension Points
- **Custom conversation styles**: Developers can add new conversation styles
- **Custom question templates**: Easy to add new question types
- **Custom context analysis**: Extensible context analysis framework
- **Custom information extraction**: Pluggable information extraction system

## Conclusion

AskuraAgent provides a powerful, flexible foundation for building dynamic conversation agents. Its general-purpose design makes it suitable for any conversation scenario, while its extensible architecture allows for easy customization and inheritance. By leveraging AskuraAgent, developers can quickly create sophisticated conversation agents that provide natural, personalized user experiences.
