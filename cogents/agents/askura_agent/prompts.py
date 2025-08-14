"""
Prompts for AskuraAgent - Structured prompts for conversation analysis and management.
"""


# Structured extraction prompts - optimized for structured_completion
CONVERSATION_ANALYSIS_PROMPTS = {
    "conversation_context": """Analyze conversation style and alignment with purpose: {conversation_purpose}

Assess key factors:
- Style: direct (goal-oriented), exploratory (curious), casual (relaxed)
- User confidence: low (hesitant), medium (balanced), high (assertive)
- Flow: natural (organic), guided (following direction), user_led (user driving)
- Sentiment: positive (enthusiastic), neutral (balanced), negative (frustrated), uncertain (confused)
- Momentum: positive (building), neutral (steady), negative (losing interest)
- On-track confidence (0.0-1.0): How well conversation aligns with purpose
  * 0.0-0.3: Off-track, not addressing purpose
  * 0.4-0.6: Partially on-track, some relevance  
  * 0.7-0.8: Mostly on-track, good alignment
  * 0.9-1.0: Highly focused on purpose

Recent messages: {recent_messages}""",
    "next_action": """Determine the optimal next action based on conversation context and user needs.

Consider these factors:
- User's conversation style and preferences: {conversation_style}
- Current conversation momentum and sentiment: {momentum}, {sentiment}  
- Missing information priorities: {missing_info}
- User confidence level: {user_confidence}
- Conversation alignment with purpose: {conversation_on_track_confidence}

Available actions: {available_actions}

Guidelines for action selection:
- If conversation off-track (confidence < 0.4): prioritize redirecting to purpose
- If conversation on-track (confidence > 0.7): focus on gathering missing information
- If user confidence is low: choose confidence-boosting, supportive actions
- If momentum is negative: consider redirecting or providing encouragement
- Balance staying on purpose with maintaining user engagement and trust

Select the most appropriate action that serves both the conversation purpose and user experience.""",
    "determine_next_action": """Classify MOST RECENT message intent and select optimal next action.

Intent Classification (focus ONLY on last message):
- "smalltalk": Greetings, pleasantries, casual conversation
- "task": Goal-oriented, information requests, specific questions, task content

Context:
- Purpose: {conversation_purpose}
- On-track confidence: {conversation_on_track_confidence}
- User style: {conversation_style}
- User confidence: {user_confidence} 
- Conversation flow: {conversation_flow}
- Sentiment: {sentiment}
- Momentum: {momentum}
- Missing info: {missing_info}
- Ready to summarize: {ready_to_summarize}

Available actions: {available_actions}
Recent messages: {recent_messages}

Decision Guidelines:
- If MOST RECENT message is smalltalk: respond appropriately but guide toward task
- If MOST RECENT message is task: focus on gathering missing information
- If conversation off-track (<0.4): prioritize redirecting to purpose  
- If conversation on-track (>0.7): focus on collecting missing info
- If user confidence low: choose supportive, confidence-boosting actions
- If momentum negative: provide encouragement or redirect
- Balance staying on purpose with maintaining engagement

Reasoning must explicitly reference the MOST RECENT user message.""",
}


def get_conversation_analysis_prompt(analysis_type: str, **kwargs) -> str:
    """Get a conversation analysis prompt for the specified type."""
    prompt = CONVERSATION_ANALYSIS_PROMPTS.get(analysis_type, "")
    try:
        return prompt.format(**kwargs)
    except KeyError:
        return prompt


# --- Question generation prompts -------------------------------------------------

QUESTION_GENERATION_PROMPT = """Generate ONE specific, natural question to guide the user toward goals: {conversation_purposes}

REQUIREMENTS:
- Be conversational and match the user's style: {conversation_style}
- Ask for highest-priority missing information: {missing_required_slots}
- Build on what you know: {known_slots}
- Avoid repeating questions already asked
- Provide helpful context or examples when appropriate
- Keep it concise and friendly

Next action: {next_action}
Output: Question only."""


SMALLTALK_PIVOT_PROMPT = """Be friendly, warm, and naturally conversational. Briefly acknowledge the user's greeting/smalltalk, then smoothly pivot to the task with ONE engaging question that moves toward the conversation purposes.

Requirements:
- Respond warmly to the user's greeting/smalltalk first
- Make the transition feel natural, not abrupt  
- Ask a specific, engaging question related to the conversation purpose
- Match the user's conversational tone
- Keep it concise but personable

Context:
- Conversation purposes: {conversation_purposes}
- Last user message: {last_user_message}

Output: One friendly response with a natural pivot question."""


def get_question_generation_prompt(**kwargs) -> str:
    try:
        return QUESTION_GENERATION_PROMPT.format(**kwargs)
    except KeyError:
        return QUESTION_GENERATION_PROMPT


def get_smalltalk_pivot_prompt(**kwargs) -> str:
    try:
        return SMALLTALK_PIVOT_PROMPT.format(**kwargs)
    except KeyError:
        return SMALLTALK_PIVOT_PROMPT


# --- Unified next-question prompt ----------------------------------------------

UNIFIED_NEXT_QUESTION_PROMPT = """Generate ONE specific, helpful question to guide the user's thinking.

REQUIREMENTS:
- Ask ONE specific question that helps the user think about their plans
- Provide context or examples when helpful (e.g., "What time of year", "Which cities", "What type of activities")  
- Avoid generic questions like "Can you tell me more about your travel plans?"
- Be conversational but informative
- Focus on the highest-priority missing information
- When user shows interest but may lack knowledge, provide concrete options/suggestions

Context: Intent={intent_type}, Action={next_action_label}, Missing={missing_required_slots}, Known={known_slots}
Suggestions: {enriched_suggestions}

Special handling: If enriched_suggestions exist, incorporate them as specific choices rather than open-ended questions.
Output: Question only."""


def get_unified_next_question_prompt(**kwargs) -> str:
    try:
        # Ensure enriched_suggestions is always present
        if "enriched_suggestions" not in kwargs:
            kwargs["enriched_suggestions"] = {}
        return UNIFIED_NEXT_QUESTION_PROMPT.format(**kwargs)
    except KeyError as e:
        logger = __import__("logging").getLogger(__name__)
        logger.warning(f"Missing prompt parameter: {e}")
        return UNIFIED_NEXT_QUESTION_PROMPT
