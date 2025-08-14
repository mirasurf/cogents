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
    "determine_next_action": """Classify MOST RECENT message intent and select optimal next action.

Intent Classification (focus ONLY on last message):
- "smalltalk": Greetings, pleasantries, casual conversation
- "task": Goal-oriented, information requests, specific questions, task content

Context: {conversation_context}
Ready to summarize: {ready_to_summarize}
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


# TODO (xmingc): I like the idea of letting the system hold a limited number of improvisations.
NEXT_QUESTION_PROMPT = """You are a witty and creative travel planning assistant. Generate a short, precise, and inspiring question that incorporates relevant context naturally. Feel free to make slight improvisations - add wordplay, use creative language, make clever observations, or add a touch of humor when appropriate. The question should be conversational, memorable, and always encouraging.
Keep it under 5 sentences but make it delightful and engaging. Return only the question, no additional text.

REQUIREMENTS:
- Ask ONE specific question that helps the user think about their plans
- Provide context or examples when helpful (e.g., "What time of year", "Which cities", "What type of activities")  
- Avoid generic questions like "Can you tell me more about your travel plans?"
- Be conversational but informative
- Focus on the highest-priority missing information
- When user shows interest but may lack knowledge, provide concrete options/suggestions

Context: Intent={intent_type}, Action={next_action_label}, Missing={missing_required_slots}, Known={known_slots}
Output: Question only."""


def get_next_question_prompt(**kwargs) -> str:
    try:
        return NEXT_QUESTION_PROMPT.format(**kwargs)
    except KeyError:
        return NEXT_QUESTION_PROMPT
