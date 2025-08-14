"""
Prompts for AskuraAgent - Structured prompts for conversation analysis and management.
"""


# Structured extraction prompts - optimized for structured_completion
CONVERSATION_ANALYSIS_PROMPTS = {
    "conversation_context": """Analyze the user's recent conversation to understand their communication style, preferences, conversation flow, and alignment with the conversation purpose.

Conversation Purpose: {conversation_purpose}

Look for:
- Conversation style: direct (straightforward, goal-oriented), exploratory (curious, asking questions), casual (relaxed, informal)
- Information density: how much information is packed into messages (0.0-1.0 scale)
- Conversation depth: surface (basic exchanges), moderate (some detail), deep (detailed, thoughtful)
- User confidence: low (uncertain, hesitant), medium (balanced), high (assertive, certain)
- Conversation flow: natural (organic), guided (following agent direction), user_led (user driving)
- Sentiment: positive (enthusiastic, satisfied), neutral (balanced), negative (frustrated, dissatisfied), uncertain (confused, unsure)
- Momentum: positive (building engagement), neutral (steady), negative (losing interest)
- Conversation on-track confidence: Evaluate how well the conversation aligns with its intended purpose (0.0-1.0 scale)
  * 0.0-0.3: Conversation is off-track, not addressing the purpose
  * 0.4-0.6: Conversation is partially on-track, some relevance to purpose
  * 0.7-0.8: Conversation is mostly on-track, good alignment with purpose
  * 0.9-1.0: Conversation is highly focused on the intended purpose

Consider the conversation purpose when evaluating alignment:
- Are the topics discussed relevant to the stated purpose?
- Is the conversation progressing toward the intended goal?
- Are the user's questions and responses aligned with the purpose?
- Is the conversation depth appropriate for the purpose?

Recent conversation messages:
{recent_messages}""",
    "next_action": """Determine the optimal next action for the conversation based on the current context and available options.

Consider:
- User's conversation style and preferences
- Current conversation momentum and sentiment
- Missing information priorities
- User confidence level
- Conversation alignment with purpose
- Available action options

Context information:
- Conversation purpose: {conversation_purpose}
- Conversation on-track confidence: {conversation_on_track_confidence}
- Conversation style: {conversation_style}
- Information density: {information_density}
- Conversation depth: {conversation_depth}
- User confidence: {user_confidence}
- Conversation flow: {conversation_flow}
- Sentiment: {sentiment}
- Momentum: {momentum}
- Missing information: {missing_info}

Available actions: {available_actions}

Guidelines for action selection:
- If conversation is off-track (confidence < 0.4), prioritize redirecting to purpose
- If conversation is on-track (confidence > 0.7), focus on gathering missing information
- If user confidence is low, choose confidence-boosting actions
- If momentum is negative, consider redirecting or summarizing
- Balance between staying on purpose and maintaining user engagement

Select the most appropriate next action from the available options.""",
    "determine_next_action": """Analyze the user's intent and determine the optimal next action for the conversation.

INTENT CLASSIFICATION:
First, classify the user's intent:
- "smalltalk": Greetings, pleasantries, casual conversation, social niceties
- "task": Goal-oriented conversation, information requests, specific questions, task-related content

NEXT ACTION DETERMINATION:
Based on the intent and conversation context, select the most appropriate next action.

Context information:
- Conversation purpose: {conversation_purpose}
- Conversation on-track confidence: {conversation_on_track_confidence}
- Conversation style: {conversation_style}
- Information density: {information_density}
- Conversation depth: {conversation_depth}
- User confidence: {user_confidence}
- Conversation flow: {conversation_flow}
- Sentiment: {sentiment}
- Momentum: {momentum}
- Missing information: {missing_info}
- Ready to summarize: {ready_to_summarize}

Available actions: {available_actions}

Recent user messages:
{recent_messages}

Guidelines:
- If intent is "smalltalk", respond appropriately but guide toward task
- If intent is "task", focus on gathering missing information or progressing toward goal
- If conversation is off-track (confidence < 0.4), prioritize redirecting to purpose
- If conversation is on-track (confidence > 0.7), focus on gathering missing information
- If user confidence is low, choose confidence-boosting actions
- If momentum is negative, consider redirecting or summarizing
- If ready to summarize, prioritize summary over other actions
- Balance between staying on purpose and maintaining user engagement

Provide both intent classification and next action selection with reasoning.""",
}


def get_conversation_analysis_prompt(analysis_type: str, **kwargs) -> str:
    """Get a conversation analysis prompt for the specified type."""
    prompt = CONVERSATION_ANALYSIS_PROMPTS.get(analysis_type, "")
    try:
        return prompt.format(**kwargs)
    except KeyError:
        return prompt
