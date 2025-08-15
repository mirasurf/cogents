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
    "knowledge_gap_analysis": """Analyze knowledge gap and suggest next topics to help achieve conversation purpose.

**Conversation Purpose:** {conversation_purpose}

**Current Context:**
{conversation_context}

**What We Know (Extracted Information):**
{extracted_info}

**What We're Missing (Required Information):**
{missing_info}

**Retrieved Memory:**
{memory}

**Recent Conversation:**
{recent_messages}

**Instructions:**
1. Evaluate how well current knowledge aligns with the conversation purpose
2. Identify key knowledge gaps that prevent achieving the purpose
3. Suggest 3-5 specific next topics that would help bridge these gaps
4. Provide a clear summary of the overall knowledge gap
5. Consider user's conversation style and preferences when suggesting topics

**Analysis should help determine:**
- Whether we have enough information to proceed
- What critical information is still needed
- How to prioritize gathering remaining information
- Topics that would naturally engage the user based on their style""",
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
    "message_routing": """Evaluate if the user's message requires deep thinking or can be handled with a quick response to guide conversation.

**Conversation Purpose:** {conversation_purpose}

**Current User Message:** {user_message}

**Conversation Context:**
{conversation_context}

**Current Extracted Information:**
{extracted_info}

**Decision Criteria:**

Deep thinking is required IF BOTH conditions are met:
1. **Contains Purpose-Related Info**: Message contains information directly related to the conversation purpose
2. **Needs Extraction/Reflection**: Message contains specific details, facts, preferences, or decisions that should be extracted and reflected upon

Quick response is appropriate when:
- Message is casual conversation, greetings, or small talk
- Message is off-topic from the conversation purpose
- Message asks general questions without providing extractable information
- Message needs guidance to stay on topic

**Instructions:**
1. Evaluate if the message contains information related to the conversation purpose
2. Determine if the message contains extractable information that requires reflection
3. Choose routing destination: 'start_deep_thinking' if both criteria met, otherwise 'response_generator'
4. Explain your reasoning clearly""",
}


def get_conversation_analysis_prompt(analysis_type: str, **kwargs) -> str:
    """Get a conversation analysis prompt for the specified type."""
    prompt = CONVERSATION_ANALYSIS_PROMPTS.get(analysis_type, "")
    try:
        return prompt.format(**kwargs)
    except KeyError:
        return prompt


# TODO (xmingc): I like the idea of letting the system hold a limited number of improvisations.
RESPONSE_GENERATION_PROMPT = """You are a skilled conversationalist who naturally guides discussions toward valuable information while keeping the flow engaging and human-like.

**Conversation Purpose:** {conversation_purpose}
**Missing Key Information:** {missing_required_slots}

**Current Situation:**
- User's intent: {intent_type}  
- Context: {next_action_reasoning}
- What we know: {known_slots}

**Strategic Response Guidelines:**
1. **Balance natural conversation with purposeful direction** - Be genuinely conversational but strategically guide toward missing information
2. **Use storytelling and examples** - Share relevant anecdotes or scenarios that naturally lead to the information we need
3. **Ask strategic follow-up questions** - Frame questions around genuine curiosity that happens to align with our information goals
4. **Provide context and options** - When guiding toward a topic, give examples or choices to make it easier for the user to respond
5. **Build on user's interests** - Connect their current topic to the information we need to collect

**Information Collection Strategies:**
- **For destination/location info**: Share travel experiences, ask about dream places, mention interesting locations
- **For dates/timing**: Talk about seasons, upcoming events, or personal scheduling preferences  
- **For interests/preferences**: Share enthusiasm about activities, ask about past experiences, mention options
- **For logistics (budget, group size)**: Frame around planning considerations or past experiences
- **For general context**: Use open-ended questions that invite storytelling and detailed sharing

**Response Style:**
- Keep it natural and conversational (2-4 sentences)
- Show genuine interest and enthusiasm  
- Use "I" statements and personal touches
- Provide specific examples or options when helpful
- Make questions feel like natural curiosity, not interrogation
- Never include quotes or formatting around your response

**Examples of Strategic Natural Responses:**
- I love hearing about people's travel dreams! I've been fascinated by how everyone has such different ideas about the perfect getaway. What kind of places spark your imagination?
- That's exciting! You know, I've noticed timing can make such a huge difference for trips - some people love the energy of peak season while others prefer the calm of shoulder months. Any thoughts on when might work best?
- That sounds amazing! I'm curious about what draws you most - are you more of an adventure seeker, culture explorer, or maybe someone who loves to just soak up the local vibe?

**Special Cases:**
- **If no information is missing**: Focus on deeper exploration, clarification, or moving toward completion
- **If user seems hesitant**: Provide encouragement and make sharing feel easier with specific examples or options
- **If off-topic**: Gently redirect through relevant connections or shared interests

Generate a single, natural response without quotes or formatting - just the raw conversational text that feels natural while strategically moving toward the missing information we need."""


def get_response_generation_prompt(**kwargs) -> str:
    try:
        return RESPONSE_GENERATION_PROMPT.format(**kwargs)
    except KeyError:
        return RESPONSE_GENERATION_PROMPT


def get_next_question_prompt(**kwargs) -> str:
    """Backward compatibility - redirect to response generation prompt."""
    return get_response_generation_prompt(**kwargs)
