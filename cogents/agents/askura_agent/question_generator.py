"""
Question Generator for AskuraAgent - Generates contextual questions based on conversation style and state.
"""

from typing import List, Optional

from cogents.common.llm import BaseLLMClient
from cogents.common.logging import get_logger

from .schemas import (
    AskuraState,
    ConversationContext,
    ConversationDepth,
    ConversationMomentum,
    ConversationStyle,
    UserConfidence,
)

logger = get_logger(__name__)


class QuestionGenerator:
    """Generates contextual questions based on conversation style and state."""

    def __init__(self, config, llm_client: Optional[BaseLLMClient] = None):
        """Initialize the question generator."""
        self.config = config
        self.llm = llm_client

    def generate_contextual_question(self, action: str, state: AskuraState, context: ConversationContext) -> str:
        """Generate contextual questions based on conversation style and what we know."""

        # Try LLM-based contextual question generation first
        if self.llm is not None:
            try:
                question = self._generate_llm_contextual_question(action, state, context)
                if question:
                    return question
            except Exception as e:
                logger.warning(f"LLM contextual question generation failed, falling back to templates: {e}")

        # Fallback to heuristic approach
        return self._generate_heuristic_contextual_question(action, state, context)

    def _get_base_question(self, action: str, style: ConversationStyle, depth: ConversationDepth) -> str:
        """Get the base question template for the given action, style, and depth (legacy method)."""
        # This method is now deprecated in favor of the unified contextual approach
        # but kept for backward compatibility
        return self._get_template_question(action, style, depth)

    def _generate_llm_question(self, action: str, style: ConversationStyle, depth: ConversationDepth) -> Optional[str]:
        """Generate a question using LLM based on action, style, and depth."""

        # Map enums to string descriptions for LLM
        style_map = {
            ConversationStyle.DIRECT: "direct and straightforward",
            ConversationStyle.EXPLORATORY: "curious and engaging",
            ConversationStyle.CASUAL: "casual and friendly",
        }

        depth_map = {
            ConversationDepth.SURFACE: "surface level",
            ConversationDepth.MODERATE: "moderate depth",
            ConversationDepth.DEEP: "deep and meaningful",
        }

        action_map = {
            "ask_destination": "asking about travel destination",
            "ask_dates": "asking about travel dates/timing",
            "ask_budget": "asking about budget",
            "ask_interests": "asking about interests and activities",
            "ask_group": "asking about travel companions",
            "redirect_conversation": "redirecting the conversation",
        }

        system_prompt = {
            "role": "system",
            "content": (
                "You are a travel planning assistant. Generate a natural, conversational question "
                "that matches the specified style and depth. The question should be engaging and "
                "appropriate for the given context. Return only the question, no additional text."
            ),
        }

        user_prompt = {
            "role": "user",
            "content": (
                f"Generate a {style_map.get(style, 'conversational')} question with {depth_map.get(depth, 'appropriate')} depth "
                f"for {action_map.get(action, action)}. "
                f"Make it natural and engaging for travel planning."
            ),
        }

        try:
            response = self.llm.chat_completion(messages=[system_prompt, user_prompt], temperature=0.7, max_tokens=100)

            if isinstance(response, str) and response.strip():
                return response.strip()
        except Exception as e:
            logger.warning(f"LLM question generation error: {e}")

        return None

    def _generate_llm_contextual_question(
        self, action: str, state: AskuraState, context: ConversationContext
    ) -> Optional[str]:
        """Generate a contextual question using LLM based on action, state, and context."""

        # Map enums to string descriptions for LLM
        style_map = {
            ConversationStyle.DIRECT: "direct and straightforward",
            ConversationStyle.EXPLORATORY: "curious and engaging",
            ConversationStyle.CASUAL: "casual and friendly",
        }

        depth_map = {
            ConversationDepth.SURFACE: "surface level",
            ConversationDepth.MODERATE: "moderate depth",
            ConversationDepth.DEEP: "deep and meaningful",
        }

        action_map = {
            "ask_destination": "asking about travel destination",
            "ask_dates": "asking about travel dates/timing",
            "ask_budget": "asking about budget",
            "ask_interests": "asking about interests and activities",
            "ask_group": "asking about travel companions",
            "redirect_conversation": "redirecting the conversation",
        }

        # Extract contextual information
        information_slots = state.extracted_information_slots
        contextual_info = self._extract_contextual_info(action, information_slots, context)

        # TODO (xmingc): I like the idea of letting the system hold a limited number of improvisations.
        system_prompt = {
            "role": "system",
            "content": (
                "You are a witty and creative travel planning assistant. Generate a short, precise, and inspiring question "
                "that incorporates relevant context naturally. Feel free to make slight improvisations - add wordplay, "
                "use creative language, make clever observations, or add a touch of humor when appropriate. "
                "The question should be conversational, memorable, and always encouraging. "
                "Keep it under 2 sentences but make it delightful and engaging. Return only the question, no additional text."
            ),
        }

        user_prompt = {
            "role": "user",
            "content": (
                f"Generate a {style_map.get(context.conversation_style, 'conversational')} question with "
                f"{depth_map.get(context.conversation_depth, 'appropriate')} depth for {action_map.get(action, action)}. "
                f"Context: {contextual_info}. "
                f"Feel free to be creative - use wordplay, make clever observations, or add a touch of humor "
                f"based on the context. Make it feel like a friendly, knowledgeable travel buddy is asking. "
                f"Keep it precise, inspiring, and memorable."
            ),
        }

        try:
            response = self.llm.chat_completion(messages=[system_prompt, user_prompt], temperature=0.8, max_tokens=80)

            if isinstance(response, str) and response.strip():
                return response.strip()
        except Exception as e:
            logger.warning(f"LLM contextual question generation error: {e}")

        return None

    def _extract_contextual_info(self, action: str, information_slots: dict, context: ConversationContext) -> str:
        """Extract relevant contextual information for LLM prompt."""
        context_parts = []

        # Add known information based on action
        if action == "ask_destination" and information_slots.get("interests"):
            interests = information_slots["interests"]
            if isinstance(interests, list):
                context_parts.append(f"User interests: {', '.join(interests)}")

        elif action == "ask_dates" and information_slots.get("destination"):
            context_parts.append(f"Destination: {information_slots['destination']}")

        elif action == "ask_budget" and information_slots.get("destination"):
            context_parts.append(f"Destination: {information_slots['destination']}")

        elif action == "ask_interests" and information_slots.get("destination"):
            context_parts.append(f"Destination: {information_slots['destination']}")

        elif action == "ask_group" and information_slots.get("interests"):
            interests = information_slots["interests"]
            if isinstance(interests, list):
                context_parts.append(f"User interests: {', '.join(interests)}")

        # Add conversation context
        if context.user_confidence == UserConfidence.LOW:
            context_parts.append("User seems uncertain")
        elif context.user_confidence == UserConfidence.HIGH:
            context_parts.append("User is confident")

        if context.conversation_momentum == ConversationMomentum.POSITIVE:
            context_parts.append("Conversation is flowing well")
        elif context.conversation_momentum == ConversationMomentum.NEGATIVE:
            context_parts.append("Conversation needs redirection")

        if context.information_density > 0.7:
            context_parts.append("User is providing lots of information")
        elif context.information_density < 0.3:
            context_parts.append("User is being brief")

        return "; ".join(context_parts) if context_parts else "No specific context available"

    def _generate_heuristic_contextual_question(
        self, action: str, state: AskuraState, context: ConversationContext
    ) -> str:
        """Generate contextual question using heuristic approach (fallback method)."""

        # Get the appropriate style and depth
        style = context.conversation_style
        depth = context.conversation_depth

        # Get the base question template
        base_question = self._get_template_question(action, style, depth)

        # Add contextual elements based on what we know
        contextual_elements = self._generate_contextual_elements(action, state, context)

        # Combine contextual elements with base question
        if contextual_elements:
            context_prefix = ", ".join(contextual_elements) + ", "
            return context_prefix + base_question.lower()
        else:
            return base_question

    def _get_template_question(self, action: str, style: ConversationStyle, depth: ConversationDepth) -> str:
        """Get the base question template for the given action, style, and depth (fallback method)."""

        # Default question templates
        templates = {
            "ask_destination": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "Where would you like to travel?",
                    ConversationDepth.MODERATE: "What destination are you considering for your trip?",
                    ConversationDepth.DEEP: "What kind of place speaks to your travel dreams right now?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "I'm curious about your travel dreams! What kind of destination is calling to you right now?",
                    ConversationDepth.MODERATE: "I'd love to understand your travel vision. What kind of destination resonates with what you're looking for?",
                    ConversationDepth.DEEP: "Tell me about the destination that's been on your mind. What draws you to it?",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "So, where are you thinking of going? Any place in mind?",
                    ConversationDepth.MODERATE: "Where's calling to you for this trip?",
                    ConversationDepth.DEEP: "What destination has been on your radar lately?",
                },
            },
            "ask_dates": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "When are you planning to travel?",
                    ConversationDepth.MODERATE: "What time frame are you considering for this trip?",
                    ConversationDepth.DEEP: "When do you envision this journey happening?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "When do you see yourself taking this trip? Are you thinking of a specific time or season?",
                    ConversationDepth.MODERATE: "I'm wondering about the timing. When feels right for this adventure?",
                    ConversationDepth.DEEP: "What time of year feels most aligned with what you're seeking from this trip?",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "When are you thinking of going? Any particular time?",
                    ConversationDepth.MODERATE: "When's the timing looking like for this trip?",
                    ConversationDepth.DEEP: "When do you think you'll be ready for this journey?",
                },
            },
            "ask_budget": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "What's your budget for this trip?",
                    ConversationDepth.MODERATE: "What budget range are you comfortable with?",
                    ConversationDepth.DEEP: "What investment feels right for this experience?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "I'd love to help you plan within your comfort zone. What budget range are you thinking for this adventure?",
                    ConversationDepth.MODERATE: "Let's talk about budget in a way that feels comfortable. What range works for you?",
                    ConversationDepth.DEEP: "I want to ensure this trip fits your financial comfort zone. What feels right to you?",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "What kind of budget are you working with?",
                    ConversationDepth.MODERATE: "What's your budget looking like for this?",
                    ConversationDepth.DEEP: "What budget feels comfortable for this kind of trip?",
                },
            },
            "ask_interests": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "What activities interest you most?",
                    ConversationDepth.MODERATE: "What experiences are you looking for?",
                    ConversationDepth.DEEP: "What kind of experiences would make this trip meaningful for you?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "What experiences are you most excited about for this trip? I'd love to hear what makes your heart sing!",
                    ConversationDepth.MODERATE: "I'm curious about what experiences would make this trip special for you. What calls to you?",
                    ConversationDepth.DEEP: "What experiences would make this journey truly meaningful and memorable for you?",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "What kinds of things do you like to do when you travel?",
                    ConversationDepth.MODERATE: "What activities are you thinking about for this trip?",
                    ConversationDepth.DEEP: "What experiences are you hoping to have on this journey?",
                },
            },
            "ask_group": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "Who will be traveling with you?",
                    ConversationDepth.MODERATE: "Who's in your travel party?",
                    ConversationDepth.DEEP: "Who will be sharing this journey with you?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "Who will be sharing this journey with you? I'd love to understand your travel party!",
                    ConversationDepth.MODERATE: "I'm curious about your travel companions. Who will be part of this adventure?",
                    ConversationDepth.DEEP: "Who will be experiencing this journey alongside you? I'd love to understand the group dynamic.",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "Who's coming along on this trip?",
                    ConversationDepth.MODERATE: "Who's in your travel group?",
                    ConversationDepth.DEEP: "Who will be joining you on this adventure?",
                },
            },
            "redirect_conversation": {
                ConversationStyle.DIRECT: {
                    ConversationDepth.SURFACE: "Let me ask you something different. What's most important to you for this trip?",
                    ConversationDepth.MODERATE: "Let's shift focus. What's the key thing you want from this trip?",
                    ConversationDepth.DEEP: "Let me approach this differently. What's at the heart of what you're seeking?",
                },
                ConversationStyle.EXPLORATORY: {
                    ConversationDepth.SURFACE: "I sense we might be on different pages. Let's start fresh - what's really calling to you about this trip?",
                    ConversationDepth.MODERATE: "I feel like we might be missing something. What's the essence of what you're looking for?",
                    ConversationDepth.DEEP: "I want to make sure I understand your vision. What's the deeper purpose behind this trip?",
                },
                ConversationStyle.CASUAL: {
                    ConversationDepth.SURFACE: "Maybe let's try a different approach. What's the main thing you want from this trip?",
                    ConversationDepth.MODERATE: "Let's take a step back. What's really important to you for this trip?",
                    ConversationDepth.DEEP: "I want to get this right. What's the core of what you're hoping for?",
                },
            },
        }

        # Get the template for the action and style
        style_templates = templates.get(action, {})
        depth_templates = style_templates.get(style, {})

        # Get the question for the specific depth, fallback to surface
        question = depth_templates.get(
            depth, depth_templates.get(ConversationDepth.SURFACE, "Could you tell me more about that?")
        )

        return question

    def _generate_contextual_elements(self, action: str, state: AskuraState, context: ConversationContext) -> List[str]:
        """Generate contextual elements to make questions more relevant and natural (fallback method)."""

        # Try LLM-based contextual element generation first
        if self.llm is not None:
            try:
                elements = self._generate_llm_contextual_elements(action, state, context)
                if elements:
                    return elements
            except Exception as e:
                logger.warning(f"LLM contextual elements generation failed, falling back to heuristics: {e}")

        # Fallback to simple heuristic approach
        return self._generate_simple_contextual_elements(action, state, context)

    def _generate_llm_contextual_elements(
        self, action: str, state: AskuraState, context: ConversationContext
    ) -> Optional[List[str]]:
        """Generate contextual elements using LLM based on available information."""

        information_slots = state.extracted_information_slots

        # Prepare context information for LLM
        context_info = self._prepare_context_info_for_llm(action, information_slots, context)

        system_prompt = {
            "role": "system",
            "content": (
                "You are a travel planning assistant. Generate 1-2 simple, precise contextual elements "
                "that can be naturally added to questions. These should be brief phrases that provide "
                "relevant context without derailing the conversation. Return only the elements separated by '|', "
                "or 'none' if no contextual elements are needed. Keep each element under 10 words."
            ),
        }

        user_prompt = {
            "role": "user",
            "content": (
                f"Action: {action}\n"
                f"Available information: {context_info}\n"
                f"Generate contextual elements that would make the next question more relevant and natural. "
                f"Focus on what we know about the user's preferences, destination, or conversation state."
            ),
        }

        try:
            response = self.llm.chat_completion(messages=[system_prompt, user_prompt], temperature=0.3, max_tokens=50)

            if isinstance(response, str) and response.strip():
                response = response.strip().lower()
                if response == "none" or response == "no contextual elements needed":
                    return []

                # Parse elements separated by |
                elements = [elem.strip() for elem in response.split("|") if elem.strip()]
                if elements:
                    return elements

        except Exception as e:
            logger.warning(f"LLM contextual elements generation error: {e}")

        return None

    def _prepare_context_info_for_llm(self, action: str, information_slots: dict, context: ConversationContext) -> str:
        """Prepare context information in a format suitable for LLM processing."""
        context_parts = []

        # Add relevant information based on action
        if action == "ask_destination" and information_slots.get("interests"):
            interests = information_slots["interests"]
            if isinstance(interests, list):
                context_parts.append(f"User interests: {', '.join(interests)}")

        elif action in ["ask_dates", "ask_budget", "ask_interests"] and information_slots.get("destination"):
            context_parts.append(f"Destination: {information_slots['destination']}")

        elif action == "ask_group" and information_slots.get("interests"):
            interests = information_slots["interests"]
            if isinstance(interests, list):
                context_parts.append(f"User interests: {', '.join(interests)}")

        # Add conversation state context
        if context.user_confidence == UserConfidence.LOW:
            context_parts.append("User confidence: low")
        elif context.user_confidence == UserConfidence.HIGH:
            context_parts.append("User confidence: high")

        if context.conversation_momentum == ConversationMomentum.POSITIVE:
            context_parts.append("Conversation momentum: positive")
        elif context.conversation_momentum == ConversationMomentum.NEGATIVE:
            context_parts.append("Conversation momentum: negative")

        if context.information_density > 0.7:
            context_parts.append("User is providing detailed information")
        elif context.information_density < 0.3:
            context_parts.append("User is being brief")

        return "; ".join(context_parts) if context_parts else "No specific context available"

    def _generate_simple_contextual_elements(
        self, action: str, state: AskuraState, context: ConversationContext
    ) -> List[str]:
        """Generate simple contextual elements using basic heuristics (fallback method)."""
        contextual_elements = []
        information_slots = state.extracted_information_slots

        # Simple destination context
        if action in ["ask_dates", "ask_budget", "ask_interests"] and information_slots.get("destination"):
            contextual_elements.append(f"For {information_slots['destination']}")

        # Simple confidence-boosting context
        if context.user_confidence == UserConfidence.LOW:
            if action == "ask_interests":
                contextual_elements.append("There are no wrong answers")
            elif action == "ask_budget":
                contextual_elements.append("We can work with any budget")

        # Simple momentum-boosting context
        if context.conversation_momentum == ConversationMomentum.POSITIVE:
            if action == "ask_destination":
                contextual_elements.append("I'm excited to help you plan")
            elif action == "ask_interests":
                contextual_elements.append("I love hearing about travel dreams")

        return contextual_elements
