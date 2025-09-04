"""
Enhanced Trip Extraction Tool - Demonstrates context-aware extraction.

This tool shows how to use current extractions to build upon previously
extracted information rather than starting from scratch each time.
"""

from typing import Any, Dict, Optional

from cogents_core.llm import get_llm_client_instructor
from cogents_core.logging_config import get_logger
from langchain_core.tools import tool

from .models import TripPlanContext

logger = get_logger(__name__)
llm_client = get_llm_client_instructor(provider="openrouter")


@tool
def extract_trip_plan_context_enhanced(user_message: str) -> TripPlanContext:
    """
    Extract comprehensive trip planning information from user messages with context awareness.

    This enhanced version uses previously extracted information to build upon
    rather than starting from scratch each time, making it more effective for
    conversations where information is distributed across multiple messages.

    Args:
        user_message: The user's travel request message to analyze
        current_extractions: Optional dict containing previously extracted information
                           from the conversation state

    Returns:
        TripPlanContext with enhanced extraction using conversation context
    """
    logger.info(f"Extracting trip context from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]

        # For LangChain tools, we'll extract context from the tool invocation
        # The enhanced information extractor will provide context through the tool_context
        context_info = ""

        prompt = f"""Extract comprehensive trip planning information from the user's travel request.

{context_info}

Analyze the message for ALL relevant trip planning details including:

DESTINATION & LOCATION:
- Specific destinations (countries, cities, regions)
- Travel intent when no specific destination mentioned (beach vacation, cultural trip, adventure travel)
- Location preferences (tropical, mountainous, urban, rural)

DATES & TIMING:
- Specific dates (MM/DD/YYYY, Month Day Year, etc.)
- Duration (3 days, 1 week, 2 weeks, etc.)
- Flexibility indicators (flexible, specific dates, around X time)
- Seasonal preferences (summer, winter, spring, fall)
- Relative timing (next month, in 3 months, etc.)

BUDGET & COSTS:
- Specific amounts ($1000, €2000, etc.)
- Budget levels (budget, cheap, moderate, luxury, expensive)
- Per person vs total budget indicators
- Currency mentions
- Budget ranges ($1000-2000)

INTERESTS & ACTIVITIES:
- Specific activities (hiking, museums, food tours, shopping)
- Travel styles (adventure, relaxation, cultural, luxury)
- Must-see attractions or experiences
- Things they want to avoid
- Interest categories (history, nature, nightlife, etc.)

GROUP COMPOSITION:
- Number of travelers (specific numbers or indicators like "couple", "family")
- Group composition (solo, couple, family, friends, colleagues)
- Age information (ages of travelers, age ranges)
- Special needs (accessibility, dietary restrictions, etc.)

User message: {safe_msg}

Extract all available information into the structured format. If there's existing context, 
build upon it rather than replacing it. Set confidence based on how much information 
was clearly extractable from the message."""

        result: TripPlanContext = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=TripPlanContext,
            temperature=0.2,
            max_tokens=1000,
        )

        # Log extraction summary
        extracted_fields = []
        if result.destination:
            extracted_fields.append("destination")
        if result.start_date or result.end_date or result.duration:
            extracted_fields.append("dates")
        if result.budget_level or result.min_amount or result.max_amount:
            extracted_fields.append("budget")
        if result.activities or result.travel_style:
            extracted_fields.append("interests")
        if result.group_size or result.group_composition:
            extracted_fields.append("group")

        result.extracted_fields = extracted_fields

        logger.info(f"Enhanced extraction - Fields: {extracted_fields}, Confidence: {result.confidence}")
        return result

    except Exception as e:
        logger.error(f"Error in enhanced trip extraction: {e}")
        raise RuntimeError(f"Error extracting trip plan context: {e}")


def extract_trip_plan_context_simple(
    user_message: str, current_extractions: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """
    Simple callable version of the enhanced trip extraction tool.

    This demonstrates how a regular function can also use the context
    and be called by the information extractor.

    Args:
        user_message: The user's message to extract from
        current_extractions: Optional current extraction state

    Returns:
        Dict containing extracted trip information
    """
    try:
        # Build context-aware extraction prompt
        context_info = ""
        if current_extractions and current_extractions.get("trip_plan_context"):
            existing_context = current_extractions["trip_plan_context"]
            context_info = f"""
CURRENTLY EXTRACTED INFORMATION:
{existing_context}

Use this existing information as a foundation and extract any NEW or UPDATED information from the current message.
If the current message provides more specific details than what's already extracted, update those fields.
If the current message provides completely new information, add it to the existing context.
"""

        # Create a context-aware prompt
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]

        prompt = f"""Extract comprehensive trip planning information from the user's travel request.

{context_info}

Analyze the message for ALL relevant trip planning details including:

DESTINATION & LOCATION:
- Specific destinations (countries, cities, regions)
- Travel intent when no specific destination mentioned (beach vacation, cultural trip, adventure travel)
- Location preferences (tropical, mountainous, urban, rural)

DATES & TIMING:
- Specific dates (MM/DD/YYYY, Month Day Year, etc.)
- Duration (3 days, 1 week, 2 weeks, etc.)
- Flexibility indicators (flexible, specific dates, around X time)
- Seasonal preferences (summer, winter, spring, fall)
- Relative timing (next month, in 3 months, etc.)

BUDGET & COSTS:
- Specific amounts ($1000, €2000, etc.)
- Budget levels (budget, cheap, moderate, luxury, expensive)
- Per person vs total budget indicators
- Currency mentions
- Budget ranges ($1000-2000)

INTERESTS & ACTIVITIES:
- Specific activities (hiking, museums, food tours, shopping)
- Travel styles (adventure, relaxation, cultural, luxury)
- Must-see attractions or experiences
- Things they want to avoid
- Interest categories (history, nature, nightlife, etc.)

GROUP COMPOSITION:
- Number of travelers (specific numbers or indicators like "couple", "family")
- Group composition (solo, couple, family, friends, colleagues)
- Age information (ages of travelers, age ranges)
- Special needs (accessibility, dietary restrictions, etc.)

User message: {safe_msg}

Extract all available information into the structured format. If there's existing context, 
build upon it rather than replacing it. Set confidence based on how much information 
was clearly extractable from the message."""

        result: TripPlanContext = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=TripPlanContext,
            temperature=0.2,
            max_tokens=1000,
        )

        # Log extraction summary
        extracted_fields = []
        if result.destination:
            extracted_fields.append("destination")
        if result.start_date or result.end_date or result.duration:
            extracted_fields.append("dates")
        if result.budget_level or result.min_amount or result.max_amount:
            extracted_fields.append("budget")
        if result.activities or result.travel_style:
            extracted_fields.append("interests")
        if result.group_size or result.group_composition:
            extracted_fields.append("group")

        result.extracted_fields = extracted_fields

        logger.info(f"Enhanced extraction - Fields: {extracted_fields}, Confidence: {result.confidence}")
        return result.model_dump() if hasattr(result, "model_dump") else dict(result)

    except Exception as e:
        logger.error(f"Error in simple trip extraction: {e}")
        return {}


# Example usage in conversation flow:
#
# Message 1: "I want to go to Japan"
# - Extracts: destination="Japan"
#
# Message 2: "In spring next year"
# - Uses context: destination="Japan"
# - Extracts: destination="Japan", season="spring", year="next year"
#
# Message 3: "With a budget of $3000"
# - Uses context: destination="Japan", season="spring", year="next year"
# - Extracts: destination="Japan", season="spring", year="next year", budget_level="moderate", min_amount=3000
#
# This builds up the complete picture across multiple messages rather than
# trying to extract everything from each individual message.
