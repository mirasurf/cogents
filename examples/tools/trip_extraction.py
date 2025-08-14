from typing import Any, Dict

from langchain_core.tools import tool

from cogents.common.llm import get_llm_client_instructor
from cogents.common.logging import get_logger

from .models import BudgetInfo, DateInfo, DestinationInfo, GroupInfo, InterestsInfo

logger = get_logger(__name__)
llm_client = get_llm_client_instructor(provider="openrouter")

__all__ = [
    "extract_destination_info",
    "extract_date_info",
    "extract_budget_info",
    "extract_interests_info",
    "extract_group_info",
]


# Structured extraction prompts - optimized for structured_completion
EXTRACTION_PROMPTS = {
    "destination": """Extract destination information from the user's message.

Look for:
- Specific destinations (countries, cities, regions)
- Travel intent when no specific destination is mentioned (beach vacation, cultural trip, adventure travel)
- Location preferences (tropical, mountainous, urban, rural)

User message: {user_message}""",
    "dates": """Extract date and timing information from the user's message.

Look for:
- Specific dates (MM/DD/YYYY, Month Day Year, etc.)
- Duration (3 days, 1 week, 2 weeks, etc.)
- Flexibility indicators (flexible, specific dates, around X time)
- Seasonal preferences (summer, winter, spring, fall)
- Relative timing (next month, in 3 months, etc.)

User message: {user_message}""",
    "budget": """Extract budget information from the user's message.

Look for:
- Specific amounts ($1000, €2000, etc.)
- Budget levels (budget, cheap, moderate, luxury, expensive)
- Per person vs total budget indicators
- Currency mentions
- Budget ranges ($1000-2000)

User message: {user_message}""",
    "interests": """Extract interests and activity preferences from the user's message.

Look for:
- Specific activities (hiking, museums, food tours, shopping)
- Travel styles (adventure, relaxation, cultural, luxury)
- Must-see attractions or experiences
- Things they want to avoid
- Interest categories (history, nature, nightlife, etc.)

User message: {user_message}""",
    "group": """Extract group composition information from the user's message.

Look for:
- Number of travelers (specific numbers or indicators like "couple", "family")
- Group composition (solo, couple, family, friends, colleagues)
- Age information (ages of travelers, age ranges)
- Special needs (accessibility, dietary restrictions, etc.)

User message: {user_message}""",
}


def get_extraction_prompt(extraction_type: str, **kwargs) -> str:
    """Get an extraction prompt for the specified type."""
    prompt = EXTRACTION_PROMPTS.get(extraction_type, "")
    try:
        return prompt.format(**kwargs)
    except KeyError:
        return prompt


@tool
def extract_destination_info(user_message: str) -> Dict[str, Any]:
    """
    Extract destination and location information from user travel requests.

    This tool analyzes user messages to identify specific destinations, travel intentions,
    and location preferences. It can detect both explicit destinations (cities, countries)
    and implicit travel intents (beach vacation, cultural trip, adventure travel).

    Args:
        user_message: The user's travel request message to analyze.
                      Examples: "I want to visit Japan", "Looking for a beach vacation",
                      "Planning a cultural trip to Europe"

    Returns:
        Dictionary containing:
        - destination: Specific location name or travel intent description
        - confidence: Confidence score (0.0-1.0) in the extraction
        - location_type: "specific" for named places, "intent" for travel types
        - travel_intent: Category of travel (beach, cultural, adventure, etc.)

    Example outputs:
        - "I want to visit Tokyo" → {"destination": "Tokyo", "confidence": 0.9, "location_type": "specific"}
        - "Looking for a beach vacation" → {"destination": "Beach destination", "confidence": 0.8, "travel_intent": "beach"}
    """
    logger.info(f"Extracting destination info from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]
        prompt = get_extraction_prompt("destination", user_message=safe_msg)

        result: DestinationInfo = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=DestinationInfo,
            temperature=0.3,
            max_tokens=500,
        )

        logger.info(f"Extracted destination: {result.destination} with confidence {result.confidence}")
        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error extracting destination info: {e}")


@tool
def extract_date_info(user_message: str) -> Dict[str, Any]:
    """
    Extract travel dates, duration, and timing preferences from user messages.

    This tool identifies specific dates, trip duration, seasonal preferences, and
    flexibility indicators in travel requests. It handles various date formats and
    relative time expressions commonly used in travel planning.

    Args:
        user_message: The user's travel request message to analyze.
                      Examples: "Planning a trip for June 2024", "Want to travel for 2 weeks",
                      "Looking for a summer vacation", "Flexible with dates"

    Returns:
        Dictionary containing:
        - start_date: Specific start date (if mentioned)
        - end_date: Specific end date (if mentioned)
        - duration: Trip duration (e.g., "2 weeks", "5 days")
        - flexibility: Flexibility level ("flexible", "specific", "around")
        - season: Seasonal preference (spring, summer, fall, winter)
        - confidence: Confidence score (0.0-1.0) in the extraction

    Example outputs:
        - "Planning a trip for June 2024" → {"season": "summer", "confidence": 0.8}
        - "Want to travel for 2 weeks" → {"duration": "2 weeks", "confidence": 0.9}
        - "Flexible with dates" → {"flexibility": "flexible", "confidence": 0.9}
    """
    logger.info(f"Extracting date info from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]
        prompt = get_extraction_prompt("dates", user_message=safe_msg)

        result: DateInfo = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=DateInfo,
            temperature=0.3,
            max_tokens=500,
        )

        logger.info(f"Extracted dates: {result.start_date} to {result.end_date}, duration: {result.duration}")
        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error extracting date info: {e}")


@tool
def extract_budget_info(user_message: str) -> Dict[str, Any]:
    """
    Extract budget and cost preferences from user travel requests.

    This tool identifies budget levels, specific amounts, currency preferences, and
    cost-related indicators in travel planning messages. It can detect both explicit
    amounts and qualitative budget descriptions.

    Args:
        user_message: The user's travel request message to analyze.
                      Examples: "Budget of $2000", "Looking for luxury travel",
                      "Want something affordable", "Around €1500 per person"

    Returns:
        Dictionary containing:
        - level: Budget category ("budget", "moderate", "luxury")
        - min_amount: Minimum budget amount (numeric)
        - max_amount: Maximum budget amount (numeric)
        - currency: Currency code (USD, EUR, etc.)
        - per_person: Whether budget is per person (True/False)
        - confidence: Confidence score (0.0-1.0) in the extraction

    Example outputs:
        - "Budget of $2000" → {"level": "moderate", "min_amount": 2000, "currency": "USD", "confidence": 0.9}
        - "Looking for luxury travel" → {"level": "luxury", "confidence": 0.8}
        - "Around €1500 per person" → {"min_amount": 1500, "currency": "EUR", "per_person": True, "confidence": 0.9}
    """
    logger.info(f"Extracting budget info from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]
        prompt = get_extraction_prompt("budget", user_message=safe_msg)

        result: BudgetInfo = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=BudgetInfo,
            temperature=0.3,
            max_tokens=500,
        )

        logger.info(f"Extracted budget: {result.level}, range: {result.min_amount}-{result.max_amount}")
        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error extracting budget info: {e}")


@tool
def extract_interests_info(user_message: str) -> Dict[str, Any]:
    """
    Extract travel interests, activities, and preferences from user messages.

    This tool identifies specific activities, travel styles, must-see attractions,
    and preferences that travelers mention in their requests. It helps understand
    what experiences and activities the user is looking for.

    Args:
        user_message: The user's travel request message to analyze.
                      Examples: "Love hiking and nature", "Interested in food and culture",
                      "Want to avoid touristy places", "Looking for adventure activities"

    Returns:
        Dictionary containing:
        - activities: List of specific activities (hiking, museums, food tours, etc.)
        - travel_style: Overall travel style (adventure, cultural, relaxation, luxury)
        - must_see: List of must-visit attractions or experiences
        - avoid: List of things to avoid or skip
        - confidence: Confidence score (0.0-1.0) in the extraction

    Example outputs:
        - "Love hiking and nature" → {"activities": ["hiking", "nature"], "travel_style": "adventure", "confidence": 0.9}
        - "Interested in food and culture" → {"activities": ["food", "culture"], "travel_style": "cultural", "confidence": 0.8}
        - "Want to avoid touristy places" → {"avoid": ["touristy places"], "confidence": 0.7}
    """
    logger.info(f"Extracting interests info from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]
        prompt = get_extraction_prompt("interests", user_message=safe_msg)

        result: InterestsInfo = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=InterestsInfo,
            temperature=0.3,
            max_tokens=500,
        )

        logger.info(f"Extracted {len(result.activities)} activities and style: {result.travel_style}")
        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error extracting interests info: {e}")


@tool
def extract_group_info(user_message: str) -> Dict[str, Any]:
    """
    Extract group composition, size, and traveler information from user messages.

    This tool identifies the number of travelers, group composition (solo, couple,
    family, friends), age information, and special needs or requirements mentioned
    in travel requests. This helps tailor recommendations to the specific group.

    Args:
        user_message: The user's travel request message to analyze.
                      Examples: "Traveling with my family", "Solo trip for 2 people",
                      "Couple looking for romantic getaway", "Group of 6 friends"

    Returns:
        Dictionary containing:
        - size: Number of travelers (integer)
        - composition: Group type ("solo", "couple", "family", "friends", "colleagues")
        - ages: Age information or ranges (if mentioned)
        - special_needs: List of special requirements (accessibility, dietary, etc.)
        - confidence: Confidence score (0.0-1.0) in the extraction

    Example outputs:
        - "Traveling with my family" → {"composition": "family", "confidence": 0.9}
        - "Solo trip for 2 people" → {"size": 2, "composition": "friends", "confidence": 0.8}
        - "Couple looking for romantic getaway" → {"size": 2, "composition": "couple", "confidence": 0.9}
    """
    logger.info(f"Extracting group info from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]
        prompt = get_extraction_prompt("group", user_message=safe_msg)

        result: GroupInfo = llm_client.structured_completion(
            messages=[{"role": "user", "content": prompt}],
            response_model=GroupInfo,
            temperature=0.3,
            max_tokens=500,
        )

        logger.info(f"Extracted group: {result.size} people, composition: {result.composition}")
        return result.model_dump()

    except Exception as e:
        raise RuntimeError(f"Error extracting group info: {e}")
