from langchain_core.tools import tool

from cogents.common.llm import get_llm_client_instructor
from cogents.common.logging import get_logger

from .models import TripPlanContext

logger = get_logger(__name__)
llm_client = get_llm_client_instructor(provider="openrouter")


@tool
def extract_trip_plan_context(user_message: str) -> TripPlanContext:
    """
    Extract comprehensive trip planning information from user messages in a single LLM request.

    Args:
        user_message: The user's travel request message to analyze.
                     Examples: "Planning a 2-week trip to Japan in spring 2024 for $3000,
                     love cultural experiences and food", "Looking for a budget beach vacation
                     for a family of 4 this summer"

    Returns:
        Dictionary containing comprehensive trip plan context:
        - destination: Specific location or travel intent
        - location_type: Type of location (city, country, region, etc.)
        - travel_intent: Travel intent category (beach, cultural, adventure, etc.)
        - start_date/end_date: Specific dates if mentioned
        - duration: Trip duration (e.g., "2 weeks", "5 days")
        - flexibility: Date flexibility level
        - season: Seasonal preference
        - budget_level: Budget category (budget, moderate, luxury)
        - min_amount/max_amount: Budget range
        - currency: Currency preference
        - per_person: Whether budget is per person
        - activities: List of preferred activities
        - travel_style: Overall travel style
        - must_see: Must-visit attractions
        - avoid: Things to avoid
        - group_size: Number of travelers
        - group_composition: Group type (solo, couple, family, friends)
        - ages: Age information
        - special_needs: Special requirements
        - confidence: Overall extraction confidence
        - extracted_fields: List of successfully extracted fields

    Example:
        Input: "Planning a 2-week cultural trip to Japan in spring 2024 for $3000, couple"
        Output: {
            "destination": "Japan",
            "location_type": "country",
            "duration": "2 weeks",
            "season": "spring",
            "budget_level": "moderate",
            "min_amount": 3000,
            "currency": "USD",
            "travel_style": "cultural",
            "group_size": 2,
            "group_composition": "couple",
            "confidence": 0.9,
            "extracted_fields": ["destination", "duration", "season", "budget", "group"]
        }
    """
    logger.info(f"Extracting comprehensive trip context from: {user_message[:100]}...")

    try:
        safe_msg = (user_message or "").strip()
        if len(safe_msg) > 4000:
            safe_msg = safe_msg[:4000]

        # Build comprehensive extraction prompt
        prompt = f"""Extract comprehensive trip planning information from the user's travel request.

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

Extract all available information into the structured format. Set confidence based on how much information was clearly extractable from the message."""

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

        logger.info(f"Extracted comprehensive context - Fields: {extracted_fields}, Confidence: {result.confidence}")
        return result

    except Exception as e:
        raise RuntimeError(f"Error extracting trip plan context: {e}")
