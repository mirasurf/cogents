"""
Trip model for CogentNano AlphaVersion.
"""

from datetime import date, datetime
from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TripStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class TripDuration(str, Enum):
    """Valid trip duration values."""

    DAYS = "days"
    WEEKS = "weeks"
    MONTHS = "months"


class TripSeason(str, Enum):
    """Valid trip season values."""

    SPRING = "spring"
    SUMMER = "summer"
    FALL = "fall"
    AUTUMN = "autumn"
    WINTER = "winter"


class TripBudget(str, Enum):
    """Valid trip budget levels."""

    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    LUXURY = "luxury"


class Participant(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None


class Trip(BaseModel):
    id: str
    name: str
    destination: str
    start_date: date
    end_date: date
    budget_min: float
    budget_max: float
    summary: str
    participants: List[Participant] = Field(default_factory=list)
    status: TripStatus = TripStatus.PLANNING
    content: str = ""
    created_at: datetime = Field(default_factory=datetime.now)
    updated_at: datetime = Field(default_factory=datetime.now)


# Structured output models
class DestinationInfo(BaseModel):
    """Structured destination information."""

    destination: Optional[str] = Field(description="Extracted destination name or travel intent")
    confidence: float = Field(description="Confidence score between 0 and 1")
    location_type: Optional[str] = Field(description="Type of location (city, country, region, etc.)")
    travel_intent: Optional[str] = Field(
        description="Travel intent if no specific destination (beach, cultural, adventure, etc.)"
    )


class DateInfo(BaseModel):
    """Structured date and duration information."""

    start_date: Optional[str] = Field(description="Start date if specified")
    end_date: Optional[str] = Field(description="End date if specified")
    duration: Optional[str] = Field(description="Trip duration (e.g., '1 week', '3 days')")
    flexibility: Optional[str] = Field(description="Date flexibility (flexible, specific, seasonal)")
    season: Optional[str] = Field(description="Preferred season if mentioned")
    confidence: float = Field(description="Confidence score between 0 and 1")


class BudgetInfo(BaseModel):
    """Structured budget information."""

    level: Optional[str] = Field(description="Budget level (budget, moderate, luxury)")
    min_amount: Optional[float] = Field(description="Minimum budget amount")
    max_amount: Optional[float] = Field(description="Maximum budget amount")
    currency: Optional[str] = Field(description="Currency (default USD)")
    per_person: Optional[bool] = Field(description="Whether budget is per person")
    confidence: float = Field(description="Confidence score between 0 and 1")


class InterestsInfo(BaseModel):
    """Structured interests and activities information."""

    activities: List[str] = Field(description="List of preferred activities")
    travel_style: Optional[str] = Field(description="Travel style (adventure, relaxation, cultural, etc.)")
    must_see: List[str] = Field(description="Must-see attractions or experiences")
    avoid: List[str] = Field(description="Things to avoid")
    confidence: float = Field(description="Confidence score between 0 and 1")


class GroupInfo(BaseModel):
    """Structured group information."""

    size: Optional[int] = Field(description="Number of travelers")
    composition: Optional[str] = Field(description="Group composition (solo, couple, family, friends)")
    ages: Optional[str] = Field(description="Age range or specific ages")
    special_needs: List[str] = Field(description="Special needs or requirements")
    confidence: float = Field(description="Confidence score between 0 and 1")


class ResearchResult(BaseModel):
    """Structured research result."""

    destination: str = Field(description="Destination researched")
    summary: str = Field(description="Research summary")
    attractions: List[str] = Field(description="Key attractions")
    activities: List[str] = Field(description="Recommended activities")
    travel_tips: List[str] = Field(description="Travel tips and advice")
    best_time: Optional[str] = Field(description="Best time to visit")
    confidence: float = Field(description="Confidence in research quality")
