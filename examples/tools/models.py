"""
Trip model for CogentNano AlphaVersion.
"""

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


class TripStatus(Enum):
    PLANNING = "planning"
    ACTIVE = "active"
    COMPLETED = "completed"
    CANCELLED = "cancelled"


class Participant(BaseModel):
    name: str
    email: Optional[str] = None
    phone: Optional[str] = None


class TripPlanContext(BaseModel):
    """Unified trip plan context containing all extracted information."""

    # Destination information
    destination: Optional[str] = Field(default=None, description="Extracted destination name or travel intent")
    location_type: Optional[str] = Field(default=None, description="Type of location (city, country, region, etc.)")
    travel_intent: Optional[str] = Field(
        default=None, description="Travel intent if no specific destination (beach, cultural, adventure, etc.)"
    )

    # Date and timing information
    start_date: Optional[str] = Field(default=None, description="Start date if specified")
    end_date: Optional[str] = Field(default=None, description="End date if specified")
    duration: Optional[str] = Field(default=None, description="Trip duration (e.g., '1 week', '3 days')")
    flexibility: Optional[str] = Field(default=None, description="Date flexibility (flexible, specific, seasonal)")
    season: Optional[str] = Field(default=None, description="Preferred season if mentioned")

    # Budget information
    budget_level: Optional[str] = Field(default=None, description="Budget level (budget, moderate, luxury)")
    min_amount: Optional[float] = Field(default=None, description="Minimum budget amount")
    max_amount: Optional[float] = Field(default=None, description="Maximum budget amount")
    currency: Optional[str] = Field(default=None, description="Currency (default USD)")
    per_person: Optional[bool] = Field(default=None, description="Whether budget is per person")

    # Interests and activities
    activities: List[str] = Field(default_factory=list, description="List of preferred activities")
    travel_style: Optional[str] = Field(
        default=None, description="Travel style (adventure, relaxation, cultural, etc.)"
    )
    must_see: List[str] = Field(default_factory=list, description="Must-see attractions or experiences")
    avoid: List[str] = Field(default_factory=list, description="Things to avoid")

    # Group information
    group_size: Optional[int] = Field(default=None, description="Number of travelers")
    group_composition: Optional[str] = Field(
        default=None, description="Group composition (solo, couple, family, friends)"
    )
    ages: Optional[str] = Field(default=None, description="Age range or specific ages")
    special_needs: List[str] = Field(default_factory=list, description="Special needs or requirements")

    # Overall confidence and metadata
    confidence: float = Field(description="Overall confidence score between 0 and 1")
    extracted_fields: List[str] = Field(
        default_factory=list, description="List of fields that were successfully extracted"
    )


class ResearchResult(BaseModel):
    """Structured research result."""

    destination: str = Field(description="Destination researched")
    summary: str = Field(description="Research summary")
    attractions: List[str] = Field(description="Key attractions")
    activities: List[str] = Field(description="Recommended activities")
    travel_tips: List[str] = Field(description="Travel tips and advice")
    best_time: Optional[str] = Field(description="Best time to visit")
    confidence: float = Field(description="Confidence in research quality")
