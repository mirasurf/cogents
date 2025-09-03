from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel


class BaseWebPage(ABC):
    @abstractmethod
    async def goto(self, url: str, **kwargs) -> None:
        """Navigates to the specified URL."""

    @abstractmethod
    async def act(self, action_description: str, **kwargs) -> Any:
        """
        Executes an action on the page using natural language.
        This would internally use an AI model to interpret the action
        and translate it into Playwright commands.
        """

    @abstractmethod
    async def extract(
        self, schema: Union[Dict, BaseModel], selector: Optional[str] = None, **kwargs
    ) -> Union[Dict, BaseModel]:
        """
        Extracts structured data from the page based on a Pydantic-like schema.
        This would leverage an AI model to identify and parse data according to the schema.
        """

    @abstractmethod
    async def observe(self, query: str, **kwargs) -> List[Dict[str, Any]]:
        """
        Discovers available actions or elements on the page based on a natural language query.
        This would use an AI model to analyze the page and identify relevant elements.
        """

    @abstractmethod
    async def agent(
        self, prompt: str, provider: str, model: str, options: Optional[Dict] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Automates an entire workflow autonomously based on a high-level natural language prompt.
        This represents the most agentic capability, orchestrating Act, Extract, Observe internally.
        """


class BaseWebSurfer(ABC):
    """
    Base class for web surfer agents.
    """

    @abstractmethod
    async def launch(self, headless: bool = True, browser_type: str = "chromium", **kwargs) -> BaseWebPage:
        """
        Launches a new browser instance and returns a BaseWebPage.
        """

    @abstractmethod
    async def close(self):
        """Closes the browser instance."""
