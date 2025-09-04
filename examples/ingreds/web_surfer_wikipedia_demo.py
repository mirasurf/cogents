#!/usr/bin/env python3
"""
Wikipedia Web Surfer Demo

This example demonstrates complex web interactions with Wikipedia using the WebSurfer
implementation with browser-use integration.

Features demonstrated:
1. Navigation to Wikipedia
2. Search functionality
3. Article navigation
4. Content extraction from Wikipedia articles
5. Following links and exploring related content

Requirements:
- OpenAI API key (set OPENAI_API_KEY environment variable)
- Browser-use library (included in thirdparty/)
"""

import asyncio
import os
import sys
from typing import List

from pydantic import BaseModel

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), "../.."))

import logging

from cogents.core.base.llm import get_llm_client
from cogents.core.base.logging import setup_logging
from cogents.ingreds.web_surfer.web_surfer import WebSurfer

# Setup logging
setup_logging()
logger = logging.getLogger(__name__)
llm_client = get_llm_client(provider="openrouter", instructor=True)


class WikipediaArticle(BaseModel):
    """Schema for extracting Wikipedia article data"""

    title: str
    summary: str
    main_sections: List[str]
    categories: List[str]


async def demo_wikipedia_search():
    """Demonstrate Wikipedia search functionality."""
    logger.info("🔍 Demo: Wikipedia Search")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)

        # Navigate to Wikipedia
        await page.navigate("https://en.wikipedia.org")
        logger.info("✅ Successfully navigated to Wikipedia")

        # Search for "Artificial Intelligence"
        search_result = await page.act("Search for 'Artificial Intelligence' using the search box")
        logger.info(f"✅ Search completed: {search_result}")

        # Wait a moment for the page to load
        await asyncio.sleep(2)

    finally:
        await web_surfer.close()


async def demo_wikipedia_article_extraction():
    """Demonstrate extracting structured data from Wikipedia articles."""
    logger.info("📊 Demo: Wikipedia Article Extraction")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)

        # Navigate directly to an AI article
        await page.navigate("https://en.wikipedia.org/wiki/Machine_learning")
        logger.info("✅ Successfully navigated to Machine Learning article")

        # Extract structured information from the article
        article_data = await page.extract(
            instruction="Extract the article title, first paragraph summary, main section headings, and categories",
            schema=WikipediaArticle,
        )

        logger.info("✅ Extracted article data:")
        if isinstance(article_data, WikipediaArticle):
            logger.info(f"  Title: {article_data.title}")
            logger.info(f"  Summary: {article_data.summary[:200]}...")
            logger.info(f"  Main sections: {', '.join(article_data.main_sections[:5])}")
            logger.info(f"  Categories: {', '.join(article_data.categories[:3])}")
        else:
            logger.info(f"  Raw data: {article_data}")

    finally:
        await web_surfer.close()


async def demo_wikipedia_navigation():
    """Demonstrate navigating between Wikipedia articles."""
    logger.info("🧭 Demo: Wikipedia Navigation")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)

        # Start at Python programming language article
        await page.navigate("https://en.wikipedia.org/wiki/Python_(programming_language)")
        logger.info("✅ Started at Python programming language article")

        # Click on a link to explore related content
        navigation_result = await page.act(
            "Click on the first link in the 'See also' section or find a link related to 'machine learning' or 'data science'"
        )
        logger.info(f"✅ Navigation result: {navigation_result}")

        # Wait for navigation
        await asyncio.sleep(2)

        # Extract information about the new page
        page_info = await page.act("Tell me what Wikipedia article we're currently viewing and provide a brief summary")
        logger.info(f"✅ Current page info: {page_info}")

    finally:
        await web_surfer.close()


async def demo_wikipedia_comparison():
    """Demonstrate comparing information from multiple Wikipedia articles."""
    logger.info("⚖️ Demo: Wikipedia Article Comparison")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)

        # Visit first article: Machine Learning
        await page.navigate("https://en.wikipedia.org/wiki/Machine_learning")
        ml_info = await page.act(
            "Extract the key definition and main applications of machine learning from this article"
        )
        logger.info(f"✅ Machine Learning info: {ml_info}")

        # Visit second article: Deep Learning
        await page.navigate("https://en.wikipedia.org/wiki/Deep_learning")
        dl_info = await page.act("Extract the key definition and main applications of deep learning from this article")
        logger.info(f"✅ Deep Learning info: {dl_info}")

        # Compare the two
        logger.info("📝 Comparison completed - in a real application, you could now analyze the differences")

    finally:
        await web_surfer.close()


async def demo_wikipedia_autonomous_research():
    """Demonstrate autonomous research on Wikipedia."""
    logger.info("🤖 Demo: Autonomous Wikipedia Research")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        # Create autonomous agent for Wikipedia research
        agent = await web_surfer.agent(
            prompt="""Go to Wikipedia and research 'Natural Language Processing'. 
            Find the main article, read the introduction, and then explore one related topic 
            by clicking on a relevant link. Summarize what you learned about both topics.""",
            use_vision=True,
            max_failures=2,
            max_actions_per_step=3,
        )

        # Run the research agent
        result = await agent.run()
        logger.info(f"✅ Research completed: {result}")

    finally:
        await web_surfer.close()


async def run_wikipedia_demos():
    """Run all Wikipedia demonstration functions."""
    logger.info("🚀 Starting Wikipedia WebSurfer Demo")

    try:
        # Run demos with delays to manage rate limits
        await demo_wikipedia_search()
        await asyncio.sleep(2)  # Brief pause between demos

        await demo_wikipedia_article_extraction()
        await asyncio.sleep(2)

        await demo_wikipedia_navigation()
        await asyncio.sleep(2)

        await demo_wikipedia_comparison()
        await asyncio.sleep(2)

        # Skip autonomous demo initially to avoid rate limits
        logger.info("🤖 Skipping autonomous research demo to manage rate limits")
        logger.info("    To enable it, uncomment the autonomous demo call below")
        # await demo_wikipedia_autonomous_research()

        logger.info("🎉 All Wikipedia demos completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


def main():
    """Main entry point."""
    try:
        asyncio.run(run_wikipedia_demos())
    except KeyboardInterrupt:
        logger.info("👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
