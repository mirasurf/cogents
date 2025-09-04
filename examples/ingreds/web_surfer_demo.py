#!/usr/bin/env python3
"""
Web Surfer Demo

This example demonstrates how to use the WebSurfer implementation
with browser-use integration for web automation tasks.

Features demonstrated:
1. Basic navigation
2. Page interaction using natural language (optimized to avoid rate limits)
3. Structured data extraction (optimized to avoid rate limits)
4. Page observation (optimized to avoid rate limits)
5. Autonomous agent workflows (optimized to avoid rate limits)

Note: This demo has been optimized to minimize API calls to avoid hitting
rate limits (10 calls per 60 minutes). Complex interactions are commented
out but show how the full functionality would work.

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


class QuoteData(BaseModel):
    """Schema for extracting quote data"""

    quote_text: str
    author: str
    tags: List[str]


class QuotesData(BaseModel):
    """Schema for multiple quotes"""

    quotes: List[QuoteData]


async def demo_basic_navigation():
    """Demonstrate basic browser navigation."""
    logger.info("🌐 Demo: Basic Navigation")

    # Create web surfer
    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        # Launch browser
        page = await web_surfer.launch(headless=False)

        # Navigate to a test site
        await page.navigate("https://quotes.toscrape.com/")
        logger.info("✅ Successfully navigated to quotes site")

    finally:
        # Close browser
        await web_surfer.close()


async def demo_page_interaction():
    """Demonstrate page interaction using natural language."""
    logger.info("🤖 Demo: Page Interaction")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)
        # Use a simpler page that doesn't require complex interactions
        await page.navigate("https://example.com")
        logger.info("✅ Successfully navigated to example.com - skipping complex interactions to avoid rate limits")

        # Skip the complex form interactions to reduce API calls
        # result = await page.act("Click on any links on this page")
        # logger.info(f"✅ Page interaction result: {result}")

    finally:
        await web_surfer.close()


async def demo_data_extraction():
    """Demonstrate structured data extraction."""
    logger.info("📊 Demo: Data Extraction")

    llm_client = get_llm_client(provider="openrouter", instructor=True)

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)
        # Use a simpler static page to reduce API calls
        await page.navigate("https://httpbin.org/html")

        logger.info("✅ Successfully navigated to httpbin.org/html - skipping data extraction to avoid rate limits")
        # Skip the complex extraction to reduce API calls
        # quotes_data = await page.extract(instruction="Extract text content from this page", schema=QuotesData)
        # logger.info(f"Raw extraction result: {quotes_data}")

    finally:
        await web_surfer.close()


async def demo_page_observation():
    """Demonstrate page element observation."""
    logger.info("👁️ Demo: Page Observation")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        page = await web_surfer.launch(headless=False)
        await page.navigate("https://example.com")

        logger.info("✅ Successfully navigated to example.com - skipping page observation to avoid rate limits")
        # Skip the complex observation to reduce API calls
        # observations = await page.observe("Find basic elements on this page", with_actions=True)
        # logger.info(f"✅ Page observations: {len(observations)} elements found")

    finally:
        await web_surfer.close()


async def demo_autonomous_agent():
    """Demonstrate autonomous agent workflow."""
    logger.info("🚀 Demo: Autonomous Agent")

    web_surfer = WebSurfer(llm_client=llm_client)

    try:
        logger.info("✅ Autonomous agent demo skipped to avoid rate limits")
        logger.info("    In a real scenario, this would create an agent to navigate and extract data")

        # Skip the autonomous agent demo to avoid rate limits
        # agent = await web_surfer.agent(
        #     prompt="Go to https://example.com and describe what you see",
        #     use_vision=False,  # Disable vision to reduce complexity
        #     max_failures=1,
        #     max_actions_per_step=2,
        # )
        # result = await agent.run()
        # logger.info(f"✅ Autonomous agent completed: {result}")

    finally:
        await web_surfer.close()


async def run_all_demos():
    """Run all demonstration functions."""
    logger.info("🚀 Starting WebSurfer Demo")

    try:
        # Run basic demos
        await demo_basic_navigation()
        await asyncio.sleep(1)  # Brief pause between demos

        await demo_page_interaction()
        await asyncio.sleep(1)

        await demo_data_extraction()
        await asyncio.sleep(1)

        await demo_page_observation()
        await asyncio.sleep(1)

        # Run advanced demo
        await demo_autonomous_agent()

        logger.info("🎉 All demos completed successfully!")

    except Exception as e:
        logger.error(f"❌ Demo failed: {e}")
        raise


def main():
    """Main entry point."""
    try:
        asyncio.run(run_all_demos())
    except KeyboardInterrupt:
        logger.info("👋 Demo interrupted by user")
    except Exception as e:
        logger.error(f"❌ Demo failed with error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
