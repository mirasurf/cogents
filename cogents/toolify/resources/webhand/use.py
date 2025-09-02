import asyncio

from .page import Webhand

# Assuming the above code is saved as webhand/page.py


async def my_automation_script():
    wh = Webhand()
    try:
        page = await wh.launch(headless=False)  # See the browser actions
        await page.goto("https://www.google.com")

        # Use Act
        await page.act("type 'latest news' into the search bar")
        await page.act("click the search button")
        print(f"After search, URL: {page._page.url}")

        # Use Extract
        # Let's say we define a schema for search results
        class SearchResultSchema(BaseModel):
            title: str
            link: str
            snippet: str

        # This part would require advanced AI to understand search results
        # For simulation, it would be hard to extract real-time data from Google
        # but conceptually, an LLM would parse the page for these fields.
        # For a simple example, let's try to extract a 'title' from the current page
        await page._page.evaluate(
            """
            document.body.innerHTML = `
                <h1>Latest News</h1>
                <a href="https://example.com/news1">News Article 1</a>
                <p>A short snippet of news article 1...</p>
            `;
        """
        )
        first_result = await page.extract(schema={"title": str, "link": str, "snippet": str})
        print(f"First Search Result (simulated): {first_result}")

        # Use Observe
        actionable_items = await page.observe("find all links on the page")
        print(f"Found {len(actionable_items)} links.")
        # print(actionable_items) # uncomment to see all

        # Use Agent
        # A more complex scenario where the agent autonomously navigates and interacts
        # For instance, a job application agent as simulated in the WebhandPage class.
        agent_outcome = await page.agent("apply for job")
        print(f"Agent workflow complete: {agent_outcome}")

    except Exception as e:
        print(f"An error occurred during automation: {e}")
    finally:
        await wh.close()


if __name__ == "__main__":
    asyncio.run(my_automation_script())
