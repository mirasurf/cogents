# webhand/page.py
import asyncio
from typing import Any, Dict, List, Optional, Union

from playwright.async_api import Browser as PlaywrightBrowser
from playwright.async_api import Page as PlaywrightPage
from playwright.async_api import async_playwright
from pydantic import BaseModel, Field, create_model

from cogents.base.base_websurfer import BaseWebPage, BaseWebSurfer
from cogents.common.typing_compat import override


class WebSurferPage(BaseWebPage):
    """
    Represents a single page in a browser, providing powerful web automation capabilities.
    Inspired by Stagehand's Act, Extract, Observe, and Agent primitives.
    """

    def __init__(self, playwright_page: PlaywrightPage):
        self._page = playwright_page

    @override
    async def goto(self, url: str, **kwargs) -> None:
        """Navigates to the specified URL."""
        await self._page.goto(url, **kwargs)

    @override
    async def act(self, action_description: str, **kwargs) -> Any:
        """
        Executes an action on the page using natural language.
        This would internally use an AI model to interpret the action
        and translate it into Playwright commands.
        """
        print(f"Webhand: Acting on '{action_description}'...")
        # In a real implementation, this would involve calling an LLM
        # and then executing Playwright actions.
        # For this example, let's simulate a simple action.
        if "click login button" in action_description.lower():
            await self._page.click("button:has-text('Login'), a:has-text('Login')")
            return {"status": "success", "action": "clicked login button"}
        elif "fill email" in action_description.lower() and "with" in action_description.lower():
            _, _, email = action_description.lower().partition("with")
            await self._page.fill("input[type='email'], #email", email.strip())
            return {"status": "success", "action": f"filled email with {email.strip()}"}
        else:
            print(f"Webhand: Could not interpret '{action_description}'. Falling back to generic click.")
            # A more sophisticated agent would try to find a best guess
            # For now, let's just do a generic action if not recognized
            try:
                # Attempt to click something if possible, e.g., if it's a direct element
                await self._page.click(action_description)
                return {"status": "success", "action": f"attempted direct click on {action_description}"}
            except Exception as e:
                return {"status": "failure", "error": str(e), "action": "generic click attempt failed"}

    @override
    async def extract(
        self, schema: Union[Dict, BaseModel], selector: Optional[str] = None, **kwargs
    ) -> Union[Dict, BaseModel]:
        """
        Extracts structured data from the page based on a Pydantic-like schema.
        This would leverage an AI model to identify and parse data according to the schema.
        """
        print(f"Webhand: Extracting data with schema: {schema}")
        extracted_data = {}

        # Simulate data extraction. In a real scenario, an LLM would read the page
        # and fill the schema.
        if isinstance(schema, dict):
            # Create a dynamic Pydantic model from the dictionary for validation
            DynamicSchema = create_model("DynamicSchema", **{k: (v, Field(...)) for k, v in schema.items()})
            for field_name, field_type in schema.items():
                # Simple heuristic: try to find an element with text matching the field name, or a common selector
                text_content = await self._page.locator(f"text='{field_name}' >> .. >> *:visible").first.text_content()
                if text_content:
                    extracted_data[field_name] = text_content.strip()
                elif field_name == "price":  # specific example for price
                    price_text = await self._page.locator(".price, [itemprop='price']").first.text_content()
                    if price_text:
                        try:
                            extracted_data[field_name] = float(price_text.replace("$", "").replace(",", "").strip())
                        except ValueError:
                            extracted_data[field_name] = price_text.strip()  # keep as string if conversion fails
                # Further AI processing would refine this significantly.
            try:
                return DynamicSchema(**extracted_data)
            except Exception as e:
                print(f"Webhand: Schema validation error during extraction: {e}")
                return extracted_data  # return raw data if validation fails
        elif isinstance(schema, type) and issubclass(schema, BaseModel):
            for field_name, field_info in schema.model_fields.items():
                # Similar simple heuristic for Pydantic model
                text_content = await self._page.locator(f"text='{field_name}' >> .. >> *:visible").first.text_content()
                if text_content:
                    extracted_data[field_name] = text_content.strip()
                elif field_name == "price":
                    price_text = await self._page.locator(".price, [itemprop='price']").first.text_content()
                    if price_text:
                        try:
                            extracted_data[field_name] = float(price_text.replace("$", "").replace(",", "").strip())
                        except ValueError:
                            extracted_data[field_name] = price_text.strip()
            try:
                return schema(**extracted_data)
            except Exception as e:
                print(f"Webhand: Schema validation error during extraction: {e}")
                return extracted_data
        else:
            raise TypeError("Schema must be a dictionary or a Pydantic BaseModel.")

    @override
    async def observe(self, query: str = "find clickable elements", **kwargs) -> List[Dict[str, Any]]:
        """
        Discovers available actions or elements on the page based on a natural language query.
        This would use an AI model to analyze the page and identify relevant elements.
        """
        print(f"Webhand: Observing for: '{query}'...")
        # Simulate observation. A real AI would analyze the DOM and visual cues.
        results = []
        if "clickable elements" in query.lower():
            elements = await self._page.query_selector_all("a, button, input[type='submit']")
            for i, el in enumerate(elements):
                text = (
                    await el.text_content() or await el.get_attribute("aria-label") or await el.get_attribute("value")
                )
                if text and text.strip():
                    results.append(
                        {"type": "clickable", "text": text.strip(), "selector": f"__websurfer_gen_selector_{i}"}
                    )
        elif "form fields" in query.lower():
            elements = await self._page.query_selector_all("input:not([type='hidden']), textarea, select")
            for i, el in enumerate(elements):
                label = await el.get_attribute("placeholder") or await self._page.evaluate(
                    "el => el.labels[0]?.textContent", el
                )
                if label and label.strip():
                    results.append(
                        {"type": "form_field", "label": label.strip(), "selector": f"__websurfer_gen_selector_{i}"}
                    )
        else:
            print(f"Webhand: Generic observation for '{query}'. Returning all visible text elements.")
            elements = await self._page.query_selector_all("body *:visible")
            for i, el in enumerate(elements):
                text = await el.text_content()
                if text and text.strip():
                    results.append(
                        {
                            "type": "text_element",
                            "text": text.strip()[:100],
                            "selector": f"__websurfer_gen_selector_{i}",
                        }
                    )  # Truncate for brevity

        return results

    @override
    async def agent(
        self, prompt: str, provider: str = "openai", model: str = "gpt-4", options: Optional[Dict] = None, **kwargs
    ) -> Dict[str, Any]:
        """
        Automates an entire workflow autonomously based on a high-level natural language prompt.
        This represents the most agentic capability, orchestrating Act, Extract, Observe internally.
        """
        print(f"Webhand Agent: Executing workflow for: '{prompt}' using {provider}/{model}...")
        # This is where the core AI orchestration happens.
        # The agent would repeatedly:
        # 1. Observe the page state.
        # 2. Reason about the next best action based on the prompt.
        # 3. Use `act` or `extract` to perform the action.
        # 4. Loop until the goal is achieved or a stop condition is met.

        # For this example, let's simulate a very basic agentic flow.
        history = []
        if "apply for job" in prompt.lower():
            print("Agent: Navigating to job application page...")
            # Simulate initial navigation and form filling
            await self.goto("https://www.example.com/job-application")  # Replace with a real URL
            history.append({"action": "goto", "url": "https://www.example.com/job-application"})

            observations = await self.observe("form fields")
            print(f"Agent: Found form fields: {observations}")

            await self.act("fill email with test@example.com")
            history.append({"action": "fill_email", "email": "test@example.com"})

            await self.act("fill name with John Doe")
            history.append({"action": "fill_name", "name": "John Doe"})

            await self.act("click submit button")
            history.append({"action": "click_submit"})

            # In a real scenario, the agent would evaluate if the application was successful
            # by observing success messages or new page content.
            current_url = self._page.url
            if "success" in current_url or "thank-you" in current_url:
                status = "success"
                message = "Job application simulated successfully!"
            else:
                status = "partial_success"
                message = "Job application steps executed, but final status unknown (simulated)."

            return {"status": status, "message": message, "history": history}
        else:
            return {
                "status": "unsupported_prompt",
                "message": "This agent prompt is not yet implemented in simulation.",
            }


class WebSurfer(BaseWebSurfer):
    """
    The main entry point for Webhand, providing methods to launch browsers.
    """

    def __init__(self):
        self._playwright_instance = None
        self._browser = None

    @override
    async def launch(self, headless: bool = True, browser_type: str = "chromium", **kwargs) -> BaseWebPage:
        """
        Launches a new browser instance and returns a WebSurferPage.
        """
        if self._playwright_instance is None:
            self._playwright_instance = await async_playwright().start()

        if browser_type == "chromium":
            self._browser: PlaywrightBrowser = await self._playwright_instance.chromium.launch(
                headless=headless, **kwargs
            )
        elif browser_type == "firefox":
            self._browser = await self._playwright_instance.firefox.launch(headless=headless, **kwargs)
        elif browser_type == "webkit":
            self._browser = await self._playwright_instance.webkit.launch(headless=headless, **kwargs)
        else:
            raise ValueError(f"Unsupported browser type: {browser_type}")

        page = await self._browser.new_page()
        return BaseWebPage(page)

    @override
    async def close(self):
        """Closes the browser instance and Playwright."""
        if self._browser:
            await self._browser.close()
        if self._playwright_instance:
            await self._playwright_instance.stop()


# Example Usage
async def main():
    websurfer = WebSurfer()
    try:
        page = await websurfer.launch(headless=False)  # Set to True for background execution
        await page.goto("https://www.example.com")
        print(f"Current URL: {page._page.url}")

        # --- Act ---
        await page.act("click a link that says 'More Information'")
        # Simulate clicking a non-existent element for demonstration of error handling
        # await page.act("click non-existent button")

        # --- Extract ---
        class ProductSchema(BaseModel):
            title: str
            price: float
            description: Optional[str] = None

        # Simulate a page where we can extract these
        # For example, let's inject some content if on example.com
        if "example.com" in page._page.url:
            await page._page.evaluate(
                """
                 document.body.innerHTML = `
                     <h1>Product Title</h1>
                     <p class="price">$19.99</p>
                     <p>This is a great product description.</p>
                     <button>Login</button>
                     <a href="/more-info">More Information</a>
                 `;
             """
            )

        # Extract using Pydantic model
        product_data = await page.extract(ProductSchema)
        print(f"Extracted Product Data (Pydantic): {product_data.model_dump_json(indent=2)}")

        # Extract using dictionary schema
        customer_info = await page.extract(schema={"customer_name": str, "order_id": int})
        print(f"Extracted Customer Info (Dict): {customer_info}")

        # --- Observe ---
        clickable_elements = await page.observe("find clickable elements")
        print(f"Clickable elements: {clickable_elements}")

        form_fields = await page.observe("find form fields")
        print(f"Form fields: {form_fields}")

        # --- Agent ---
        # Note: The agent method is highly simulated here and will only run a predefined flow.
        # In a real Webhand, this would involve a complex LLM interaction loop.
        agent_result = await page.agent("apply for job")
        print(f"Agent result: {agent_result}")

    except Exception as e:
        print(f"An error occurred: {e}")
    finally:
        await websurfer.close()


if __name__ == "__main__":
    asyncio.run(main())
