import asyncio
import logging
from typing import Any, Dict, List, Optional, Union

from pydantic import BaseModel

from cogents.core.base.base_websurfer import BaseWebPage, BaseWebSurfer, ObserveResult
from cogents.core.base.llm import BaseLLMClient
from cogents.core.base.typing_compat import override

try:
    from browser_use import Agent, BrowserSession, Tools
    from browser_use.agent.views import AgentSettings
except ImportError as e:
    raise ImportError(f"Failed to import browser-use components: {e}")

logger = logging.getLogger(__name__)


class BrowserUseLLMAdapter:
    """Adapter to make cogents LLM clients compatible with browser-use."""

    def __init__(self, cogents_client: BaseLLMClient):
        self.cogents_client = cogents_client
        self.model = getattr(cogents_client, "model", "unknown")
        self._verified_api_keys = True  # Assume the cogents client is properly configured

    @property
    def provider(self) -> str:
        """Return the provider name."""
        return getattr(self.cogents_client, "provider", "cogents")

    @property
    def name(self) -> str:
        """Return the model name."""
        return self.model

    @property
    def model_name(self) -> str:
        """Return the model name for legacy support."""
        return self.model

    async def ainvoke(self, messages: List[Any], output_format: Optional[type] = None, **kwargs) -> Any:
        """Invoke the LLM with messages."""
        try:
            # Convert browser-use messages to cogents format
            cogents_messages = []
            for msg in messages:
                if hasattr(msg, "role"):
                    # Extract text content properly from browser-use message objects
                    content_text = ""
                    if hasattr(msg, "text"):
                        # Use the convenient .text property that handles both string and list formats
                        content_text = msg.text
                    elif hasattr(msg, "content"):
                        # Fallback: handle content directly
                        if isinstance(msg.content, str):
                            content_text = msg.content
                        elif isinstance(msg.content, list):
                            # Extract text from content parts
                            text_parts = []
                            for part in msg.content:
                                if hasattr(part, "text") and hasattr(part, "type") and part.type == "text":
                                    text_parts.append(part.text)
                            content_text = "\n".join(text_parts)
                        else:
                            content_text = str(msg.content)
                    else:
                        content_text = str(msg)

                    cogents_messages.append({"role": msg.role, "content": content_text})
                elif isinstance(msg, dict):
                    # Already in the right format
                    cogents_messages.append(msg)
                else:
                    # Handle other message formats
                    cogents_messages.append({"role": "user", "content": str(msg)})

            # Return response wrapped in expected format
            from browser_use.llm.views import ChatInvokeCompletion, ChatInvokeUsage

            # Create usage information (dummy values for now)
            usage = ChatInvokeUsage(
                prompt_tokens=0,
                prompt_cached_tokens=None,
                prompt_cache_creation_tokens=None,
                prompt_image_tokens=None,
                completion_tokens=0,
                total_tokens=0,
            )

            # Choose completion method based on output_format
            if output_format is not None:
                # Use structured completion for structured output
                try:
                    if hasattr(self.cogents_client, "structured_completion"):
                        if asyncio.iscoroutinefunction(self.cogents_client.structured_completion):
                            structured_response = await self.cogents_client.structured_completion(
                                cogents_messages, output_format
                            )
                        else:
                            structured_response = self.cogents_client.structured_completion(
                                cogents_messages, output_format
                            )
                        return ChatInvokeCompletion(completion=structured_response, usage=usage)
                    else:
                        # Fall back to regular completion + JSON parsing if structured_completion not available
                        if asyncio.iscoroutinefunction(self.cogents_client.completion):
                            response = await self.cogents_client.completion(cogents_messages)
                        else:
                            response = self.cogents_client.completion(cogents_messages)

                        # Try to parse as JSON and create structured object
                        import json

                        response_str = str(response)
                        try:
                            parsed_data = json.loads(response_str)
                            if isinstance(parsed_data, dict):
                                parsed_object = output_format(**parsed_data)
                                return ChatInvokeCompletion(completion=parsed_object, usage=usage)
                            else:
                                raise ValueError("Parsed JSON is not a dictionary")
                        except (json.JSONDecodeError, ValueError, TypeError) as parse_error:
                            logger.error(
                                f"Failed to parse response as JSON for {output_format.__name__}: {parse_error}"
                            )
                            logger.error(f"Response content: {response_str}")
                            # Create minimal fallback structured object
                            if hasattr(output_format, "model_fields") and "action" in output_format.model_fields:
                                fallback_data = {
                                    "thinking": f"Parse error: {str(parse_error)}",
                                    "evaluation_previous_goal": "Unable to parse structured response",
                                    "memory": response_str[:500],  # Truncate for safety
                                    "next_goal": "Retry with simpler approach",
                                    "action": [],
                                }
                                try:
                                    parsed_object = output_format(**fallback_data)
                                    return ChatInvokeCompletion(completion=parsed_object, usage=usage)
                                except Exception:
                                    pass
                            raise parse_error

                except Exception as e:
                    logger.error(f"Error in structured completion: {e}")
                    raise
            else:
                # Use regular completion for string output
                if asyncio.iscoroutinefunction(self.cogents_client.completion):
                    response = await self.cogents_client.completion(cogents_messages)
                else:
                    response = self.cogents_client.completion(cogents_messages)

                return ChatInvokeCompletion(completion=str(response), usage=usage)

        except Exception as e:
            logger.error(f"Error in LLM adapter: {e}")
            raise


class WebSurferPage(BaseWebPage):
    """Web page implementation using browser-use."""

    def __init__(self, browser_session: BrowserSession, llm_client=None):
        self.browser_session = browser_session
        self.llm_client = llm_client
        self.tools = None
        if llm_client:
            self.tools = Tools()

    @override
    async def navigate(self, url: str, **kwargs) -> None:
        """Navigates to the specified URL."""
        try:
            # Navigate using browser session event system
            from browser_use.browser.events import NavigateToUrlEvent

            event = self.browser_session.event_bus.dispatch(NavigateToUrlEvent(url=url))
            await event
            await event.event_result(raise_if_any=True, raise_if_none=False)

            logger.info(f"Successfully navigated to {url}")
        except Exception as e:
            logger.error(f"Failed to navigate to {url}: {e}")
            raise

    @override
    async def act(self, instruction: str, observe_results: Optional[ObserveResult] = None, **kwargs) -> Any:
        """
        Executes an action on the page using natural language.
        Uses browser-use Agent for autonomous action execution.
        """
        try:
            if not self.llm_client:
                raise ValueError("LLM client is required for action execution")

            # Create a browser-use compatible LLM adapter
            llm_adapter = BrowserUseLLMAdapter(self.llm_client)

            # Create agent for this specific action
            agent = Agent(
                task=instruction,
                llm=llm_adapter,
                browser=self.browser_session,
                settings=AgentSettings(use_vision=True, max_failures=2, max_actions_per_step=3),
            )

            # Execute the action
            history = await agent.run()

            # Return the final result
            final_result = history.final_result() if history else None
            logger.info(f"Action '{instruction}' completed with result: {final_result}")

            return final_result

        except Exception as e:
            logger.error(f"Failed to execute action '{instruction}': {e}")
            raise

    @override
    async def extract(
        self, instruction: str, schema: Union[Dict, BaseModel], selector: Optional[str] = None, **kwargs
    ) -> Union[Dict, BaseModel]:
        """
        Extracts structured data from the page based on a Pydantic-like schema.
        """
        try:
            if not self.llm_client:
                raise ValueError("LLM client is required for data extraction")

            # Create a browser-use compatible LLM adapter
            llm_adapter = BrowserUseLLMAdapter(self.llm_client)

            # Prepare task instruction
            task_instruction = f"Extract data from the current page: {instruction}"
            if selector:
                task_instruction += f" Focus on elements matching selector: {selector}"

            # Determine output model
            output_model = None
            if isinstance(schema, type) and issubclass(schema, BaseModel):
                output_model = schema
            elif isinstance(schema, dict):
                # Convert dict schema to Pydantic model if needed
                # For now, we'll extract as text and structure it
                pass

            # Create agent for extraction
            agent = Agent(
                task=task_instruction,
                llm=llm_adapter,
                browser=self.browser_session,
                output_model_schema=output_model,
                settings=AgentSettings(use_vision=True, max_failures=2),
            )

            # Execute extraction
            history = await agent.run()
            result = history.final_result() if history else None

            if output_model and result:
                try:
                    # Try to parse as structured output
                    if isinstance(result, str):
                        return output_model.model_validate_json(result)
                    elif isinstance(result, dict):
                        return output_model.model_validate(result)
                    else:
                        return result
                except Exception as parse_error:
                    logger.warning(f"Failed to parse structured output: {parse_error}")
                    # Fall back to text extraction
                    return {"page_text": str(result)}

            # Return as dict or original schema format
            if isinstance(schema, dict):
                return {"page_text": str(result)} if result else {}
            elif isinstance(result, str):
                return {"page_text": result}
            else:
                return result or {}

        except Exception as e:
            logger.error(f"Failed to extract data with instruction '{instruction}': {e}")
            raise

    @override
    async def observe(self, instruction: str, with_actions: bool = True, **kwargs) -> List[ObserveResult]:
        """
        Discovers available actions or elements on the page based on a natural language query.
        """
        try:
            if not self.llm_client:
                raise ValueError("LLM client is required for page observation")

            # Create a browser-use compatible LLM adapter
            llm_adapter = BrowserUseLLMAdapter(self.llm_client)

            # Create observation task
            observe_task = f"Analyze the current page and identify elements that match: {instruction}"
            if with_actions:
                observe_task += " Provide recommended actions for each identified element."

            # Create agent for observation
            agent = Agent(
                task=observe_task,
                llm=llm_adapter,
                browser=self.browser_session,
                settings=AgentSettings(
                    use_vision=True, max_failures=1, max_actions_per_step=1  # Just observe, don't act
                ),
            )

            # Execute observation
            history = await agent.run()
            result = history.final_result() if history else None

            # Parse result into ObserveResult format
            observe_results = []
            if result:
                # This is a simplified parsing - in a real implementation,
                # you might want to use browser-use's DOM inspection capabilities
                observe_results.append(
                    ObserveResult(
                        selector="xpath=//body",  # Placeholder selector
                        description=str(result),
                        backendNodeId=0,  # Placeholder
                        method="click" if with_actions else "observe",
                        arguments=[] if not with_actions else [""],
                    )
                )

            logger.info(f"Observation completed, found {len(observe_results)} elements")
            return observe_results

        except Exception as e:
            logger.error(f"Failed to observe page with instruction '{instruction}': {e}")
            raise


class WebSurfer(BaseWebSurfer):
    """Web surfer implementation using browser-use."""

    def __init__(self, llm_client=None):
        self.llm_client = llm_client
        self.browser_session = None
        self._browser = None

    @override
    async def launch(self, headless: bool = True, browser_type: str = "chromium", **kwargs) -> BaseWebPage:
        """
        Launches a new browser instance and returns a BaseWebPage.
        """
        try:
            # Create browser instance with configuration
            self._browser = BrowserSession(headless=headless, **kwargs)

            # Launch browser
            await self._browser.start()

            self.browser_session = self._browser

            # Create and return WebSurferPage
            page = WebSurferPage(self.browser_session, self.llm_client)

            logger.info(f"Browser launched successfully (headless={headless}, type={browser_type})")
            return page

        except Exception as e:
            logger.error(f"Failed to launch browser: {e}")
            raise

    @override
    async def close(self):
        """Closes the browser instance."""
        try:
            if self.browser_session:
                await self.browser_session.stop()
                self.browser_session = None
                self._browser = None
                logger.info("Browser closed successfully")
        except Exception as e:
            logger.error(f"Failed to close browser: {e}")
            raise

    @override
    async def agent(self, prompt: str, **kwargs) -> "Agent":
        """
        Creates an autonomous agent that can execute complex web workflows.
        Returns the browser-use Agent directly for full functionality access.
        """
        try:
            if not self.llm_client:
                raise ValueError("LLM client is required for autonomous agent")

            if not self.browser_session:
                # Auto-launch browser if not already launched
                await self.launch(headless=kwargs.get("headless", True))

            # Create browser-use compatible LLM adapter
            llm_adapter = BrowserUseLLMAdapter(self.llm_client)

            # Create browser-use agent
            browser_use_agent = Agent(
                task=prompt,
                llm=llm_adapter,
                browser=self.browser_session,
                settings=AgentSettings(
                    use_vision=kwargs.get("use_vision", True),
                    max_failures=kwargs.get("max_failures", 3),
                    max_actions_per_step=kwargs.get("max_actions_per_step", 4),
                ),
            )

            logger.info(f"Autonomous browser-use agent created for task: {prompt}")

            # Return the browser-use agent directly
            return browser_use_agent

        except Exception as e:
            logger.error(f"Failed to create agent for prompt '{prompt}': {e}")
            raise
