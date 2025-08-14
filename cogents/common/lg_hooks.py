import time
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from cogents.common.logging import get_logger

logger = get_logger(__name__)


class NodeLoggingCallback(BaseCallbackHandler):
    def on_tool_end(self, output, run_id, parent_run_id, **kwargs):
        logger.info(f"[TOOL END] output={output}")

    def on_chain_end(self, outputs, run_id, parent_run_id, **kwargs):
        logger.info(f"[CHAIN END] outputs={outputs}")

    def on_llm_end(self, response, run_id, parent_run_id, **kwargs):
        logger.info(f"[LLM END] response={response}")

    def on_custom_event(self, event_name, payload, **kwargs):
        logger.info(f"[EVENT] {event_name}: {payload}")


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self, model_name: Optional[str] = None, verbose: bool = True):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.model_name = model_name
        self.verbose = verbose
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history: List[Dict] = []

        # Common model pricing (per 1K tokens) - can be overridden
        self.model_pricing = {
            "gpt-4": {"input": 0.03, "output": 0.06},
            "gpt-4-turbo": {"input": 0.01, "output": 0.03},
            "gpt-3.5-turbo": {"input": 0.0015, "output": 0.002},
            "claude-3-opus": {"input": 0.015, "output": 0.075},
            "claude-3-sonnet": {"input": 0.003, "output": 0.015},
            "claude-3-haiku": {"input": 0.00025, "output": 0.00125},
        }

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Track when LLM calls start"""
        self.llm_calls += 1
        if self.verbose:
            logger.info(f"[TOKEN CALLBACK] LLM call #{self.llm_calls} started")

    def on_llm_end(self, response, run_id: Optional[str] = None, **kwargs):
        """Enhanced token usage tracking with multiple extraction methods"""
        usage_data = self._extract_token_usage(response)

        if usage_data:
            prompt_tokens = usage_data.get("prompt_tokens", 0)
            completion_tokens = usage_data.get("completion_tokens", 0)
            total_tokens = usage_data.get("total_tokens", 0)

            # Update totals
            self.total_prompt_tokens += prompt_tokens
            self.total_completion_tokens += completion_tokens

            # Store in history
            call_data = {
                "run_id": run_id,
                "timestamp": datetime.now().isoformat(),
                "prompt_tokens": prompt_tokens,
                "completion_tokens": completion_tokens,
                "total_tokens": total_tokens,
                "model_name": self.model_name or "unknown",
            }
            self.token_usage_history.append(call_data)

            if self.verbose:
                self._log_token_usage(prompt_tokens, completion_tokens, total_tokens, run_id)

    def _extract_token_usage(self, response) -> Optional[Dict]:
        """Extract token usage from various response formats"""
        usage = None

        # Method 1: Standard OpenAI format
        if hasattr(response, "llm_output") and response.llm_output:
            usage = response.llm_output.get("token_usage") or response.llm_output.get("usage")

        # Method 2: Alternative metadata format
        elif hasattr(response, "response_metadata") and response.response_metadata:
            usage = response.response_metadata.get("token_usage") or response.response_metadata.get("usage")

        # Method 3: Direct response attributes
        elif hasattr(response, "token_usage"):
            usage = response.token_usage

        # Method 4: Check if response itself is a dict with usage
        elif isinstance(response, dict):
            usage = response.get("token_usage") or response.get("usage")

        return usage

    def _log_token_usage(
        self, prompt_tokens: int, completion_tokens: int, total_tokens: int, run_id: Optional[str] = None
    ):
        """Log token usage with detailed information"""
        run_info = f" (run_id: {run_id})" if run_id else ""
        model_info = f" [{self.model_name}]" if self.model_name else ""

        logger.info(f"[TOKEN USAGE]{model_info}{run_info}")
        logger.info(f"  Prompt: {prompt_tokens:,} tokens")
        logger.info(f"  Completion: {completion_tokens:,} tokens")
        logger.info(f"  Total: {total_tokens:,} tokens")

        # Show session totals
        session_total = self.total_tokens()
        logger.info(f"  Session Total: {session_total:,} tokens")

        # Show cost estimation if model is known
        if self.model_name and self.model_name in self.model_pricing:
            cost = self._estimate_cost(prompt_tokens, completion_tokens)
            logger.info(f"  Estimated Cost: ${cost:.4f}")

    def _estimate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Estimate cost based on model pricing"""
        if not self.model_name or self.model_name not in self.model_pricing:
            return 0.0

        pricing = self.model_pricing[self.model_name]
        prompt_cost = (prompt_tokens / 1000) * pricing["input"]
        completion_cost = (completion_tokens / 1000) * pricing["output"]

        return prompt_cost + completion_cost

    def total_tokens(self) -> int:
        """Get total tokens used in this session"""
        return self.total_prompt_tokens + self.total_completion_tokens

    def get_session_summary(self) -> Dict:
        """Get comprehensive session summary"""
        session_duration = time.time() - self.session_start

        return {
            "session_duration_seconds": session_duration,
            "llm_calls": self.llm_calls,
            "total_prompt_tokens": self.total_prompt_tokens,
            "total_completion_tokens": self.total_completion_tokens,
            "total_tokens": self.total_tokens(),
            "model_name": self.model_name,
            "estimated_cost": self._estimate_cost(self.total_prompt_tokens, self.total_completion_tokens),
            "token_usage_history": self.token_usage_history,
        }

    def print_session_summary(self):
        """Print a formatted session summary"""
        summary = self.get_session_summary()

        logger.info("\n" + "=" * 50)
        logger.info("TOKEN USAGE SESSION SUMMARY")
        logger.info("=" * 50)
        logger.info(f"Session Duration: {summary['session_duration_seconds']:.2f} seconds")
        logger.info(f"LLM Calls: {summary['llm_calls']}")
        logger.info(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}")
        logger.info(f"Total Completion Tokens: {summary['total_completion_tokens']:,}")
        logger.info(f"Total Tokens: {summary['total_tokens']:,}")

        if self.model_name:
            logger.info(f"Model: {self.model_name}")
            logger.info(f"Estimated Cost: ${summary['estimated_cost']:.4f}")

        logger.info("=" * 50)

    def reset_session(self):
        """Reset all counters for a new session"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history = []
        if self.verbose:
            logger.info("[TOKEN CALLBACK] Session reset")

    def set_model_pricing(self, model_name: str, input_price: float, output_price: float):
        """Set custom pricing for a model"""
        self.model_pricing[model_name] = {"input": input_price, "output": output_price}
        if self.verbose:
            logger.info(
                f"[TOKEN CALLBACK] Set pricing for {model_name}: ${input_price}/1K input, ${output_price}/1K output"
            )
