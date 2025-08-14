import time
from datetime import datetime
from typing import Dict, List, Optional

from langchain_core.callbacks import BaseCallbackHandler

from cogents.common.logging import color_text, get_logger

logger = get_logger(__name__)


class NodeLoggingCallback(BaseCallbackHandler):
    def __init__(self, node_id: Optional[str] = None):
        self.node_id = node_id

    def _prefix(self) -> str:
        return f"[{self.node_id}] " if self.node_id else ""

    def on_tool_end(self, output, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[TOOL END] output={output}", "cyan", ["dim"]))

    def on_chain_end(self, output, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[CHAIN END] output={output}", "blue", ["dim"]))

    def on_llm_end(self, response, run_id, parent_run_id, **kwargs):
        logger.info(color_text(f"{self._prefix()}[LLM END] response={response}", "magenta", ["dim"]))

    def on_custom_event(self, event_name, payload, **kwargs):
        logger.info(color_text(f"{self._prefix()}[EVENT] {event_name}: {payload}", "yellow", ["dim"]))


class TokenUsageCallback(BaseCallbackHandler):
    def __init__(self, model_name: Optional[str] = None, verbose: bool = True):
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.model_name = model_name
        self.verbose = verbose
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history: List[Dict] = []

    def on_llm_start(self, serialized: Dict, prompts: List[str], **kwargs):
        """Track when LLM calls start"""
        self.llm_calls += 1
        if self.verbose:
            logger.info(color_text(f"[TOKEN CALLBACK] LLM call #{self.llm_calls} started", "cyan", ["dim"]))

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

        logger.info(color_text(f"[TOKEN USAGE]{model_info}{run_info}", "magenta", ["dim"]))
        logger.info(color_text(f"  Prompt: {prompt_tokens:,} tokens", None, ["dim"]))
        logger.info(color_text(f"  Completion: {completion_tokens:,} tokens", None, ["dim"]))
        logger.info(color_text(f"  Total: {total_tokens:,} tokens", None, ["dim"]))

        # Show session totals
        session_total = self.total_tokens()
        logger.info(color_text(f"  Session Total: {session_total:,} tokens", None, ["dim"]))

        # Cost estimation removed

    # Cost estimation methods removed

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
            "token_usage_history": self.token_usage_history,
        }

    def print_session_summary(self):
        """Print a formatted session summary"""
        summary = self.get_session_summary()

        logger.info(color_text("\n" + "=" * 50, "blue", ["dim"]))
        logger.info(color_text("TOKEN USAGE SESSION SUMMARY", "blue", ["dim"]))
        logger.info(color_text("=" * 50, "blue", ["dim"]))
        logger.info(color_text(f"Session Duration: {summary['session_duration_seconds']:.2f} seconds", None, ["dim"]))
        logger.info(color_text(f"LLM Calls: {summary['llm_calls']}", None, ["dim"]))
        logger.info(color_text(f"Total Prompt Tokens: {summary['total_prompt_tokens']:,}", None, ["dim"]))
        logger.info(color_text(f"Total Completion Tokens: {summary['total_completion_tokens']:,}", None, ["dim"]))
        logger.info(color_text(f"Total Tokens: {summary['total_tokens']:,}", None, ["dim"]))

        if self.model_name:
            logger.info(color_text(f"Model: {self.model_name}", None, ["dim"]))

        logger.info(color_text("=" * 50, "blue", ["dim"]))

    def reset_session(self):
        """Reset all counters for a new session"""
        self.total_prompt_tokens = 0
        self.total_completion_tokens = 0
        self.session_start = time.time()
        self.llm_calls = 0
        self.token_usage_history = []
        if self.verbose:
            logger.info(color_text("[TOKEN CALLBACK] Session reset", "cyan", ["dim"]))

    # Pricing configuration removed
