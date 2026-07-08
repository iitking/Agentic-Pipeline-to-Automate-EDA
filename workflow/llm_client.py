"""LLM Client provider supporting Gemini, OpenAI, Ollama, and intelligent heuristic fallbacks."""

import os
import json
import logging
from typing import Any
from dotenv import load_dotenv

load_dotenv()
logger = logging.getLogger(__name__)


class LLMClient:
    """Provides unified access to LLMs with automatic heuristic fallback."""

    def __init__(self, provider: str | None = None, model: str | None = None):
        self.provider = provider or os.getenv("LLM_PROVIDER", "auto").lower()
        self.gemini_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        self.openai_key = os.getenv("OPENAI_API_KEY")
        self.ollama_url = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

        # Determine active provider
        if self.provider == "auto":
            if self.gemini_key:
                self.provider = "gemini"
            elif self.openai_key:
                self.provider = "openai"
            else:
                self.provider = "heuristic"

        self.model = model
        logger.info(f"LLM Client initialized with provider: {self.provider}")

    def generate_json(self, system_prompt: str, user_prompt: str, fallback_generator: Any = None) -> dict[str, Any]:
        """Generate structured JSON response using available LLM, falling back to heuristic generator."""
