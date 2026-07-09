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
        if self.provider == "gemini" and self.gemini_key:
            try:
                return self._call_gemini(system_prompt, user_prompt)
            except Exception as e:
                logger.warning(f"Gemini API call failed: {e}. Using heuristic fallback.")

        elif self.provider == "openai" and self.openai_key:
            try:
                return self._call_openai(system_prompt, user_prompt)
            except Exception as e:
                logger.warning(f"OpenAI API call failed: {e}. Using heuristic fallback.")

        # Heuristic fallback
        if fallback_generator and callable(fallback_generator):
            return fallback_generator()

        return {}

    def _call_gemini(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call Google Gemini model."""
        try:
            import google.generativeai as genai
            genai.configure(api_key=self.gemini_key)
            model_name = self.model or "gemini-1.5-flash"
            model = genai.GenerativeModel(
                model_name=model_name,
                system_instruction=system_prompt,
                generation_config={"response_mime_type": "application/json"},
            )
            response = model.generate_content(user_prompt)
            return json.loads(response.text)
        except Exception:
            # Try new google.genai SDK
            from google import genai as new_genai
            client = new_genai.Client(api_key=self.gemini_key)
            model_name = self.model or "gemini-2.5-flash"
            response = client.models.generate_content(
                model=model_name,
                contents=f"{system_prompt}\n\n{user_prompt}",
                config={"response_mime_type": "application/json"},
            )
            return json.loads(response.text)

    def _call_openai(self, system_prompt: str, user_prompt: str) -> dict[str, Any]:
        """Call OpenAI model."""
        from openai import OpenAI
        client = OpenAI(api_key=self.openai_key)
        model_name = self.model or "gpt-4o-mini"
        response = client.chat.completions.create(
            model=model_name,
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            temperature=0.2,
        )
        content = response.choices[0].message.content or "{}"
        return json.loads(content)
