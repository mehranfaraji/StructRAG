import os
import time
from typing import Optional

import requests


class GeminiAPI:
    """Minimal wrapper around the Gemini generative language REST API."""

    def __init__(self, model: str = "gemini-1.5-flash-latest", api_key: Optional[str] = None):
        self.model = model
        self.api_key = api_key or os.environ.get("GOOGLE_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Gemini API key not provided. Set the GOOGLE_API_KEY environment variable or "
                "pass --gemini_api_key when running main.py."
            )
        self.url = f"https://generativelanguage.googleapis.com/v1beta/models/{self.model}:generateContent"

    def response(self, input_text: str, max_new_tokens: int = 1024) -> str:
        start_time = time.time()
        payload = {
            "contents": [
                {
                    "parts": [
                        {
                            "text": input_text
                        }
                    ]
                }
            ],
            "generationConfig": {
                "maxOutputTokens": max_new_tokens,
            },
        }
        headers = {"Content-Type": "application/json"}

        try:
            resp = requests.post(
                self.url,
                params={"key": self.api_key},
                json=payload,
                headers=headers,
                timeout=120,
            )
            resp.raise_for_status()
        except requests.RequestException as exc:
            raise RuntimeError(f"Gemini request failed: {exc}") from exc

        data = resp.json()
        candidates = data.get("candidates") or []
        if not candidates:
            raise RuntimeError(f"Gemini API returned no candidates: {data}")

        content = candidates[0].get("content") or {}
        parts = content.get("parts") or []
        if not parts:
            raise RuntimeError(f"Gemini API returned empty content: {data}")

        text = parts[0].get("text", "")
        elapsed = (time.time() - start_time) / 60
        print(f"GeminiAPI call took {elapsed:.2f} minutes")
        return text
