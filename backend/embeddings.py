import os
import httpx
from typing import Union
from dotenv import load_dotenv

load_dotenv()

MISTRAL_EMBED_URL = "https://api.mistral.ai/v1/embeddings"
EMBED_MODEL = "mistral-embed"
EMBED_DIMENSIONS = 1024


def _get_api_key() -> str:
    key = os.getenv("MISTRAL_API_KEY")
    if not key:
        raise ValueError("MISTRAL_API_KEY not set in environment")
    return key


async def embed_text(text: Union[str, list[str]]) -> list[list[float]]:
    """
    Generate embeddings using Mistral Embed.

    Args:
        text: Single string or list of strings to embed

    Returns:
        List of embedding vectors (1024 dimensions each)
    """
    if isinstance(text, str):
        text = [text]

    async with httpx.AsyncClient() as client:
        response = await client.post(
            MISTRAL_EMBED_URL,
            headers={
                "Authorization": f"Bearer {_get_api_key()}",
                "Content-Type": "application/json"
            },
            json={
                "model": EMBED_MODEL,
                "input": text
            },
            timeout=30.0
        )
        response.raise_for_status()
        data = response.json()

    return [item["embedding"] for item in data["data"]]


async def embed_single(text: str) -> list[float]:
    """Embed a single text and return the vector."""
    embeddings = await embed_text(text)
    return embeddings[0]


def embed_text_sync(text: Union[str, list[str]]) -> list[list[float]]:
    """Synchronous version for batch processing."""
    import requests

    if isinstance(text, str):
        text = [text]

    response = requests.post(
        MISTRAL_EMBED_URL,
        headers={
            "Authorization": f"Bearer {_get_api_key()}",
            "Content-Type": "application/json"
        },
        json={
            "model": EMBED_MODEL,
            "input": text
        },
        timeout=30.0
    )
    response.raise_for_status()
    data = response.json()

    return [item["embedding"] for item in data["data"]]
