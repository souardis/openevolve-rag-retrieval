"""
Sanity-check both API keys before OpenEvolve ever touches them: one trivial
call to each provider, through the same OpenAI-compatible interface OpenEvolve
itself will use. Run with: uv run python scripts/check_keys.py
"""

import os

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()


def check_anthropic() -> None:
    client = OpenAI(
        base_url="https://api.anthropic.com/v1/",
        api_key=os.environ["ANTHROPIC_API_KEY"],
    )
    response = client.chat.completions.create(
        model="claude-haiku-5-5",
        messages=[{"role": "user", "content": "reply with one word: ok"}],
        max_tokens=50,
    )
    choice = response.choices[0]
    print(f"content={choice.message.content!r} finish_reason={choice.finish_reason!r}")


def check_deepseek() -> None:
    client = OpenAI(
        base_url="https://api.deepseek.com",
        api_key=os.environ["DEEPSEEK_API_KEY"],
    )
    response = client.chat.completions.create(
        model="deepseek-v4-pro",
        messages=[{"role": "user", "content": "reply with one word: ok"}],
        max_tokens=50,
    )
    choice = response.choices[0]
    print(f"content={choice.message.content!r} finish_reason={choice.finish_reason!r}")


if __name__ == "__main__":
    print("checking Anthropic...")
    check_anthropic()
    print("checking DeepSeek...")
    check_deepseek()
