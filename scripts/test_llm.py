import os
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

print("Testing OpenAI API client...")
openai_key = os.getenv("OPENAI_API_KEY")
if openai_key:
    try:
        client = OpenAI(api_key=openai_key)
        resp = client.chat.completions.create(
            model=os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"),
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("-> OpenAI OK:", resp.choices[0].message.content)
    except Exception as e:
        print("-> OpenAI Failed:", e)
else:
    print("-> OpenAI not configured (no key)")

print("\nTesting OpenRouter API client...")
or_key = os.getenv("OPENROUTER_API_KEY")
or_model = os.getenv("OPENROUTER_MODEL")
if or_key and or_model:
    try:
        client = OpenAI(
            api_key=or_key,
            base_url="https://openrouter.ai/api/v1",
            default_headers={
                "HTTP-Referer": "http://localhost:8000",
                "X-Title": "Spark63 CSR Bot Test"
            }
        )
        resp = client.chat.completions.create(
            model=or_model,
            messages=[{"role": "user", "content": "Hello"}],
            max_tokens=10
        )
        print("-> OpenRouter OK:", resp.choices[0].message.content)
    except Exception as e:
        print("-> OpenRouter Failed:", e)
else:
    print("-> OpenRouter not configured")
