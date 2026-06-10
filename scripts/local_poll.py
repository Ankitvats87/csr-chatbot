import os
import time
import random
import certifi
import httpx
from dotenv import load_dotenv

# Load .env file
load_dotenv()

TOKEN = os.getenv("TELEGRAM_BOT_TOKEN")
LOCAL_URL = os.getenv("TELEGRAM_RELAY_URL", "http://localhost:8000/webhook/telegram")
SECRET = os.getenv("TELEGRAM_WEBHOOK_SECRET", "")

if not TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is not set")

if ":" not in TOKEN:
    raise RuntimeError("TELEGRAM_BOT_TOKEN is malformed. Expected format: <bot_id>:<secret>")

print("=" * 60)
print("             SPARK63 LOCAL TG-POLLING AGENT")
print("=" * 60)
print(f"Relay Target:   {LOCAL_URL}")
print(f"Auth Secret:    {SECRET if SECRET else '(none)'}")
print("=" * 60)

print("Checking and clearing active remote webhook...")
try:
    resp = httpx.post(
        f"https://api.telegram.org/bot{TOKEN}/deleteWebhook",
        verify=False,
        timeout=30.0,
    )
    data = resp.json()
    if resp.status_code == 200 and data.get("ok"):
        print("-> Webhook successfully cleared.")
    else:
        print(f"-> Warning (deleteWebhook not ok): {resp.text}")
except Exception as e:
    print(f"-> Warning (could not delete webhook): {e}")

print()
print("Local polling agent is active. Send messages to your bot on Telegram!")
print("Press Ctrl+C to terminate the agent.")
print()

headers = {}
if SECRET:
    headers["X-Telegram-Bot-Api-Secret-Token"] = SECRET

client = httpx.Client(
    timeout=35.0,
    verify=False,
)

offset = 0
backoff = 1
max_backoff = 15

try:
    while True:
        try:
            response = client.get(
                f"https://api.telegram.org/bot{TOKEN}/getUpdates",
                params={
                    "timeout": 25,
                    "offset": offset,
                    "allowed_updates": ["message", "callback_query"],
                },
            )
            response.raise_for_status()
            payload = response.json()

            if not payload.get("ok"):
                print(f"[Polling Error] Telegram API returned not ok: {payload}")
                time.sleep(2)
                continue

            updates = payload.get("result", [])
            print(f"[Polling] fetched {len(updates)} update(s)")

            if updates:
                for update in updates:
                    print(f"[Update] id={update.get('update_id')}")
                    print(update)

            backoff = 1

            for update in updates:
                offset = update["update_id"] + 1
                try:
                    relay_resp = client.post(
                        LOCAL_URL,
                        json=update,
                        headers=headers,
                        timeout=30.0,
                    )
                    print(
                        f"[Relayed] update_id={update['update_id']} "
                        f"status={relay_resp.status_code} body={relay_resp.text}"
                    )
                except httpx.HTTPError as e:
                    print(f"[Relay Error] Failed to relay update {update['update_id']}: {e}")

        except httpx.HTTPStatusError as e:
            print(f"[Polling Error] HTTP status error: {e}")
        except httpx.RequestError as e:
            print(f"[Polling Error] Network connection failed: {e}")

        sleep_for = min(backoff, max_backoff) + random.uniform(0, 0.5)
        time.sleep(sleep_for)
        backoff = min(backoff * 2, max_backoff)

except KeyboardInterrupt:
    print("\nPolling agent stopped by user.")

finally:
    client.close()