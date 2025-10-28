#!/usr/bin/env python3
"""
Super-advanced Telegram + Gemini bot (main script).
Includes:
- Text chat
- /image (image generation)
- /code (coding assistant)
- Subscriptions placeholders
- SQLite persistence
- Safe async handling (asyncio.to_thread)
"""

import os
import io
import time
import base64
import asyncio
import logging
import tempfile
from typing import Dict, List, Optional, Tuple

import aiosqlite
import google.generativeai as genai
from telegram import Update, InputFile, LabeledPrice
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    ContextTypes,
    filters,
    PreCheckoutQueryHandler,
)
from telegram.constants import ChatAction

import aiohttp

# Configuration (set these as environment variables; do NOT hardcode)
TELEGRAM_TOKEN = os.getenv("TELEGRAM_TOKEN")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
TELEGRAM_PROVIDER_TOKEN = os.getenv("TELEGRAM_PROVIDER_TOKEN", "")
STRIPE_API_KEY = os.getenv("STRIPE_API_KEY", "")
STRIPE_SUCCESS_URL = os.getenv("STRIPE_SUCCESS_URL", "")
STRIPE_CANCEL_URL = os.getenv("STRIPE_CANCEL_URL", "")

RATE_LIMIT_FREE = int(os.getenv("RATE_LIMIT_FREE", "20"))
RATE_LIMIT_PREMIUM = int(os.getenv("RATE_LIMIT_PREMIUM", "120"))
GLOBAL_CONCURRENCY_FREE = int(os.getenv("GLOBAL_CONCURRENCY_FREE", "4"))
GLOBAL_CONCURRENCY_PREMIUM = int(os.getenv("GLOBAL_CONCURRENCY_PREMIUM", "12"))

DB_PATH = os.getenv("BOT_DB_PATH", "bot_history.sqlite3")
ADMIN_IDS = set(int(x) for x in os.getenv("ADMIN_IDS", "").split(",") if x.strip().isdigit())

MAX_TEXT_PROMPT = int(os.getenv("MAX_TEXT_PROMPT", "2000"))
MAX_IMAGE_PROMPT = int(os.getenv("MAX_IMAGE_PROMPT", "1000"))

MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "gemini-pro")
IMAGE_MODEL_NAME = os.getenv("GEMINI_IMAGE_MODEL", "image-alpha-001")

if not TELEGRAM_TOKEN or not GEMINI_API_KEY:
    raise SystemExit("Set TELEGRAM_TOKEN and GEMINI_API_KEY in environment variables.")

logging.basicConfig(format="%(asctime)s - %(name)s - %(levelname)s - %(message)s", level=logging.INFO)
logger = logging.getLogger("super-advanced-gemini-bot")

genai.configure(api_key=GEMINI_API_KEY)

# Globals
user_chats: Dict[int, object] = {}
rate_buckets: Dict[int, Dict[str, float]] = {}
global_semaphore_free = asyncio.Semaphore(GLOBAL_CONCURRENCY_FREE)
global_semaphore_premium = asyncio.Semaphore(GLOBAL_CONCURRENCY_PREMIUM)
DB_INIT_LOCK = asyncio.Lock()

# DB helpers (create tables, save/get messages)
async def init_db():
    async with DB_INIT_LOCK:
        async with aiosqlite.connect(DB_PATH) as db:
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS messages (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    role TEXT NOT NULL,
                    text TEXT NOT NULL,
                    created_at REAL NOT NULL
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    price_cents INTEGER NOT NULL,
                    currency TEXT NOT NULL,
                    duration_days INTEGER NOT NULL,
                    description TEXT
                )
                """
            )
            await db.execute(
                """
                CREATE TABLE IF NOT EXISTS subscriptions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id INTEGER NOT NULL,
                    plan_id INTEGER NOT NULL,
                    status TEXT NOT NULL,
                    started_at REAL NOT NULL,
                    expires_at REAL,
                    provider TEXT,
                    provider_ref TEXT
                )
                """
            )
            await db.commit()

async def save_message(user_id: int, role: str, text: str):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute(
            "INSERT INTO messages (user_id, role, text, created_at) VALUES (?, ?, ?, ?)",
            (user_id, role, text, time.time()),
        )
        await db.commit()

async def get_conversation(user_id: int, limit: Optional[int] = 200):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute(
            "SELECT role, text, created_at FROM messages WHERE user_id = ? ORDER BY id ASC LIMIT ?",
            (user_id, limit or -1),
        )
        rows = await cur.fetchall()
        return [(r[0], r[1], r[2]) for r in rows]

async def clear_history_db(user_id: int):
    async with aiosqlite.connect(DB_PATH) as db:
        await db.execute("DELETE FROM messages WHERE user_id = ?", (user_id,))
        await db.commit()

# Rate limiter
def _refill_tokens(bucket: Dict[str, float], capacity: float, refill_interval: float = 60.0):
    now = time.time()
    elapsed = now - bucket.get("last", now)
    tokens = bucket.get("tokens", capacity)
    tokens = min(capacity, tokens + (elapsed / refill_interval) * capacity)
    bucket["tokens"] = tokens
    bucket["last"] = now

def allow_request(user_id: int, is_premium: bool) -> bool:
    capacity = float(RATE_LIMIT_PREMIUM if is_premium else RATE_LIMIT_FREE)
    bucket = rate_buckets.setdefault(user_id, {"tokens": capacity, "last": time.time()})
    _refill_tokens(bucket, capacity)
    if bucket["tokens"] >= 1.0:
        bucket["tokens"] -= 1.0
        return True
    return False

# Gemini helpers
def start_new_gemini_chat():
    return genai.GenerativeModel(MODEL_NAME).start_chat(history=[])

async def ensure_user_chat(user_id: int):
    if user_id not in user_chats:
        chat = await asyncio.to_thread(start_new_gemini_chat)
        user_chats[user_id] = chat
    return user_chats[user_id]

async def send_to_gemini_with_retries(chat, user_message: str, max_retries: int = 3, base_delay: float = 1.0):
    last_exception = None
    for attempt in range(1, max_retries + 1):
        try:
            resp = await asyncio.to_thread(chat.send_message, user_message)
            return resp
        except Exception as e:
            last_exception = e
            wait = base_delay * (2 ** (attempt - 1))
            logger.warning("Gemini text call failed (attempt %d/%d): %s — retrying in %.1fs", attempt, max_retries, e, wait)
            await asyncio.sleep(wait)
    raise last_exception

# Image helper (tries common genai shapes)
async def generate_image_from_prompt(prompt: str, size: str = "1024x1024"):
    if not prompt or len(prompt) > MAX_IMAGE_PROMPT:
        raise ValueError("Prompt empty or too long.")

    def _call_image_api(p: str):
        try:
            return genai.images.generate(model=IMAGE_MODEL_NAME, prompt=p, size=size)
        except Exception:
            try:
                return genai.Image.create(prompt=p, model=IMAGE_MODEL_NAME, size=size)
            except Exception as e:
                raise e

    resp = await asyncio.to_thread(_call_image_api, prompt)

    b64data = None
    image_url = None
    try:
        if hasattr(resp, "data"):
            d0 = resp.data[0]
            if isinstance(d0, dict):
                b64data = d0.get("b64_json") or d0.get("b64")
                image_url = d0.get("url") or d0.get("image_url")
            else:
                b64data = getattr(d0, "b64_json", None) or getattr(d0, "b64", None)
                image_url = getattr(d0, "url", None) or getattr(d0, "image_url", None)
        else:
            if isinstance(resp, list) and resp:
                item = resp[0]
                if isinstance(item, dict):
                    b64data = item.get("b64_json") or item.get("b64")
                    image_url = item.get("url")
            elif isinstance(resp, dict):
                b64data = resp.get("b64_json") or resp.get("b64")
                image_url = resp.get("url")
    except Exception:
        pass

    if b64data:
        return base64.b64decode(b64data)

    if image_url:
        async with aiohttp.ClientSession() as session:
            async with session.get(image_url) as r:
                if r.status == 200:
                    return await r.read()
                else:
                    raise ValueError(f"Failed to download image from URL, status={r.status}")

    if hasattr(resp, "binary") and isinstance(resp.binary, (bytes, bytearray)):
        return bytes(resp.binary)

    raise ValueError("Image generation returned unknown response format. Check genai library version.")

def split_text_chunks(text: str, limit: int = 4096):
    return [text[i:i+limit] for i in range(0, len(text), limit)]

BANNED_WORDS = {"illegal", "bomb", "attack"}

def is_prompt_allowed(prompt: str):
    p = prompt.lower()
    for w in BANNED_WORDS:
        if w in p:
            return False, f"Prompt contains banned word: {w}"
    return True, None

# Command handlers (start/image/code + helpers)
async def cmd_start(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await ensure_user_chat(user_id)
    await save_message(user_id, "system", "Started with /start")
    await update.message.reply_text(
        "🤖 Advanced Gemini Bot — ab image generation aur coding features bhi hain.\n\n"
        "Commands:\n"
        "/image <prompt> - generate image\n"
        "/code <language?> <prompt> - generate code\n"
        "/new /clear /export /plans /subscribe /myplan\n"
    )

async def cmd_image(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /image <prompt>")
        return
    prompt = " ".join(context.args).strip()
    if len(prompt) > MAX_IMAGE_PROMPT:
        await update.message.reply_text("Prompt bahut lamba hai. Shorter prompt bhejo.")
        return
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        await update.message.reply_text(f"Prompt disallowed: {reason}")
        return

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.UPLOAD_PHOTO)
    except Exception:
        pass

    sub = await get_active_subscription_simple(update.effective_user.id)
    is_premium = bool(sub)
    sem = global_semaphore_premium if is_premium else global_semaphore_free

    async with sem:
        try:
            sent = await update.message.reply_text("Image bana raha hoon... thoda intazaar karo.")
            image_bytes = await generate_image_from_prompt(prompt, size="1024x1024")
            bio = io.BytesIO(image_bytes)
            bio.name = "image.png"
            bio.seek(0)
            await update.message.reply_photo(photo=bio, caption=f"Image result for: {prompt[:120]}")
            await sent.delete()
        except Exception as e:
            logger.exception("Image generation failed: %s", e)
            await update.message.reply_text("❌ Image generate karne me problem aayi. Thodi der baad try karo.")

async def cmd_code(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not context.args:
        await update.message.reply_text("Usage: /code <language?> <prompt> — e.g. /code python sort a list")
        return
    first = context.args[0].lower()
    common_langs = {"python", "js", "javascript", "java", "c", "cpp", "c++", "go", "bash", "sh", "shell", "ruby", "rust", "ts", "typescript"}
    if first in common_langs:
        lang = first
        prompt = " ".join(context.args[1:]).strip()
    else:
        lang = None
        prompt = " ".join(context.args).strip()

    if not prompt:
        await update.message.reply_text("Prompt missing. /code <language?> <prompt>")
        return
    if len(prompt) > MAX_TEXT_PROMPT:
        await update.message.reply_text("Prompt too long. Shorten it.")
        return
    allowed, reason = is_prompt_allowed(prompt)
    if not allowed:
        await update.message.reply_text(f"Prompt disallowed: {reason}")
        return

    system_instruction = (
        "You are an expert programming assistant. Provide concise, correct code only. "
        "Return code inside a single fenced code block with the language tag if possible. "
        "Also include a short explanation (2-3 lines) after the code. "
        "Do not include extra commentary."
    )
    if lang:
        system_instruction += f" Preferred language: {lang}."

    def start_code_chat():
        m = genai.GenerativeModel(MODEL_NAME)
        return m.start_chat(history=[])

    chat = await asyncio.to_thread(start_code_chat)
    combined = system_instruction + "\n\nUser prompt:\n" + prompt

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    sub = await get_active_subscription_simple(update.effective_user.id)
    is_premium = bool(sub)
    sem = global_semaphore_premium if is_premium else global_semaphore_free

    async with sem:
        try:
            resp = await send_to_gemini_with_retries(chat, combined, max_retries=3)
            reply_text = getattr(resp, "text", str(resp))
            chunks = split_text_chunks(reply_text, 4096)
            for c in chunks:
                await update.message.reply_text(c)
        except Exception as e:
            logger.exception("Code generation failed: %s", e)
            await update.message.reply_text("❌ Code generate karne me problem aayi. Thodi der baad try karo.")

# Minimal subscription helper (used to choose premium concurrency). For full subscription
# use the subscription file previously provided.
async def get_active_subscription_simple(user_id: int):
    # returns True/False if user has an active subscription — simple placeholder
    # look into subscriptions table (if used)
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT COUNT(*) FROM subscriptions WHERE user_id=? AND status='active'", (user_id,))
        r = await cur.fetchone()
        return r[0] > 0 if r else False

async def cmd_plans(update: Update, context: ContextTypes.DEFAULT_TYPE):
    async with aiosqlite.connect(DB_PATH) as db:
        cur = await db.execute("SELECT id, name, price_cents, currency, duration_days, description FROM plans ORDER BY id ASC")
        rows = await cur.fetchall()
        if not rows:
            await update.message.reply_text("Abhi koi plan configured nahi hai.")
            return
        lines = []
        for r in rows:
            pid, name, price_cents, currency, duration_days, desc = r
            price = f"{price_cents/100:.2f} {currency.upper()}"
            lines.append(f"ID:{pid} | {name} | {price} | {duration_days}d\n{desc}")
        await update.message.reply_text("\n\n".join(lines))

async def cmd_help(update: Update, context: ContextTypes.DEFAULT_TYPE):
    await update.message.reply_text(
        "Commands:\n"
        "/image <prompt> — generate an image\n"
        "/code <language?> <prompt> — generate code\n"
        "/plans /subscribe /myplan /export /new /clear"
    )

async def cmd_new(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    chat = await asyncio.to_thread(start_new_gemini_chat)
    user_chats[user_id] = chat
    await clear_history_db(user_id)
    await save_message(user_id, "system", "New conversation started with /new")
    await update.message.reply_text("✅ Nayi conversation shuru ho gayi!")

async def cmd_clear(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    await clear_history_db(user_id)
    user_chats.pop(user_id, None)
    await update.message.reply_text("🗑️ History clear ho gayi!")

async def cmd_export(update: Update, context: ContextTypes.DEFAULT_TYPE):
    user_id = update.effective_user.id
    conv = await get_conversation(user_id, limit=None)
    if not conv:
        await update.message.reply_text("Koi history nahi hai export karne ke liye.")
        return
    with tempfile.NamedTemporaryFile("w+", delete=False, encoding="utf-8", suffix=".txt") as f:
        for role, text, ts in conv:
            ts_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(ts))
            f.write(f"[{ts_str}] {role}:\n{text}\n\n")
        tmpname = f.name
    try:
        await update.message.reply_document(document=InputFile(tmpname), filename=f"conversation_{user_id}.txt")
    except Exception as e:
        logger.exception("Failed to send export file: %s", e)
        await update.message.reply_text("❌ Export bhejne me problem aayi.")
    finally:
        try:
            os.unlink(tmpname)
        except Exception:
            pass

# generic text handler (fallback to normal Gemini chat)
async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE):
    if not update.message or not update.message.text:
        return
    user_id = update.effective_user.id
    text = update.message.text.strip()
    lowered = text.lower()
    if lowered.startswith("image:") or lowered.startswith("img:"):
        prompt = text.split(":", 1)[1].strip()
        context.args = [prompt]
        await cmd_image(update, context)
        return

    sub = await get_active_subscription_simple(user_id)
    is_premium = bool(sub)
    if not allow_request(user_id, is_premium=is_premium):
        await update.message.reply_text("⚠️ Rate limit applied — thodi der baad try karo.")
        return

    try:
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action=ChatAction.TYPING)
    except Exception:
        pass

    await init_db()
    chat = await ensure_user_chat(user_id)
    await save_message(user_id, "user", text)

    sem = global_semaphore_premium if is_premium else global_semaphore_free
    async with sem:
        try:
            resp = await send_to_gemini_with_retries(chat, text)
            reply_text = getattr(resp, "text", str(resp))
            await save_message(user_id, "assistant", reply_text)
            for chunk in split_text_chunks(reply_text):
                await update.message.reply_text(chunk)
        except Exception as e:
            logger.exception("Gemini chat failed: %s", e)
            await update.message.reply_text("❌ Gemini se reply nahi mila — thodi der baad try karo.")

# Error handler and main
async def error_handler(update: object, context: ContextTypes.DEFAULT_TYPE):
    logger.error("Update %s caused error %s", update, context.error)
    try:
        if hasattr(update, "message") and update.message:
            await update.message.reply_text("❌ Technical issue. Thodi der baad try karo.")
    except Exception:
        pass

def main():
    loop = asyncio.get_event_loop()
    loop.run_until_complete(init_db())

    app = Application.builder().token(TELEGRAM_TOKEN).build()
    app.add_handler(CommandHandler("start", cmd_start))
    app.add_handler(CommandHandler("help", cmd_help))
    app.add_handler(CommandHandler("image", cmd_image))
    app.add_handler(CommandHandler("code", cmd_code))
    app.add_handler(CommandHandler("plans", cmd_plans))
    app.add_handler(CommandHandler("new", cmd_new))
    app.add_handler(CommandHandler("clear", cmd_clear))
    app.add_handler(CommandHandler("export", cmd_export))
    app.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))
    app.add_error_handler(error_handler)

    logger.info("🤖 Super-advanced Gemini bot starting...")
    app.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    main()
