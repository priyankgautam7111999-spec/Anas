```markdown
# Gemini Telegram Bot

This repo contains a super-advanced Telegram bot that uses Google Gemini for:
- Text chat
- Code generation (/code)
- Image generation (/image)
- Subscriptions & payments (optional)
- Persistent history (SQLite)
- Deployment via Heroku or Cloud Run

Setup:
1. Copy `.env.example` and fill the environment variables.
2. Create a Python virtualenv and install dependencies:
   pip install -r requirements.txt
3. Run locally:
   python bot_super_advanced.py

Deployment:
- Heroku: Use the provided `Procfile` and GitHub Actions workflow (deploy-heroku.yml).
- Cloud Run: Use the `Dockerfile` and Cloud Run workflow (deploy-cloud-run.yml).

Secrets should be set as environment variables (NOT committed into repo).
```
