# AI Book Rewriter

> Intelligent tool for rewriting large texts (books, articles) via a local LLM proxy.
> Supports GUI and web interfaces, multilingual UI, multiple prompt presets.

---

## Table of Contents

- [Features](#features)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Configuration](#configuration)
- [Project Structure](#project-structure)
- [Troubleshooting](#troubleshooting)
- [Changelog](#changelog)

---

## Features

- **Smart block splitting** — semantic boundaries (paragraphs, sentences)
- **Context-aware rewriting** — each block sees neighbouring blocks for smooth transitions
- **Adaptive generation** — temperature auto-adjusted on quality failures
- **Quality validation** — similarity, length ratio, lexical diversity checks
- **Resume support** — saves progress, continues from last block
- **GUI** — dark-theme tkinter desktop app
- **Web UI** — Flask + SSE real-time progress, works in any browser
- **Multilingual UI** — Russian / English / Chinese (中文)
- **5 prompt presets** — Literary, Academic, Simplified, Creative, Translation
- **Local proxy integration** — OpenAI-compatible API (e.g. LLM-API-Key-Proxy)

---

## Requirements

- Python 3.10+
- Local LLM proxy running at `http://127.0.0.1:8000/v1` (OpenAI-compatible)

---

## Installation

```bash
git clone https://github.com/ShmidtS/AI-rewriter.git
cd AI-rewriter
python -m venv .venv

# Windows:
.venv\Scripts\activate
# Linux/Mac:
source .venv/bin/activate

pip install -r requirements.txt
cp .env.example .env
# Edit .env with your proxy settings
```

---

## Usage

### Desktop GUI (default)

```bash
python main.py
# or explicitly:
python main.py --gui
# or via module:
python -m app.gui
```

### Web Interface

```bash
python main.py --web
# Open http://127.0.0.1:5000 in your browser
```

Custom host/port:

```bash
python main.py --web --host 0.0.0.0 --port 8080
```

Via module:

```bash
python -m app.web --host 0.0.0.0 --port 8080
```

### CLI Interface (placeholder)

```bash
python main.py --cli
# or:
python -m app.cli
```

> **Note:** CLI is a placeholder for future implementation.

### Windows quick-start

```bash
start.bat
```

---

## Configuration

Create `.env` from `.env.example`:

```env
# Proxy URL (OpenAI-compatible)
OPENAI_BASE_URL=http://127.0.0.1:8000/v1

# Auth token for the proxy
AUTH_TOKEN=sk-admin74203

# Default model
MODEL=colin/gpt-5.4

# Context window size
CONTEXT_WINDOW=1000000

# Max output tokens per block
MAX_OUTPUT_TOKENS=32768

# Block target size (characters)
BLOCK_TARGET_CHARS=15000

# Rewrite length ratio limits
MIN_REWRITE_LENGTH_RATIO=0.40
MAX_REWRITE_LENGTH_RATIO=1.60

# UI language: ru / en / zh
UI_LANG=en
```

### Connection Profiles

The application supports three connection profiles:

| Profile | Description |
|---------|-------------|
| `direct` | Direct connection to LLM API |
| `proxy` | Connection via local proxy (recommended) |
| `auto` | Automatic selection based on availability |

---

## Project Structure

```
AI-rewriter/
├── main.py              # Entry point (GUI or web)
├── requirements.txt
├── .env                 # Your local config (not committed)
├── .env.example
│
├── core/                # Rewriting engine
│   ├── config.py        # All constants from .env
│   ├── context.py       # GlobalContext (characters, plot, themes)
│   ├── text_engine.py   # Block splitting, validation, metrics
│   ├── api_client.py    # HTTP client to proxy (SSE streaming)
│   ├── prompts.py       # 5 system prompt presets + user prompt builder
│   ├── state_manager.py # Atomic state save/load
│   └── rewriter.py      # Main orchestration loop
│
├── gui/
│   └── app.py           # Tkinter desktop app (dark theme, i18n)
│
├── web/
│   ├── app.py           # Flask web app + SSE
│   ├── templates/
│   │   └── index.html   # Single-page UI
│   └── static/
│       ├── style.css    # Dark theme CSS
│       └── app.js       # SSE client, form handling
│
├── i18n/
│   ├── ru.json          # Russian
│   ├── en.json          # English
│   └── zh.json          # Chinese
│
└── prompts/
    └── catalog.json     # Prompt presets metadata
```

---

## Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: No module named 'core'` | Run from project root, not from subdirectories |
| Connection refused to proxy | Ensure proxy is running at `http://127.0.0.1:8000` |
| Quality validation fails frequently | Adjust `MIN_REWRITE_LENGTH_RATIO` and `MAX_REWRITE_LENGTH_RATIO` in `.env` |
| Out of memory errors | Reduce `BLOCK_TARGET_CHARS` or `CONTEXT_WINDOW` |
| GUI doesn't start | Check Python version (3.10+ required), reinstall `sv-ttk` |

### Getting Help

- Check [API.md](API.md) for Web API documentation
- Check [PROMPTS.md](PROMPTS.md) for prompt preset guide
- Open an issue on [GitHub](https://github.com/ShmidtS/AI-rewriter/issues)

---

## Changelog

### v2.0.0 (2026-03-30)

- Full modular refactoring: `core/`, `gui/`, `web/`, `i18n/`
- Added Flask web interface with SSE real-time progress
- Added multilingual UI: Russian, English, Chinese
- Added 5 prompt presets (literary, academic, simplified, creative, translation)
- Unified entry point `main.py` (`--web` flag for web mode)
- `core/api_client.py`: requests + SSE streaming, 502 retry logic
- `core/rewriter.py`: decoupled from GUI, works headless
- `core/prompts.py`: preset system with i18n labels
- `core/context.py`: GlobalContext with `from_json` classmethod
- Web UI: dark theme, SSE events, file upload/download, model refresh
- i18n system with fallback to English

### v1.1.0

- Migrated from Google Gemini to local OpenAI-compatible proxy
- SSE streaming support
- Adaptive temperature

### v1.0.0 (2025-01-30)

- Initial release
- Tkinter GUI, Gemini API, quality validation, resume support

---

## Author

ShmidtS — [GitHub](https://github.com/ShmidtS)

---

## License

See [LICENSE](../../LICENSE) file for details.
