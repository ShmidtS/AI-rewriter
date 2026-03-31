# AI Book Rewriter

> Intelligent tool for rewriting large texts (books, articles) via a local LLM proxy.
> Supports GUI and web interfaces, multilingual UI, multiple prompt presets, parallel processing.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

---

## ✨ Features

- **Smart block splitting** — semantic boundaries (paragraphs, sentences)
- **Context-aware rewriting** — each block sees neighbouring blocks for smooth transitions
- **Adaptive generation** — temperature auto-adjusted on quality failures
- **Quality validation** — similarity, length ratio, lexical diversity checks
- **Resume support** — saves progress, continues from last block
- **GUI** — dark-theme tkinter desktop app
- **Web UI** — Flask + SSE real-time progress, works in any browser
- **Multilingual UI** — 8 languages: RU, EN, ZH, DE, FR, ES, JA, KO
- **5 prompt presets** — Literary, Academic, Simplified, Creative, Translation
- **Parallel processing** — optional parallel mode for faster rewriting of large documents (enable via Web UI "Parallel mode" checkbox)
- **Extended language support** — output text in any of 8 supported languages with strict language enforcement via prompt injection
- **Local proxy integration** — OpenAI-compatible API

---

## 🚀 Quick Start

```bash
# Clone and setup
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

# Run GUI (default)
python main.py

# Run Web UI
python main.py --web
# Open http://127.0.0.1:5000
```

---

## 📖 Documentation Structure

```
docs/
├── en/                     # English documentation
│   ├── README.md           # Full documentation
│   ├── API.md              # Web API reference
│   └── PROMPTS.md          # Prompt presets guide
│
├── ru/                     # Russian documentation (Русский)
│   ├── README.md           # Полная документация
│   ├── API.md              # Справочник Web API
│   └── PROMPTS.md          # Руководство по промптам
│
└── zh/                     # Chinese documentation (中文)
    ├── README.md           # 完整文档
    ├── API.md              # Web API 参考
    └── PROMPTS.md          # 提示词预设指南
```

---

## 🔧 Configuration

Create `.env` from `.env.example`:

```env
# Proxy URL (OpenAI-compatible)
OPENAI_BASE_URL=http://127.0.0.1:8000/v1

# Auth token for the proxy
AUTH_TOKEN=your-api-key

# Default model
MODEL=gemini/gemini-2.5-flash

# UI language: ru / en / zh
UI_LANG=en
```

### Supported UI & Output Languages

| Code | Language |
|------|----------|
| `ru` | Русский (Russian) |
| `en` | English |
| `zh` | 中文 (Chinese) |
| `de` | Deutsch (German) |
| `fr` | Français (French) |
| `es` | Español (Spanish) |
| `ja` | 日本語 (Japanese) |
| `ko` | 한국어 (Korean) |

#### Language Enforcement Improvements

All 5 prompt presets include a **strict language enforcement block** that is injected at runtime based on the selected output language. This ensures:

- The model outputs text **ONLY** in the specified language
- No accidental translation to English or any other language
- Consistent output even when the input is in a different language
- Every word in the output must match the target language

This is critical for the Translation preset, where users need reliable output language control regardless of the source text language.

### Parallel Processing

The rewrite engine supports a **parallel processing mode** for faster processing of large documents:

- Enable via the "Parallel mode" checkbox in the Web UI (`python main.py --web`)
- Multiple text blocks are processed concurrently by the LLM
- Significantly reduces total rewrite time for books and long documents
- Quality validation and context-awareness are preserved for each block independently
- Works via the `parallel_mode=true` form parameter on the `POST /api/start` endpoint

### Connection Profiles

The application supports three connection profiles:

| Profile | Description |
|---------|-------------|
| `direct` | Direct connection to LLM API |
| `proxy` | Connection via local proxy (recommended) |
| `auto` | Automatic selection based on availability |

---

## 📋 Requirements

- Python 3.10+
- Local LLM proxy running at `http://127.0.0.1:8000/v1` (OpenAI-compatible)

---

## 📜 License

See [LICENSE](LICENSE) file for details.

---

## 👤 Author

ShmidtS — [GitHub](https://github.com/ShmidtS)
