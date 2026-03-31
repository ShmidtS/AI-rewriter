# AI Book Rewriter

> Intelligent tool for rewriting large texts (books, articles) via a local LLM proxy.
> Supports GUI and web interfaces, multilingual UI, multiple prompt presets, parallel processing.

[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-green.svg)](https://www.python.org/)

---

## 📚 Documentation

| Language | Documentation |
|----------|---------------|
| **English** | [README.md](docs/en/README.md) · [API Reference](docs/en/API.md) · [Prompt Guide](docs/en/PROMPTS.md) |
| **Русский** | [README.md](docs/ru/README.md) · [API Справочник](docs/ru/API.md) · [Руководство по промптам](docs/ru/PROMPTS.md) |
| **中文** | [README.md](docs/zh/README.md) · [API 参考](docs/zh/API.md) · [提示词指南](docs/zh/PROMPTS.md) |

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

**Language enforcement:** Every prompt preset injects a strict language instruction into the system prompt. The LLM is explicitly told to output ONLY in the specified language and never deviate. This ensures consistent output regardless of input language.

### Parallel Processing

For large documents, enable **parallel mode** via the "Parallel mode" toggle in the Web UI (`python main.py --web`). This processes multiple text blocks concurrently, significantly reducing total rewrite time for books and long documents.

See [docs/en/README.md](docs/en/README.md) for full configuration options.

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
