# AI 书籍改写工具

> 通过本地 LLM 代理改写大型文本（书籍、文章）的智能工具。
> 支持 GUI 和 Web 界面、多语言界面、多种提示词预设、并行处理。

---

## 目录

- [功能特点](#功能特点)
- [系统要求](#系统要求)
- [安装](#安装)
- [使用方法](#使用方法)
- [配置](#配置)
- [项目结构](#项目结构)
- [故障排除](#故障排除)
- [更新日志](#更新日志)

---

## 功能特点

- **智能分块** — 按语义边界（段落、句子）分割文本
- **上下文感知改写** — 每个块可看到相邻块，确保过渡自然
- **自适应生成** — 质量失败时自动调整温度
- **质量验证** — 相似度、长度比例、词汇多样性检查
- **断点续写** — 保存进度，从最后一个块继续
- **GUI** — 深色主题 tkinter 桌面应用
- **Web UI** — Flask + SSE 实时进度，支持任何浏览器
- **多语言界面** — RU / EN / ZH / DE / FR / ES / JA / KO
- **5 种提示词预设** — 文学、学术、简化、创意、翻译
- **并行处理** — 可选的并行模式，用于更快改写大文档（通过 Web UI 的"并行模式"复选框启用）
- **扩展语言支持** — 支持 8 种语言的文本输出，通过提示词注入实现严格语言强制
- **本地代理集成** — OpenAI 兼容 API（如 LLM-API-Key-Proxy）

---

## 系统要求

- Python 3.10+
- 本地 LLM 代理运行于 `http://127.0.0.1:8000/v1`（OpenAI 兼容）

---

## 安装

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
# 编辑 .env 配置您的代理设置
```

---

## 使用方法

### 桌面 GUI（默认）

```bash
python main.py
# 或明确指定:
python main.py --gui
# 或通过模块:
python -m app.gui
```

### Web 界面

```bash
python main.py --web
# 在浏览器中打开 http://127.0.0.1:5000
```

自定义主机/端口:

```bash
python main.py --web --host 0.0.0.0 --port 8080
```

通过模块:

```bash
python -m app.web --host 0.0.0.0 --port 8080
```

### CLI 界面（占位符）

```bash
python main.py --cli
# 或:
python -m app.cli
```

> **注意:** CLI 是未来实现的占位符。

### Windows 快速启动

```bash
start.bat
```

---

## 配置

从 `.env.example` 创建 `.env`:

```env
# 代理 URL（OpenAI 兼容）
OPENAI_BASE_URL=http://127.0.0.1:8000/v1

# 代理授权令牌
AUTH_TOKEN=sk-admin74203

# 默认模型
MODEL=colin/gpt-5.4

# 上下文窗口大小
CONTEXT_WINDOW=1000000

# 每个块的最大输出令牌数
MAX_OUTPUT_TOKENS=32768

# 块目标大小（字符数）
BLOCK_TARGET_CHARS=15000

# 改写长度比例限制
MIN_REWRITE_LENGTH_RATIO=0.40
MAX_REWRITE_LENGTH_RATIO=1.60

# 界面语言: ru / en / zh
UI_LANG=zh
```

### 支持的界面和输出语言

| 代码 | 语言 |
|------|------|
| `ru` | Русский (俄语) |
| `en` | English (英语) |
| `zh` | 中文 |
| `de` | Deutsch (德语) |
| `fr` | Francais (法语) |
| `es` | Espanol (西班牙语) |
| `ja` | 日本語 (日语) |
| `ko` | 한국어 (韩语) |

#### 语言强制

所有 5 种提示词预设都包含**严格的语言强制块**，根据所选输出语言在运行时注入：

- 模型**仅**以指定语言输出文本
- 不会意外翻译成英语或其他语言
- 即使输入语言不同，输出仍保持一致
- 输出中的每个词都必须匹配目标语言

这对"翻译"预设至关重要，用户需要可靠的输出语言控制，无论源文本语言是什么。

### 并行处理

改写引擎支持**并行处理模式**，用于加速大型文档：

- 通过 Web UI (`python main.py --web`) 的"并行模式"复选框启用
- LLM 同时处理多个文本块
- 显著缩短书籍和长文档的总改写时间
- 每个块的质量验证和上下文感知保持不变
- 通过 `POST /api/start` 端点的表单参数 `parallel_mode=true` 使用

### 连接配置文件

应用程序支持三种连接配置:

| 配置 | 描述 |
|------|------|
| `direct` | 直接连接到 LLM API |
| `proxy` | 通过本地代理连接（推荐） |
| `auto` | 根据可用性自动选择 |

---

## 项目结构

```
AI-rewriter/
├── main.py              # 入口点（GUI 或 web）
├── requirements.txt
├── .env                 # 您的本地配置（不提交）
├── .env.example
│
├── core/                # 改写引擎
│   ├── config.py        # 来自 .env 的所有常量
│   ├── context.py       # GlobalContext（角色、情节、主题）
│   ├── text_engine.py   # 块分割、验证、指标
│   ├── api_client.py    # 到代理的 HTTP 客户端（SSE 流）
│   ├── prompts.py       # 5 个系统提示词预设
│   ├── state_manager.py # 原子状态保存/加载
│   └── rewriter.py      # 主编排循环
│
├── gui/
│   └── app.py           # Tkinter 桌面应用（深色主题，i18n）
│
├── web/
│   ├── app.py           # Flask web 应用 + SSE
│   ├── templates/
│   │   └── index.html   # 单页 UI
│   └── static/
│       ├── style.css    # 深色主题 CSS
│       └── app.js       # SSE 客户端，表单处理
│
├── i18n/
│   ├── ru.json          # 俄语
│   ├── en.json          # 英语
│   └── zh.json          # 中文
│
└── prompts/
    └── catalog.json     # 提示词预设元数据
```

---

## 故障排除

### 常见问题

| 问题 | 解决方案 |
|------|----------|
| `ModuleNotFoundError: No module named 'core'` | 从项目根目录运行，不要从子目录运行 |
| Connection refused to proxy | 确保代理在 `http://127.0.0.1:8000` 运行 |
| 质量验证频繁失败 | 在 `.env` 中调整 `MIN_REWRITE_LENGTH_RATIO` 和 `MAX_REWRITE_LENGTH_RATIO` |
| 内存不足错误 | 减少 `BLOCK_TARGET_CHARS` 或 `CONTEXT_WINDOW` |
| GUI 无法启动 | 检查 Python 版本（需要 3.10+），重新安装 `sv-ttk` |

### 获取帮助

- 查看 [API.md](API.md) 了解 Web API 文档
- 查看 [PROMPTS.md](PROMPTS.md) 了解提示词预设指南
- 在 [GitHub](https://github.com/ShmidtS/AI-rewriter/issues) 上提交 issue

---

## 更新日志

### v2.0.0 (2026-03-30)

- 完全模块化重构：`core/`、`gui/`、`web/`、`i18n/`
- 添加 Flask web 界面，支持 SSE 实时进度
- 添加多语言界面：8 种语言（RU, EN, ZH, DE, FR, ES, JA, KO）
- 添加 5 种提示词预设（文学、学术、简化、创意、翻译）
- 并行处理模式，加速处理大型文档
- 统一入口点 `main.py`（`--web` 标志用于 web 模式）
- `core/api_client.py`: requests + SSE 流，502 重试逻辑
- `core/rewriter.py`: 与 GUI 解耦，可 headless 运行
- `core/prompts.py`: 带 i18n 标签的预设系统
- `core/context.py`: GlobalContext 带 `from_json` 类方法
- Web UI: 深色主题、SSE 事件、文件上传/下载、模型刷新
- i18n 系统带英语回退
- 每个提示词中的严格语言强制机制

### v1.1.0

- 从 Google Gemini 迁移到本地 OpenAI 兼容代理
- SSE 流支持
- 自适应温度

### v1.0.0 (2025-01-30)

- 初始发布
- Tkinter GUI、Gemini API、质量验证、断点续写支持

---

## 作者

ShmidtS — [GitHub](https://github.com/ShmidtS)

---

## 许可证

详见 [LICENSE](../../LICENSE) 文件。
