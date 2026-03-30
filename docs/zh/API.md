# Web API 参考

> AI-rewriter Web API 端点完整文档。

---

## 目录

- [概述](#概述)
- [基础 URL](#基础-url)
- [身份验证](#身份验证)
- [端点](#端点)
  - [POST /api/start](#post-apistart)
  - [POST /api/stop](#post-apistop)
  - [GET /api/status](#get-apistatus)
  - [GET /api/prompts](#get-apiprompts)
  - [GET /api/i18n/<lang>.json](#get-apii18nlangjson)
  - [GET /api/output-languages](#get-apioutput-languages)
  - [GET /events](#get-events)
- [错误处理](#错误处理)
- [示例](#示例)

---

## 概述

AI-rewriter 提供 RESTful Web API 用于管理改写任务。API 支持：

- 启动和停止改写任务
- 通过 Server-Sent Events (SSE) 实时更新进度
- 多语言界面支持
- 提示词预设管理

---

## 基础 URL

```
http://127.0.0.1:5000
```

自定义部署：

```
http://<主机>:<端口>
```

---

## 身份验证

目前 API 不需要身份验证。身份验证在代理层面处理（LLM-API-Key-Proxy）。

---

## 端点

### POST /api/start

启动新的改写任务。

**请求**

- Content-Type: `multipart/form-data`

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `input_file` | File | 是 | 输入文本文件（.txt，最大 100MB） |
| `language` | String | 是 | 输出语言（如 "中文"、"English"） |
| `model` | String | 是 | 模型名称（如 "gemini/gemini-2.5-flash"） |
| `prompt_preset` | String | 是 | 提示词预设键：`literary`、`academic`、`simplified`、`creative`、`translation` |
| `style` | String | 否 | 附加风格说明 |
| `goal` | String | 否 | 附加目标说明 |
| `resume` | Boolean | 否 | 从上次位置继续（默认：`true`） |

**响应**

```json
{
  "status": "started",
  "output": "input_filename_final.txt",
  "message": "Rewrite job started successfully"
}
```

**错误响应**

```json
{
  "error": "No input file provided"
}
```

**状态码**

| 代码 | 描述 |
|------|------|
| 200 | 任务启动成功 |
| 400 | 错误请求（缺少参数） |
| 409 | 冲突（任务已在运行） |
| 500 | 内部服务器错误 |

---

### POST /api/stop

停止当前改写任务。

**请求**

无需参数。

**响应**

```json
{
  "status": "stopped",
  "message": "Job stopped successfully",
  "progress": {
    "current": 42,
    "total": 100,
    "percentage": 42.0
  }
}
```

**状态码**

| 代码 | 描述 |
|------|------|
| 200 | 任务停止成功 |
| 404 | 无运行中的任务 |

---

### GET /api/status

获取当前任务状态。

**响应**

```json
{
  "running": false,
  "current": 42,
  "total": 100,
  "percentage": 42.0,
  "filename": "input_file.txt",
  "output": "input_file_final.txt"
}
```

**字段**

| 字段 | 类型 | 描述 |
|------|------|------|
| `running` | Boolean | 任务是否正在运行 |
| `current` | Integer | 已处理块数 |
| `total` | Integer | 总块数 |
| `percentage` | Float | 进度百分比（0-100） |
| `filename` | String | 当前输入文件名 |
| `output` | String | 输出文件名 |

---

### GET /api/prompts

获取可用的提示词预设。

**响应**

```json
{
  "literary": "文学编辑",
  "academic": "学术风格",
  "simplified": "简化",
  "creative": "创意增强",
  "translation": "翻译与改编"
}
```

---

### GET /api/i18n/<lang>.json

获取特定语言的翻译。

**参数**

| 参数 | 类型 | 必需 | 描述 |
|------|------|------|------|
| `lang` | String | 是 | 语言代码：`ru`、`en`、`zh` |

**响应**

```json
{
  "_language_name": "中文",
  "app_title": "AI 书籍改写工具",
  "start": "启动",
  "stop": "停止",
  "input_file": "输入文件",
  "output_language": "输出语言",
  "model": "模型",
  "prompt_preset": "提示词预设",
  "style": "风格（可选）",
  "goal": "目标（可选）",
  "resume": "从上次位置继续",
  "progress": "进度",
  "blocks_processed": "块已处理",
  "download_result": "下载结果",
  "connected": "已连接",
  "disconnected": "已断开",
  "select_file": "选择文件...",
  "refresh_models": "刷新模型",
  "error_no_file": "请选择文件",
  "error_job_running": "任务已在运行",
  "error_generic": "发生错误"
}
```

**状态码**

| 代码 | 描述 |
|------|------|
| 200 | 翻译已找到 |
| 404 | 语言未找到 |

---

### GET /api/output-languages

获取支持的输出语言列表。

**响应**

```json
{
  "languages": [
    {"code": "ru", "name": "Русский"},
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "中文"},
    {"code": "es", "name": "Español"},
    {"code": "fr", "name": "Français"},
    {"code": "de", "name": "Deutsch"},
    {"code": "it", "name": "Italiano"},
    {"code": "pt", "name": "Português"},
    {"code": "ja", "name": "日本語"},
    {"code": "ko", "name": "한국어"}
  ]
}
```

---

### GET /events

Server-Sent Events (SSE) 端点，用于实时更新。

**事件类型**

| 事件 | 描述 | 数据 |
|------|------|------|
| `status` | 任务状态更新 | `{"running": bool, "current": int, "total": int}` |
| `progress` | 进度更新 | `{"current": int, "total": int, "pct": float}` |
| `log` | 日志消息 | `{"msg": string}` |
| `done` | 任务完成 | `{"output": string}` |
| `error` | 发生错误 | `{"error": string}` |

**SSE 流示例**

```
event: status
data: {"running": true, "current": 0, "total": 100}

event: log
data: {"msg": "正在启动改写任务..."}

event: progress
data: {"current": 1, "total": 100, "pct": 1.0}

event: log
data: {"msg": "块 1 处理成功"}

event: progress
data: {"current": 2, "total": 100, "pct": 2.0}

...

event: done
data: {"output": "input_file_final.txt"}
```

**JavaScript 客户端示例**

```javascript
const eventSource = new EventSource('/events');

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  console.log('状态:', data);
});

eventSource.addEventListener('progress', (e) => {
  const data = JSON.parse(e.data);
  updateProgressBar(data.current, data.total, data.pct);
});

eventSource.addEventListener('log', (e) => {
  const data = JSON.parse(e.data);
  appendLog(data.msg);
});

eventSource.addEventListener('done', (e) => {
  const data = JSON.parse(e.data);
  showDownloadLink(data.output);
  eventSource.close();
});

eventSource.addEventListener('error', (e) => {
  console.error('SSE 错误:', e);
  // 3 秒后自动重连
  setTimeout(() => location.reload(), 3000);
});
```

---

## 错误处理

所有错误响应遵循此格式：

```json
{
  "error": "错误消息描述"
}
```

常见错误消息：

| 错误 | 描述 |
|------|------|
| `No input file provided` | 请求中缺少 `input_file` |
| `Invalid file type` | 文件不是 .txt 文件 |
| `File too large` | 文件超过 100MB 限制 |
| `A job is already running` | 另一个任务运行中时无法启动 |
| `No job running` | 无活动任务时无法停止 |
| `Model not available` | 所选模型不可访问 |
| `Proxy connection failed` | 无法连接到 LLM 代理 |

---

## 示例

### 启动改写任务 (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/start \
  -F "input_file=@my_book.txt" \
  -F "language=中文" \
  -F "model=gemini/gemini-2.5-flash" \
  -F "prompt_preset=literary" \
  -F "resume=true"
```

### 停止任务 (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/stop
```

### 获取状态 (curl)

```bash
curl http://127.0.0.1:5000/api/status
```

### JavaScript Fetch 示例

```javascript
// 启动任务
async function startJob(formData) {
  const response = await fetch('/api/start', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// 停止任务
async function stopJob() {
  const response = await fetch('/api/stop', {
    method: 'POST'
  });
  return response.json();
}

// 获取状态
async function getStatus() {
  const response = await fetch('/api/status');
  return response.json();
}
```

### Python Requests 示例

```python
import requests

# 启动任务
with open('my_book.txt', 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/api/start', files={
        'input_file': f
    }, data={
        'language': '中文',
        'model': 'gemini/gemini-2.5-flash',
        'prompt_preset': 'literary'
    })
    print(response.json())

# 停止任务
response = requests.post('http://127.0.0.1:5000/api/stop')
print(response.json())

# 获取状态
response = requests.get('http://127.0.0.1:5000/api/status')
print(response.json())
```

---

## 速率限制

API 不实现速率限制。但是，底层 LLM 代理可能有自己的限制。

---

## CORS

开发环境中 CORS 对所有来源启用。生产环境请在 `web/app.py` 中配置允许的来源。

---

## 另见

- [README.md](README.md) - 通用文档
- [PROMPTS.md](PROMPTS.md) - 提示词预设指南
