# Web API Reference

> Complete documentation for AI-rewriter Web API endpoints.

---

## Table of Contents

- [Overview](#overview)
- [Base URL](#base-url)
- [Authentication](#authentication)
- [Endpoints](#endpoints)
  - [POST /api/start](#post-apistart)
  - [POST /api/stop](#post-apistop)
  - [GET /api/status](#get-apistatus)
  - [GET /api/models](#get-apimodels)
  - [GET /api/prompts](#get-apiprompts)
  - [GET /api/prompts/<prompt_id>](#get-apipromptsprompt_id)
  - [GET /api/prompts/categories](#get-apipromptscategories)
  - [GET /api/output-languages](#get-apioutput-languages)
  - [GET /api/download/<filename>](#get-apidownloadfilename)
  - [GET /api/i18n/<lang>.json](#get-apii18nlangjson)
  - [GET /events](#get-events)
- [Language Enforcement](#language-enforcement)
- [Parallel Mode](#parallel-mode)
- [Error Handling](#error-handling)
- [Examples](#examples)

---

## Overview

AI-rewriter provides a RESTful Web API for managing rewrite jobs. The API supports:

- Starting and stopping rewrite jobs
- Real-time progress updates via Server-Sent Events (SSE)
- Multi-language interface support
- Prompt preset management

---

## Base URL

```
http://127.0.0.1:5000
```

For custom deployment:

```
http://<host>:<port>
```

---

## Authentication

Currently, the API does not require authentication. Authentication is handled at the proxy level (LLM-API-Key-Proxy).

---

## Endpoints

### POST /api/start

Start a new rewrite job.

**Request**

- Content-Type: `multipart/form-data`

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `input_file` | File | Yes | Input text file (.txt, max 100MB) |
| `language` | String | Yes | Output language (e.g., "Русский", "English") |
| `model` | String | Yes | Model name (e.g., "gemini/gemini-2.5-flash") |
| `prompt_preset` | String | Yes | Prompt preset key: `literary`, `academic`, `simplified`, `creative`, `translation` |
| `style` | String | No | Additional style instructions |
| `goal` | String | No | Additional goal instructions |
| `resume` | Boolean | No | Resume from last position (default: `true`) |
| `output_file` | String | No | Custom output filename (auto-generated if not provided) |
| `parallel_mode` | Boolean | No | Enable parallel block processing for faster rewrites (default: `false`) |

**Response**

```json
{
  "status": "started",
  "output": "input_filename_final.txt",
  "message": "Rewrite job started successfully"
}
```

**Error Response**

```json
{
  "error": "No input file provided"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Job started successfully |
| 400 | Bad request (missing parameters) |
| 409 | Conflict (job already running) |
| 500 | Internal server error |

---

### POST /api/stop

Stop the current rewrite job.

**Request**

No parameters required.

**Response**

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

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Job stopped successfully |
| 404 | No job running |

---

### GET /api/status

Get the current job status.

**Response**

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

**Fields**

| Field | Type | Description |
|-------|------|-------------|
| `running` | Boolean | Whether a job is currently running |
| `current` | Integer | Number of blocks processed |
| `total` | Integer | Total number of blocks |
| `percentage` | Float | Progress percentage (0-100) |
| `filename` | String | Current input filename |
| `output` | String | Output filename |

---

### GET /api/models

Get the list of available LLM models from the proxy.

**Response**

```json
["gemini/gemini-2.5-flash", "colin/gpt-5.4", "openai/gpt-4o"]
```

---

### GET /api/prompts

Get all prompts with localized names and descriptions.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lang` | String | No | Language code for localization (default: `en`) |

**Response**

```json
{
  "prompts": [
    {
      "id": "literary",
      "name": "Literary Editor",
      "description": "Professional literary editing preserving meaning while improving style",
      "category": "editing",
      "preview": "Literary editor for books and fiction"
    }
  ]
}
```

---

### GET /api/prompts/<prompt_id>

Get a specific prompt by ID with localized content.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `prompt_id` | String | Yes | Prompt ID: `literary`, `academic`, `simplified`, `creative`, `translation` |

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lang` | String | No | Language code for localization (default: `en`) |

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Prompt found |
| 404 | Prompt not found |

---

### GET /api/prompts/categories

Get all prompt categories with localized names.

**Query Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lang` | String | No | Language code for localization (default: `en`) |

**Response**

```json
{
  "categories": [
    {"id": "editing", "name": "Editing"},
    {"id": "adaptation", "name": "Adaptation"},
    {"id": "enhancement", "name": "Enhancement"},
    {"id": "translation", "name": "Translation"}
  ]
}
```

---

### GET /api/output-languages

Get the list of supported output/translation languages.

**Response**

```json
{
  "languages": [
    {"code": "ru", "name": "Русский"},
    {"code": "en", "name": "English"},
    {"code": "zh", "name": "中文"},
    {"code": "de", "name": "Deutsch"},
    {"code": "fr", "name": "Francais"},
    {"code": "es", "name": "Espanol"},
    {"code": "ja", "name": "日本語"},
    {"code": "ko", "name": "한국어"}
  ]
}
```

---

### GET /api/download/<filename>

Download a completed rewrite output file.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `filename` | String | Yes | Name of the output file in the outputs directory |

**Response**

Binary file download with `Content-Disposition: attachment` header.

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | File downloaded successfully |
| 404 | File not found |

---

### Language Enforcement

All prompt presets include a strict language enforcement block injected at runtime. When you specify a language via the `language` form parameter, the system injects explicit instructions into the system prompt:

- Output MUST be only in the specified language
- No translation to any other language
- Every word must match the target language
- Even if input is in a different language, output stays in the requested language

This is critical for maintaining consistent output, especially with the Translation preset.

### Parallel Mode

The `parallel_mode` parameter on `POST /api/start` enables concurrent block processing:

- **`parallel_mode=false`** (default): Sequential processing, one block at a time
- **`parallel_mode=true`**: Multiple blocks processed concurrently

Parallel mode is recommended for large documents (books, long articles) to reduce total processing time. The API returns the same response format regardless of mode.

---

### GET /api/i18n/<lang>.json

Get translations for a specific language.

**Parameters**

| Parameter | Type | Required | Description |
|-----------|------|----------|-------------|
| `lang` | String | Yes | Language code: `ru`, `en`, `zh` |

**Response**

```json
{
  "_language_name": "English",
  "app_title": "AI Book Rewriter",
  "start": "Start",
  "stop": "Stop",
  "input_file": "Input File",
  "output_language": "Output Language",
  "model": "Model",
  "prompt_preset": "Prompt Preset",
  "style": "Style (optional)",
  "goal": "Goal (optional)",
  "resume": "Resume from last position",
  "progress": "Progress",
  "blocks_processed": "blocks processed",
  "download_result": "Download Result",
  "connected": "Connected",
  "disconnected": "Disconnected",
  "select_file": "Select file...",
  "refresh_models": "Refresh Models",
  "error_no_file": "Please select a file",
  "error_job_running": "A job is already running",
  "error_generic": "An error occurred"
}
```

**Status Codes**

| Code | Description |
|------|-------------|
| 200 | Transations found |
| 404 | Language not found |

---

### GET /api/output-languages

Get the list of supported output languages.

**Response**

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

Server-Sent Events (SSE) endpoint for real-time updates.

**Event Types**

| Event | Description | Data |
|-------|-------------|------|
| `status` | Job status update | `{"running": bool, "current": int, "total": int}` |
| `progress` | Progress update | `{"current": int, "total": int, "pct": float}` |
| `log` | Log message | `{"msg": string}` |
| `done` | Job completed | `{"output": string}` |
| `error` | Error occurred | `{"error": string}` |

**Example SSE Stream**

```
event: status
data: {"running": true, "current": 0, "total": 100}

event: log
data: {"msg": "Starting rewrite job..."}

event: progress
data: {"current": 1, "total": 100, "pct": 1.0}

event: log
data: {"msg": "Block 1 processed successfully"}

event: progress
data: {"current": 2, "total": 100, "pct": 2.0}

...

event: done
data: {"output": "input_file_final.txt"}
```

**JavaScript Client Example**

```javascript
const eventSource = new EventSource('/events');

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  console.log('Status:', data);
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
  console.error('SSE error:', e);
  // Auto-reconnect after 3 seconds
  setTimeout(() => location.reload(), 3000);
});
```

---

## Error Handling

All error responses follow this format:

```json
{
  "error": "Error message description"
}
```

Common error messages:

| Error | Description |
|-------|-------------|
| `No input file provided` | Missing `input_file` in request |
| `Invalid file type` | File is not a .txt file |
| `File too large` | File exceeds 100MB limit |
| `A job is already running` | Cannot start while another job is active |
| `No job running` | Cannot stop when no job is active |
| `Model not available` | Selected model is not accessible |
| `Proxy connection failed` | Cannot connect to LLM proxy |

---

## Examples

### Start a Rewrite Job (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/start \
  -F "input_file=@my_book.txt" \
  -F "language=English" \
  -F "model=gemini/gemini-2.5-flash" \
  -F "prompt_preset=literary" \
  -F "resume=true"
```

### Stop a Job (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/stop
```

### Get Status (curl)

```bash
curl http://127.0.0.1:5000/api/status
```

### JavaScript Fetch Example

```javascript
// Start job
async function startJob(formData) {
  const response = await fetch('/api/start', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// Stop job
async function stopJob() {
  const response = await fetch('/api/stop', {
    method: 'POST'
  });
  return response.json();
}

// Get status
async function getStatus() {
  const response = await fetch('/api/status');
  return response.json();
}
```

### Python Requests Example

```python
import requests

# Start job
with open('my_book.txt', 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/api/start', files={
        'input_file': f
    }, data={
        'language': 'English',
        'model': 'gemini/gemini-2.5-flash',
        'prompt_preset': 'literary'
    })
    print(response.json())

# Stop job
response = requests.post('http://127.0.0.1:5000/api/stop')
print(response.json())

# Get status
response = requests.get('http://127.0.0.1:5000/api/status')
print(response.json())
```

---

## Rate Limiting

The API does not implement rate limiting. However, the underlying LLM proxy may have its own rate limits.

---

## CORS

CORS is enabled for all origins in development. For production, configure allowed origins in `web/app.py`.

---

## See Also

- [README.md](README.md) - General documentation
- [PROMPTS.md](PROMPTS.md) - Prompt presets guide
