# Справочник Web API

> Полная документация по эндпоинтам Web API AI-rewriter.

---

## Содержание

- [Обзор](#обзор)
- [Базовый URL](#базовый-url)
- [Аутентификация](#аутентификация)
- [Эндпоинты](#эндпоинты)
  - [POST /api/start](#post-apistart)
  - [POST /api/stop](#post-apistop)
  - [GET /api/status](#get-apistatus)
  - [GET /api/prompts](#get-apiprompts)
  - [GET /api/i18n/<lang>.json](#get-apii18nlangjson)
  - [GET /api/output-languages](#get-apioutput-languages)
  - [GET /events](#get-events)
- [Обработка ошибок](#обработка-ошибок)
- [Примеры](#примеры)

---

## Обзор

AI-rewriter предоставляет RESTful Web API для управления задачами переписывания. API поддерживает:

- Запуск и остановку задач переписывания
- Обновление прогресса в реальном времени через Server-Sent Events (SSE)
- Поддержку многоязычного интерфейса
- Управление пресетами промптов

---

## Базовый URL

```
http://127.0.0.1:5000
```

Для кастомного развертывания:

```
http://<хост>:<порт>
```

---

## Аутентификация

В настоящее время API не требует аутентификации. Аутентификация осуществляется на уровне прокси (LLM-API-Key-Proxy).

---

## Эндпоинты

### POST /api/start

Запустить новую задачу переписывания.

**Запрос**

- Content-Type: `multipart/form-data`

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `input_file` | File | Да | Входной текстовый файл (.txt, макс. 100MB) |
| `language` | String | Да | Язык вывода (например, "Русский", "English") |
| `model` | String | Да | Название модели (например, "gemini/gemini-2.5-flash") |
| `prompt_preset` | String | Да | Ключ пресета: `literary`, `academic`, `simplified`, `creative`, `translation` |
| `style` | String | Нет | Дополнительные инструкции по стилю |
| `goal` | String | Нет | Дополнительные инструкции по цели |
| `resume` | Boolean | Нет | Продолжить с последней позиции (по умолчанию: `true`) |

**Ответ**

```json
{
  "status": "started",
  "output": "input_filename_final.txt",
  "message": "Rewrite job started successfully"
}
```

**Ответ с ошибкой**

```json
{
  "error": "No input file provided"
}
```

**Коды статуса**

| Код | Описание |
|-----|----------|
| 200 | Задача успешно запущена |
| 400 | Неверный запрос (отсутствуют параметры) |
| 409 | Конфликт (задача уже выполняется) |
| 500 | Внутренняя ошибка сервера |

---

### POST /api/stop

Остановить текущую задачу переписывания.

**Запрос**

Параметры не требуются.

**Ответ**

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

**Коды статуса**

| Код | Описание |
|-----|----------|
| 200 | Задача успешно остановлена |
| 404 | Нет выполняемой задачи |

---

### GET /api/status

Получить текущий статус задачи.

**Ответ**

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

**Поля**

| Поле | Тип | Описание |
|------|-----|----------|
| `running` | Boolean | Выполняется ли задача в данный момент |
| `current` | Integer | Количество обработанных блоков |
| `total` | Integer | Общее количество блоков |
| `percentage` | Float | Процент прогресса (0-100) |
| `filename` | String | Текущее имя входного файла |
| `output` | String | Имя выходного файла |

---

### GET /api/prompts

Получить доступные пресеты промптов.

**Ответ**

```json
{
  "literary": "Литературный редактор",
  "academic": "Академический стиль",
  "simplified": "Упрощение",
  "creative": "Творческое улучшение",
  "translation": "Перевод с адаптацией"
}
```

---

### GET /api/i18n/<lang>.json

Получить переводы для конкретного языка.

**Параметры**

| Параметр | Тип | Обязательный | Описание |
|----------|-----|--------------|----------|
| `lang` | String | Да | Код языка: `ru`, `en`, `zh` |

**Ответ**

```json
{
  "_language_name": "Русский",
  "app_title": "AI Книжный Переписчик",
  "start": "Запустить",
  "stop": "Остановить",
  "input_file": "Входной файл",
  "output_language": "Язык вывода",
  "model": "Модель",
  "prompt_preset": "Пресет промпта",
  "style": "Стиль (опционально)",
  "goal": "Цель (опционально)",
  "resume": "Продолжить с последней позиции",
  "progress": "Прогресс",
  "blocks_processed": "блоков обработано",
  "download_result": "Скачать результат",
  "connected": "Подключено",
  "disconnected": "Отключено",
  "select_file": "Выбрать файл...",
  "refresh_models": "Обновить модели",
  "error_no_file": "Пожалуйста, выберите файл",
  "error_job_running": "Задача уже выполняется",
  "error_generic": "Произошла ошибка"
}
```

**Коды статуса**

| Код | Описание |
|-----|----------|
| 200 | Переводы найдены |
| 404 | Язык не найден |

---

### GET /api/output-languages

Получить список поддерживаемых языков вывода.

**Ответ**

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

Эндпоинт Server-Sent Events (SSE) для обновлений в реальном времени.

**Типы событий**

| Событие | Описание | Данные |
|---------|----------|--------|
| `status` | Обновление статуса задачи | `{"running": bool, "current": int, "total": int}` |
| `progress` | Обновление прогресса | `{"current": int, "total": int, "pct": float}` |
| `log` | Лог-сообщение | `{"msg": string}` |
| `done` | Задача завершена | `{"output": string}` |
| `error` | Произошла ошибка | `{"error": string}` |

**Пример SSE-потока**

```
event: status
data: {"running": true, "current": 0, "total": 100}

event: log
data: {"msg": "Запуск задачи переписывания..."}

event: progress
data: {"current": 1, "total": 100, "pct": 1.0}

event: log
data: {"msg": "Блок 1 успешно обработан"}

event: progress
data: {"current": 2, "total": 100, "pct": 2.0}

...

event: done
data: {"output": "input_file_final.txt"}
```

**Пример JavaScript-клиента**

```javascript
const eventSource = new EventSource('/events');

eventSource.addEventListener('status', (e) => {
  const data = JSON.parse(e.data);
  console.log('Статус:', data);
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
  console.error('SSE ошибка:', e);
  // Автопереподключение через 3 секунды
  setTimeout(() => location.reload(), 3000);
});
```

---

## Обработка ошибок

Все ответы с ошибками имеют следующий формат:

```json
{
  "error": "Описание ошибки"
}
```

Частые сообщения об ошибках:

| Ошибка | Описание |
|--------|----------|
| `No input file provided` | Отсутствует `input_file` в запросе |
| `Invalid file type` | Файл не является .txt файлом |
| `File too large` | Файл превышает лимит 100MB |
| `A job is already running` | Нельзя запустить, пока выполняется другая задача |
| `No job running` | Нельзя остановить, когда нет активной задачи |
| `Model not available` | Выбранная модель недоступна |
| `Proxy connection failed` | Не удалось подключиться к LLM прокси |

---

## Примеры

### Запуск задачи переписывания (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/start \
  -F "input_file=@my_book.txt" \
  -F "language=Русский" \
  -F "model=gemini/gemini-2.5-flash" \
  -F "prompt_preset=literary" \
  -F "resume=true"
```

### Остановка задачи (curl)

```bash
curl -X POST http://127.0.0.1:5000/api/stop
```

### Получение статуса (curl)

```bash
curl http://127.0.0.1:5000/api/status
```

### Пример JavaScript Fetch

```javascript
// Запуск задачи
async function startJob(formData) {
  const response = await fetch('/api/start', {
    method: 'POST',
    body: formData
  });
  return response.json();
}

// Остановка задачи
async function stopJob() {
  const response = await fetch('/api/stop', {
    method: 'POST'
  });
  return response.json();
}

// Получение статуса
async function getStatus() {
  const response = await fetch('/api/status');
  return response.json();
}
```

### Пример Python Requests

```python
import requests

# Запуск задачи
with open('my_book.txt', 'rb') as f:
    response = requests.post('http://127.0.0.1:5000/api/start', files={
        'input_file': f
    }, data={
        'language': 'Русский',
        'model': 'gemini/gemini-2.5-flash',
        'prompt_preset': 'literary'
    })
    print(response.json())

# Остановка задачи
response = requests.post('http://127.0.0.1:5000/api/stop')
print(response.json())

# Получение статуса
response = requests.get('http://127.0.0.1:5000/api/status')
print(response.json())
```

---

## Ограничение запросов

API не реализует ограничение запросов. Однако, базовый LLM прокси может иметь собственные лимиты.

---

## CORS

CORS включён для всех источников в режиме разработки. Для продакшена настройте разрешённые источники в `web/app.py`.

---

## См. также

- [README.md](README.md) - Общая документация
- [PROMPTS.md](PROMPTS.md) - Руководство по пресетам промптов
