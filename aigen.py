import google.generativeai as genai
import os
import time
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple, TypedDict
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import difflib
from logging.handlers import RotatingFileHandler
import sv_ttk
import re
import string

# Глобальные переменные
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

# Загрузка API-ключа
try:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        logger.warning("Переменная GOOGLE_API_KEY не установлена.")
except Exception as e:
    logger.error(f"Ошибка загрузки .env: {e}")
    GEMINI_API_KEY = None

# Константы
AVAILABLE_MODELS = [
    "gemini-2.0-pro-exp", "gemini-2.0-flash-exp",
    "gemini-1.5-pro", "gemini-1.5-pro-001", "gemini-1.5-flash-001-tuning", "gemini-1.5-flash-002", 
    "gemini-2.5-pro-exp-03-25", 
    "gemini-2.0-pro-exp-02-05", 
    "gemini-2.0-flash-thinking-exp-01-21", "gemini-2.0-flash-thinking-exp", "gemini-2.0-flash-thinking-exp-1219", 
    "learnlm-2.0-flash-experimental"
]
REWRITER_MODEL_DEFAULT = "gemini-2.0-flash-exp"
START_MARKER = "<|~START_REWRITE~|>"
END_MARKER = "<|~END_REWRITE~|>"
BLOCK_TARGET_CHARS = 9000
MIN_REWRITE_LENGTH_RATIO = 0.50
MAX_REWRITE_LENGTH_RATIO = 1.8
SIMILARITY_THRESHOLD = 0.95
OUTPUT_TOKEN_LIMIT = 4096
MIN_BLOCK_LEN_FACTOR = 0.5
MAX_BLOCK_LEN_FACTOR = 1.5
SEARCH_RADIUS_FACTOR = 0.1
SPLIT_PRIORITY = ['. ', '! ', '? ']
MAX_RETRIES = 50
RETRY_DELAY_SECONDS = 3

SAFETY_SETTINGS = {
    'HARM_CATEGORY_HARASSMENT': 'BLOCK_NONE',
    'HARM_CATEGORY_HATE_SPEECH': 'BLOCK_NONE',
    'HARM_CATEGORY_SEXUALLY_EXPLICIT': 'BLOCK_NONE',
    'HARM_CATEGORY_DANGEROUS_CONTENT': 'BLOCK_NONE',
}
GENERATION_CONFIG = genai.types.GenerationConfig(
    temperature=0.3, top_p=0.95, max_output_tokens=OUTPUT_TOKEN_LIMIT
)

STATE_SUFFIX = "_rewrite_state.json"
INTERMEDIATE_SUFFIX = "_intermediate.txt"
FINAL_SUFFIX = "_final_rewritten.txt"

# Типы данных
class BlockInfo(TypedDict):
    block_index: int
    start_char_index: int
    end_char_index: int
    original_char_length: int
    processed: bool
    failed_attempts: int

# Обработчик логов для GUI
class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

# Конфигурация Gemini API
def configure_gemini():
    if not GEMINI_API_KEY:
        raise ValueError("API-ключ Gemini не установлен.")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        logger.info("Gemini API успешно сконфигурирован.")
        return True
    except Exception as e:
        logger.error(f"Ошибка конфигурации Gemini API: {e}")
        raise ValueError(f"Ошибка конфигурации: {e}")

# Получение доступных моделей
def list_available_models():
    try:
        models = [m.name.split('/')[-1] for m in genai.list_models() if 'generateContent' in m.supported_generation_methods]
        logger.info(f"Доступные модели: {', '.join(models)}")
        return models
    except Exception as e:
        logger.error(f"Ошибка получения моделей: {e}")
        return AVAILABLE_MODELS

# Подсчет символов
def count_chars(text: str) -> int:
    return len(text) if isinstance(text, str) else 0

def split_into_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по '. ', '! ', '? '."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def has_more_than_n_words(sentence: str, n: int = 5) -> bool:
    """Проверяет, содержит ли предложение больше n слов."""
    words = sentence.split()
    return len(words) > n

def normalize_sentence(sentence: str) -> str:
    """Приводит предложение к нижнему регистру и убирает конечные знаки препинания."""
    sentence = sentence.lower()
    sentence = sentence.rstrip(string.punctuation)
    return sentence

# Проверка ответа API
def check_api_response(response: genai.types.GenerateContentResponse, context: str) -> Tuple[Optional[str], Optional[str], bool]:
    text, error, max_tokens = None, None, False

    if response.prompt_feedback.block_reason:
        error = f"{context}: Промпт заблокирован ({response.prompt_feedback.block_reason.name})."
        logger.error(error)
        return None, error, False

    if not response.candidates:
        try:
            text = response.text.strip()
            if text:
                logger.warning(f"{context}: Нет кандидатов, используется fallback текст ({count_chars(text)} симв.).")
                return text, None, False
            error = f"{context}: Нет кандидатов и пустой fallback текст."
            logger.error(error)
            return None, error, False
        except Exception as e:
            error = f"{context}: Ошибка получения текста: {e}."
            logger.error(error)
            return None, error, False

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, 'finish_reason', None)
    finish_value = int(finish_reason.value) if finish_reason else 0

    if finish_value == 3:
        error = f"{context}: Ответ заблокирован фильтром безопасности."
        logger.error(error)
        return None, error, False
    elif finish_value == 2:
        max_tokens = True
        logger.warning(f"{context}: Ответ усечен (MAX_TOKENS).")

    try:
        if candidate.content and candidate.content.parts:
            text = "".join(part.text for part in candidate.content.parts).strip()
        elif response.text:
            text = response.text.strip()
        else:
            text = ""
        return text, None, max_tokens
    except Exception as e:
        error = f"{context}: Ошибка извлечения текста: {e}."
        logger.error(error)
        return None, error, False

# Поиск точки разделения текста
def find_split_point(text: str, start: int, target_end: int, min_len: int, max_len: int) -> int:
    text_len = len(text)
    ideal_end = min(text_len, max(start + min_len, min(target_end, start + max_len)))

    if ideal_end >= text_len:
        return text_len

    radius = int((target_end - start) * SEARCH_RADIUS_FACTOR)
    search_start = max(start + min_len, ideal_end - radius)
    search_end = min(text_len, ideal_end + radius, start + max_len)

    best_point = -1
    min_dist = float('inf')

    for seq in SPLIT_PRIORITY:
        pos = search_start
        while pos < search_end:
            try:
                idx = text.index(seq, pos, search_end) + len(seq)
                if idx > start + min_len:
                    dist = abs(idx - ideal_end)
                    if dist < min_dist:
                        min_dist = dist
                        best_point = idx
                pos = idx + 1
            except ValueError:
                break
        if best_point != -1:
            return best_point

    try:
        last_space = text.rindex(' ', search_start, min(ideal_end, start + max_len))
        if last_space > start:
            return last_space + 1
    except ValueError:
        pass

    return min(ideal_end, start + max_len)

# Разбиение текста на блоки
def split_into_blocks(text: str, target_size: int) -> Optional[List[BlockInfo]]:
    logger.info("Разбиение текста на блоки...")
    text_len = count_chars(text)
    if not text_len:
        logger.warning("Текст пуст.")
        return None

    blocks: List[BlockInfo] = []
    current_pos = 0
    block_idx = 0
    min_len = int(target_size * MIN_BLOCK_LEN_FACTOR)
    max_len = int(target_size * MAX_BLOCK_LEN_FACTOR)

    while current_pos < text_len:
        target_end = current_pos + target_size
        end_pos = find_split_point(text, current_pos, target_end, min_len, max_len)

        if end_pos <= current_pos:
            end_pos = text_len

        block_len = end_pos - current_pos
        blocks.append({
            'block_index': block_idx,
            'start_char_index': current_pos,
            'end_char_index': end_pos,
            'original_char_length': block_len,
            'processed': False,
            'failed_attempts': 0
        })
        current_pos = end_pos
        block_idx += 1

    logger.info(f"Создано {len(blocks)} блоков.")
    return blocks

# Формирование промпта для переписывания
def create_rewrite_prompt(language: str, style: str, goal: str, text_with_markers: str, original_len: int) -> str:
    min_len = int(original_len * MIN_REWRITE_LENGTH_RATIO)
    max_len = int(original_len * MAX_REWRITE_LENGTH_RATIO)
    return f"""**System Prompt v2.4 - Book Rewriter Agent**

**I. Core Directive**

You are a specialized AI Text Rewriter Agent. Your sole function is to rewrite a specific segment of text provided within a larger context, strictly adhering to the parameters and constraints given in each request. You operate as a component within a larger automated workflow that processes text in blocks. Your output must be precise, conformant, and directly usable by this workflow.

**II. Task Definition**

1.  **Identify Target Segment:** The input text will contain a segment clearly marked by  `{START_MARKER}` and `{END_MARKER}`. Your task is to rewrite *only* the text located *between* these two markers.
2.  **Context Awareness:** The text *before* the `{START_MARKER}` and *after* the `{END_MARKER}` is provided solely for context. **Do NOT modify, rewrite, or include this context in your output.**
3.  **Rewrite Parameters:** You will be given specific parameters for each rewriting task:
    *   **Language:** The target language for the rewritten text (e.g., "Русский", "English").
    *   **Style:** A description of the desired writing style (e.g., "Formal academic", "Engaging narrative", "Simple and direct").
    *   **Goal:** The specific objective of the rewrite (e.g., "Improve clarity", "Simplify vocabulary", "Adapt for a younger audience", "Increase detail").
    *   **Approximate Target Length:** A suggested character length range for the rewritten segment (e.g., "~{min_len}-{max_len} characters"). This is a guideline; prioritize quality and constraints over exact length adherence if necessary, but stay reasonably within the bounds.

**III. Strict Operational Constraints & Instructions**

1.  **Focus Exclusively:** Rewrite *only* the text content found strictly between `{START_MARKER}` and `{END_MARKER}`.
2.  **Parameter Adherence:** Strictly follow the specified `Language`, `Style`, and `Goal` parameters.
3.  **Meaning Preservation:** Preserve the core meaning, information, and narrative intent of the original segment unless the `Goal` explicitly dictates otherwise (e.g., simplification might remove nuance).
4.  **Contextual Cohesion:** Ensure the rewritten segment logically connects with the surrounding (unmodified) context provided before the `{START_MARKER}` and after the `{END_MARKER}`. Maintain smooth transitions.
5.  **Length Guideline:** Aim for a character count within the suggested `Approximate Target Length` range. Significant deviation should only occur if strictly necessary to meet other constraints (like Style or Goal).
6.  **CRITICAL - Avoid High Similarity:** The rewritten text *must be substantially different* from the original text segment. Direct copying or minor paraphrasing that results in high textual similarity (e.g., >90-95% similar) is **unacceptable**. The rewrite should be a genuine transformation.
7.  **CRITICAL - Avoid Context Sentence Repetition:** The rewritten text *must not* contain full sentences that are identical (or near-identical after normalization like lowercasing and punctuation removal) to full sentences present in the immediate context provided *before* the `{START_MARKER}` or *after* the `{END_MARKER}`. Pay close attention to the boundaries.
8.  **Output Format:**
    *   Generate *only* the rewritten text corresponding to the segment between the markers.
    *   **Do NOT include the `{START_MARKER}` or `{END_MARKER}` in your output.**
    *   **Do NOT include any of the surrounding context in your output.**
    *   **Do NOT add any explanations, apologies, or introductory/concluding remarks.** Your output must be *only* the rewritten string, ready for direct substitution into the larger text.


**IV. Execution Logic**

1.  Parse the `Parameters` and `Instructions`.
2.  Isolate the `[Original text segment to be rewritten.]` from the `Text:` section.
3.  Note the immediate contextual sentences before `{START_MARKER}` and after `{END_MARKER}`.
4.  Perform the rewrite according to all parameters and constraints (especially III.6 and III.7).
5.  Output *only* the resulting rewritten string.

**V. Final Output Requirement**

Produce a single block of text representing the rewritten segment, conforming to all specified constraints, suitable for direct programmatic use.

Now start:

Language: {language}

Style: {style}

Goal: {goal}

Approximate Target Length: ~{min_len}-{max_len} characters.
Instructions:

Rewrite ONLY the marked segment.

Adhere to parameters. Preserve core meaning but enhance vividness.

Ensure low similarity to the original marked segment.

Avoid repeating sentences from the surrounding context.

Output ONLY the rewritten text block without ANY comments.
Text:
{text_with_markers}
"""

def validate_rewritten_text(text: str, original: str, orig_len: int, prev_block: str, next_block: str, context: str) -> Tuple[bool, Optional[str]]:
    if not text.strip() and original.strip():
        return False, f"{context}: Пустой текст при непустом оригинале."
    if START_MARKER in text or END_MARKER in text:
        return False, f"{context}: Содержит маркеры."
    if original.strip() and text.strip() and original != text:
        similarity = difflib.SequenceMatcher(None, original, text).ratio()
        if similarity >= SIMILARITY_THRESHOLD:
            return False, f"{context}: Слишком похож на оригинал ({similarity:.2f})."
    if orig_len > 0:
        text_len = count_chars(text)
        min_len = orig_len * MIN_REWRITE_LENGTH_RATIO
        max_len = orig_len * MAX_REWRITE_LENGTH_RATIO + 10
        if text_len > max_len:
            return False, f"{context}: Слишком длинный ({text_len} > {max_len})."
        if text_len < min_len and orig_len > 20:
            return False, f"{context}: Слишком короткий ({text_len} < {min_len})."

    rewritten_sentences = [normalize_sentence(s) for s in split_into_sentences(text)]

    if prev_block:
        prev_sentences = set(normalize_sentence(s) for s in split_into_sentences(prev_block) if has_more_than_n_words(s))
        if any(sent in prev_sentences for sent in rewritten_sentences):
            return False, f"{context}: Содержит предложение из предыдущего блока."

    if next_block:
        next_sentences = set(normalize_sentence(s) for s in split_into_sentences(next_block) if has_more_than_n_words(s))
        if any(sent in next_sentences for sent in rewritten_sentences):
            return False, f"{context}: Содержит предложение из следующего блока."

    return True, None

# Вызов API для переписывания
def call_gemini_rewrite_api(prompt: str, model_name: str, orig_len: int, original: str, prev_block: str, next_block: str) -> Optional[str]:
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logger.error(f"Ошибка инициализации модели '{model_name}': {e}")
        return None

    for attempt in range(MAX_RETRIES):
        context = f"Попытка {attempt + 1}/{MAX_RETRIES}"
        logger.info(f"Вызов API: {context}")
        try:
            response = model.generate_content(prompt, safety_settings=SAFETY_SETTINGS, generation_config=GENERATION_CONFIG)
            text, error, max_tokens = check_api_response(response, context)
            if text is None:
                logger.error(f"{context}: Ошибка API: {error}")
            else:
                is_valid, validation_error = validate_rewritten_text(text, original, orig_len, prev_block, next_block, context)
                if is_valid:
                    logger.info(f"{context}: Успешно переписан блок ({count_chars(text)} симв.).")
                    return text
                logger.warning(f"{context}: {validation_error}")
        except Exception as e:
            logger.error(f"{context}: Ошибка вызова API: {e}")
        if attempt < MAX_RETRIES - 1:
            time.sleep(RETRY_DELAY_SECONDS)
    logger.error("Исчерпаны попытки переписывания.")
    return None

# Сохранение состояния
def save_state(filename: str, data: Dict):
    temp_file = filename + ".tmp"
    try:
        os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
        with open(temp_file, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        os.replace(temp_file, filename)
        logger.info(f"Состояние сохранено: {filename}")
    except Exception as e:
        logger.error(f"Ошибка сохранения состояния: {e}")

# Загрузка состояния
def load_state(filename: str) -> Optional[Dict]:
    if os.path.exists(filename):
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if 'processed_block_index' in state and 'original_blocks_data' in state and 'total_blocks' in state:
                logger.info(f"Состояние загружено: {filename}")
                return state
        except Exception as e:
            logger.warning(f"Ошибка загрузки состояния: {e}")
    return None

# Сохранение промежуточного файла
def save_intermediate(filename: str, content: str, context: str = ""):
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(temp_file, filename)
        logger.info(f"{context}: Промежуточный файл сохранен: {filename}")
    except Exception as e:
        logger.warning(f"{context}: Ошибка сохранения: {e}")

# Основной процесс переписывания
def rewrite_process(params: Dict, progress_callback=None, stop_event=None):
    input_file = params['input_file']
    output_file = params['output_file']
    language = params['language']
    style = params['style']
    goal = params['goal']
    model = params['rewriter_model']
    resume = params['resume']
    save_interval = params['save_interval']

    output_dir = os.path.dirname(output_file) or '.'
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    state_file = os.path.join(output_dir, base_name + STATE_SUFFIX)
    intermediate_file = os.path.join(output_dir, base_name + INTERMEDIATE_SUFFIX)

    logger.info(f"Начало переписывания: {input_file} -> {output_file}")
    logger.info(f"Язык: {language}, Модель: {model}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        if not original_text:
            logger.error("Входной файл пуст.")
            return
    except Exception as e:
        logger.error(f"Ошибка чтения файла: {e}")
        return

    blocks, processed_idx, rewritten_text = [], -1, None
    if resume and (state := load_state(state_file)):
        blocks = state['original_blocks_data']
        processed_idx = state['processed_block_index']
        with open(intermediate_file, 'r', encoding='utf-8') as f:
            rewritten_text = f.read()
        logger.info(f"Возобновление с блока {processed_idx + 2}")
    else:
        rewritten_text = original_text
        blocks = split_into_blocks(original_text, BLOCK_TARGET_CHARS)
        save_intermediate(intermediate_file, rewritten_text, "Инициализация")
        save_state(state_file, {
            'processed_block_index': -1,
            'original_blocks_data': blocks,
            'total_blocks': len(blocks),
            'timestamp': time.time()
        })

    if not blocks:
        logger.error("Не удалось разбить текст на блоки.")
        return

    total_blocks = len(blocks)
    if progress_callback:
        progress_callback(processed_idx + 1, total_blocks)

    for i in range(total_blocks):
        if stop_event and stop_event.is_set():
            logger.warning("Процесс остановлен.")
            break

        block = blocks[i]
        if i <= processed_idx or block['processed']:
            continue
        if block['failed_attempts'] >= MAX_RETRIES:
            logger.warning(f"Блок {i+1} пропущен: превышен лимит попыток.")
            continue

        start, end = block['start_char_index'], block['end_char_index']
        logger.info(f"Обработка блока {i+1}/{total_blocks} [{start}:{end}]")

        block_text = rewritten_text[start:end]
        prev_block = rewritten_text[blocks[i-1]['start_char_index']:blocks[i-1]['end_char_index']] if i > 0 else ""
        next_block = rewritten_text[blocks[i+1]['start_char_index']:blocks[i+1]['end_char_index']] if i < total_blocks - 1 else ""

        prompt = create_rewrite_prompt(language, style, goal, f"{rewritten_text[:start]}{START_MARKER}{block_text}{END_MARKER}{rewritten_text[end:]}", len(block_text))
        new_text = call_gemini_rewrite_api(prompt, model, len(block_text), block_text, prev_block, next_block)

        if new_text:
            delta = len(new_text) - len(block_text)
            rewritten_text = rewritten_text[:start] + new_text + rewritten_text[end:]
            block['end_char_index'] = start + len(new_text)
            block['processed'] = True
            block['failed_attempts'] = 0
            processed_idx = i
            save_intermediate(intermediate_file, rewritten_text, f"Блок {i+1}")
            for j in range(i + 1, total_blocks):
                blocks[j]['start_char_index'] += delta
                blocks[j]['end_char_index'] += delta
        else:
            block['failed_attempts'] += 1
            logger.error(f"Блок {i+1} не переписан.")

        if progress_callback:
            progress_callback(i + 1, total_blocks)

        if save_interval and (i + 1) % save_interval == 0:
            save_state(state_file, {
                'processed_block_index': processed_idx,
                'original_blocks_data': blocks,
                'total_blocks': total_blocks,
                'timestamp': time.time()
            })

    logger.info(f"Переписывание завершено. Обработано: {sum(1 for b in blocks if b['processed'])}/{total_blocks}")
    save_intermediate(output_file, rewritten_text, "Финал")
    save_state(state_file, {
        'processed_block_index': processed_idx,
        'original_blocks_data': blocks,
        'total_blocks': total_blocks,
        'timestamp': time.time()
    })

# GUI-приложение
class BookRewriterApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Book Rewriter")
        master.geometry("950x750")
        self.rewrite_thread = None
        self.stop_event = threading.Event()

        main_frame = ttk.Frame(master, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Настройки")
        settings_frame.pack(pady=10, fill=tk.X)

        ttk.Label(settings_frame, text="Входной файл:").grid(row=0, column=0, sticky=tk.W)
        self.input_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.input_var).grid(row=0, column=1, sticky=tk.EW)
        ttk.Button(settings_frame, text="Обзор", command=self.browse_input).grid(row=0, column=2)

        ttk.Label(settings_frame, text="Выходной файл:").grid(row=1, column=0, sticky=tk.W)
        self.output_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.output_var).grid(row=1, column=1, sticky=tk.EW)
        ttk.Button(settings_frame, text="Обзор", command=self.browse_output).grid(row=1, column=2)

        ttk.Label(settings_frame, text="Язык:").grid(row=2, column=0, sticky=tk.W)
        self.lang_var = tk.StringVar(value="Русский")
        ttk.Combobox(settings_frame, textvariable=self.lang_var, values=["Русский", "English"], state="readonly").grid(row=2, column=1, sticky=tk.W)

        ttk.Label(settings_frame, text="Стиль:").grid(row=3, column=0, sticky=tk.NW)
        self.style_text = tk.Text(settings_frame, height=4, width=50)
        self.style_text.grid(row=3, column=1, columnspan=2, sticky=tk.EW)

        ttk.Label(settings_frame, text="Цель:").grid(row=4, column=0, sticky=tk.NW)
        self.goal_text = tk.Text(settings_frame, height=4, width=50)
        self.goal_text.grid(row=4, column=1, columnspan=2, sticky=tk.EW)

        self.resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Возобновить", variable=self.resume_var).grid(row=5, column=0, sticky=tk.W)

        ttk.Label(settings_frame, text="Модель:").grid(row=6, column=0, sticky=tk.W)
        self.model_var = tk.StringVar(value=REWRITER_MODEL_DEFAULT)
        ttk.Combobox(settings_frame, textvariable=self.model_var, values=list_available_models(), state="readonly").grid(row=6, column=1, sticky=tk.EW)

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X)

        self.start_btn = ttk.Button(control_frame, text="Старт", command=self.start_rewrite)
        self.start_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(control_frame, text="Стоп", command=self.stop_rewrite, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(control_frame, textvariable=self.status_var).pack(side=tk.RIGHT)

        self.progress = ttk.Progressbar(control_frame, mode='determinate')
        self.progress.pack(fill=tk.X, expand=True)

        log_frame = ttk.LabelFrame(main_frame, text="Лог")
        log_frame.pack(fill=tk.BOTH, expand=True)
        self.log_area = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED)
        self.log_area.pack(fill=tk.BOTH, expand=True)

        gui_handler = QueueHandler(log_queue)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(gui_handler)
        self.master.after(100, self.process_log_queue)

    def browse_input(self):
        file = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt")])
        if file:
            self.input_var.set(file)
            if not self.output_var.get():
                self.output_var.set(f"{os.path.splitext(file)[0]}{FINAL_SUFFIX}")

    def browse_output(self):
        file = filedialog.asksaveasfilename(defaultextension=".txt", filetypes=[("Текстовые файлы", "*.txt")])
        if file:
            self.output_var.set(file)

    def process_log_queue(self):
        try:
            while True:
                msg = log_queue.get_nowait()
                self.log_area.configure(state=tk.NORMAL)
                self.log_area.insert(tk.END, msg + '\n')
                self.log_area.configure(state=tk.DISABLED)
                self.log_area.yview(tk.END)
        except queue.Empty:
            pass
        self.master.after(100, self.process_log_queue)

    def update_progress(self, current: int, total: int):
        self.progress['maximum'] = total
        self.progress['value'] = current
        self.status_var.set(f"Обработка: {current}/{total} ({current/total*100:.1f}%)")

    def start_rewrite(self):
        params = {
            'input_file': self.input_var.get(),
            'output_file': self.output_var.get(),
            'language': self.lang_var.get(),
            'style': self.style_text.get("1.0", tk.END).strip(),
            'goal': self.goal_text.get("1.0", tk.END).strip(),
            'rewriter_model': self.model_var.get(),
            'resume': self.resume_var.get(),
            'save_interval': 1
        }
        if not all([params['input_file'], params['output_file'], params['style'], params['goal']]):
            messagebox.showerror("Ошибка", "Заполните все обязательные поля.")
            return
        configure_gemini()
        self.stop_event.clear()
        self.start_btn['state'] = tk.DISABLED
        self.stop_btn['state'] = tk.NORMAL
        self.rewrite_thread = threading.Thread(target=rewrite_process, args=(params, self.update_progress, self.stop_event), daemon=True)
        self.rewrite_thread.start()

    def stop_rewrite(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
            self.stop_event.set()
            self.stop_btn['state'] = tk.DISABLED

    def on_closing(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
            if messagebox.askyesno("Выход", "Остановить процесс и выйти?"):
                self.stop_rewrite()
                self.master.after(200, self.master.destroy)
        else:
            self.master.destroy()

if __name__ == "__main__":
    log_file = "rewriter.log"
    file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
    logger.addHandler(file_handler)
    logger.setLevel(logging.INFO)

    root = tk.Tk()
    sv_ttk.set_theme("dark")
    app = BookRewriterApp(root)
    root.mainloop()