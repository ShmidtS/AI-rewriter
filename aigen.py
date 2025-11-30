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
from google.generativeai.types import GenerationConfig, HarmCategory, HarmBlockThreshold

# --- Конфигурация и константы (без изменений) ---
log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

try:
    load_dotenv()
    GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GOOGLE_API_KEY:
        logger.warning("Переменная GOOGLE_API_KEY не установлена.")
except Exception as e:
    logger.error(f"Ошибка загрузки .env: {e}")
    GOOGLE_API_KEY = None

REWRITER_MODEL_DEFAULT = "gemini-2.5-flash" # Обновляем модель по умолчанию
START_MARKER = "<|~START_REWRITE~|>"
END_MARKER = "<|~END_REWRITE~|>"
BLOCK_TARGET_CHARS = 15000
MIN_REWRITE_LENGTH_RATIO = 0.40
MAX_REWRITE_LENGTH_RATIO = 1.6
SIMILARITY_THRESHOLD = 0.95
OUTPUT_TOKEN_LIMIT = 65536
MIN_BLOCK_LEN_FACTOR = 0.5
MAX_BLOCK_LEN_FACTOR = 1.5
SEARCH_RADIUS_FACTOR = 0.1
SPLIT_PRIORITY = ['. ', '! ', '? ']
# Улучшенные приоритеты разбиения с учетом абзацев и диалогов
SPLIT_PRIORITY_ENHANCED = [
    '\n\n',      # Двойной перенос строки (абзац)
    '\n',        # Одиночный перенос строки
    '. ', '! ', '? ',  # Конец предложения
    '; ',        # Полуточка
    ', ',        # Запятая (последний вариант)
]
MAX_RETRIES = 20
RETRY_DELAY_SECONDS = 0
# Адаптивные параметры генерации
ADAPTIVE_TEMPERATURE_BASE = 1.0
ADAPTIVE_TEMPERATURE_MIN = 0.5
ADAPTIVE_TEMPERATURE_MAX = 1.2

STATE_SUFFIX = "_rewrite_state.json"
INTERMEDIATE_SUFFIX = "_intermediate.txt"
FINAL_SUFFIX = "_final_rewritten.txt"

SYSTEM_INSTRUCTION_BASE = """
---

### System Prompt v3.1 – Literary Text Rewriter Agent

**IMPORTANT CONTEXT**: You are a professional literary editor working on book rewriting and adaptation projects. All content you process is from published or unpublished literary works, fiction or non-fiction, and your task is purely editorial - to improve writing quality, clarity, and reader engagement. This is legitimate literary work.

**I. Core Directive**

You are a specialized AI Text Rewriter Agent with advanced capabilities for semantic transformation of literary content. Your sole function is to rewrite a specific segment of text provided within a larger context, strictly adhering to the parameters and constraints given in each request. You operate as a component within a larger automated workflow that processes text in blocks. Your output must be precise, conformant, and directly usable by this workflow.

**II. Task Definition**

1. **Identify Target Segment:** The input text will contain a segment clearly marked by `{START_MARKER}` and `{END_MARKER}`. Your task is to rewrite *only* the text located *between* these two markers. **ONLY the text between the markers should be rewritten - nothing else.**

2. **Context Awareness:** Context may be provided separately in a "Context for coherence" section. This context is provided solely for understanding narrative flow and ensuring smooth transitions. **Do NOT modify, rewrite, or include this context in your output.** Use it only to understand how your rewritten segment should connect with surrounding text.

3. **Rewrite Parameters:** You will be given specific parameters for each rewriting task:

   - **Language:** The target language for the rewritten text (e.g., "Русский", "English").
   - **Style:** A description of the desired writing style (e.g., "Formal academic", "Engaging narrative", "Simple and direct").
   - **Goal:** The specific objective of the rewrite (e.g., "Improve clarity", "Simplify vocabulary", "Adapt for a younger audience", "Increase detail").
   - **Approximate Target Length:** A suggested character length range for the rewritten segment (e.g., "~{min_len}-{max_len} characters").

**III. Enhanced Operational Constraints & Instructions**

1. **Focus Exclusively:** Rewrite *only* the text content found strictly between `{START_MARKER}` and `{END_MARKER}`. **Your output must not, under any circumstances, begin before the conceptual start of the marked segment or end after the conceptual end of the marked segment.**

2. **Context Awareness & Cohesion:** The text *before* the `{START_MARKER}` and *after* the `{END_MARKER}` is provided solely for context. **Do NOT modify, rewrite, or include ANY PART of this contextual text in your output.** However, analyze the context to:
   - Understand the narrative flow and tone
   - Ensure your rewritten segment creates smooth transitions
   - Maintain consistency in character names, terminology, and style
   - Avoid repeating sentences or phrases from the context

3. **Parameter Adherence:** Strictly follow the specified `Language`, `Style`, and `Goal` parameters. Apply them consistently throughout the rewritten segment.

4. **Meaning Preservation:** Preserve the core meaning, information, and narrative intent of the original segment unless the `Goal` explicitly dictates otherwise (e.g., simplification might remove nuance). Maintain all key facts, events, and character actions.

5. **Semantic Transformation:** The rewrite should be a genuine semantic transformation, not just a paraphrase. Consider:
   - Using different sentence structures
   - Employing varied vocabulary while maintaining meaning
   - Reorganizing information flow when appropriate
   - Adding depth and nuance where the Goal permits

6. **Length Guideline:** Aim for a character count within the suggested `Approximate Target Length` range. Slight deviations are acceptable if they improve quality.

7. **CRITICAL - Avoid High Similarity:** The rewritten text *must be substantially different* from the original text segment. Direct copying or minor paraphrasing that results in high textual similarity (e.g., >90-95% similar) is **unacceptable**. The rewrite should be a genuine transformation that maintains meaning but uses different expression.

8. **CRITICAL - Avoid Context Sentence Repetition:** The rewritten text *must not* contain full sentences that are identical (or near-identical after normalization like lowercasing and punctuation removal) to full sentences present in the immediate context provided *before* the `{START_MARKER}` or *after* the `{END_MARKER}`. **Pay extremely close attention to the sentences immediately preceding `{START_MARKER}` and immediately following `{END_MARKER}`. Do not repeat them in your output.**

9. **Lexical Diversity:** Use varied vocabulary and expressions. Avoid excessive repetition of the same words or phrases from the original. Introduce synonyms and alternative phrasings while maintaining clarity.

10. **Output Format:**

   - Generate *only* the rewritten text corresponding to the segment between the markers.
   - **Your output must begin with the rewritten version of the content that immediately follows `{START_MARKER}`.**
   - **Your output must end with the rewritten version of the content that immediately precedes `{END_MARKER}`.**
   - **Do NOT include the `{START_MARKER}` or `{END_MARKER}` in your output.**
   - **Do NOT include any of the surrounding context in your output.**
   - **Do NOT add any explanations, apologies, or introductory/concluding remarks.** Your output must be *only* the rewritten string, ready for direct substitution into the larger text. **If the original segment is empty, output an empty string.**

**IV. Execution Logic**

1. Parse the `Parameters` and `Instructions`.

2. Analyze the context before `{START_MARKER}` and after `{END_MARKER}` to understand narrative flow, tone, and key elements.

3. Isolate the `[Original text segment to be rewritten.]` from the `Text:` section.

4. Perform a semantic analysis of the original segment to identify:
   - Core meaning and key information
   - Narrative elements (characters, actions, descriptions)
   - Style characteristics
   - Structural patterns

5. Rewrite the segment according to all parameters and constraints, ensuring:
   - Semantic transformation (different expression, same meaning)
   - Lexical diversity
   - Smooth transitions with context
   - Adherence to Style and Goal parameters

6. Output *only* the resulting rewritten string.

**V. Final Output Requirement**

Produce a single block of text representing the rewritten segment, conforming to all specified constraints, suitable for direct programmatic use. The output should demonstrate clear semantic transformation while maintaining narrative coherence and contextual fit.

---
"""

class BlockInfo(TypedDict):
    block_index: int
    start_char_index: int
    end_char_index: int
    original_char_length: int
    processed: bool
    failed_attempts: int

class QueueHandler(logging.Handler):
    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))

def configure_gemini():
    if not GOOGLE_API_KEY:
        raise ValueError("API-ключ Gemini не установлен.")
    try:
        genai.configure(api_key=GOOGLE_API_KEY)
        logger.info("Gemini API успешно сконфигурирован.")
        return True
    except Exception as e:
        logger.error(f"Ошибка конфигурации Gemini API: {e}")
        raise ValueError(f"Ошибка конфигурации: {e}")

def list_available_models():
    """Получает и отображает ВСЕ доступные модели через API без фильтрации."""
    try:
        # Получаем полный список моделей через API без фильтрации
        models_list = genai.list_models()
        api_models = []

        for model in models_list:
            if 'generateContent' in model.supported_generation_methods:
                model_name = model.name.split('/')[-1]
                api_models.append(model_name)
                logger.debug(f"Найдена модель: {model_name}")

        all_models = sorted(api_models)

        if all_models:
            logger.info(f"Загружено {len(all_models)} моделей через API.")
            # Выводим полный список в лог для отладки
            logger.debug("Все доступные модели:\n" + "\n".join(all_models))
            return all_models
        else:
            logger.error("Не найдено доступных моделей через API.")
            return []

    except Exception as e:
        logger.error(f"Ошибка получения списка моделей через API: {e}")
        return []


def count_chars(text: str) -> int:
    return len(text) if isinstance(text, str) else 0

def split_into_sentences(text: str) -> List[str]:
    """Разбивает текст на предложения по '. ', '! ', '? '."""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    return [s.strip() for s in sentences if s.strip()]

def has_more_than_n_words(sentence: str, n: int = 10) -> bool:
    """Проверяет, содержит ли предложение больше n слов."""
    words = sentence.split()
    return len(words) > n

def normalize_sentence(sentence: str) -> str:
    """Приводит предложение к нижнему регистру и убирает конечные знаки препинания."""
    sentence = sentence.lower()
    sentence = sentence.rstrip(string.punctuation)
    # Убираем множественные пробелы
    sentence = re.sub(r'\s+', ' ', sentence).strip()
    return sentence

def calculate_text_quality_metrics(original: str, rewritten: str) -> Dict[str, float]:
    """
    Вычисляет метрики качества переписанного текста.
    Возвращает словарь с метриками: similarity, length_ratio, diversity.
    """
    metrics = {}
    
    # Метрика схожести (чем меньше, тем лучше для переписывания)
    if original.strip() and rewritten.strip():
        metrics['similarity'] = difflib.SequenceMatcher(None, original, rewritten).ratio()
    else:
        metrics['similarity'] = 1.0 if not rewritten.strip() else 0.0
    
    # Соотношение длин
    orig_len = len(original)
    rew_len = len(rewritten)
    if orig_len > 0:
        metrics['length_ratio'] = rew_len / orig_len
    else:
        metrics['length_ratio'] = 1.0
    
    # Метрика разнообразия (уникальные слова)
    orig_words = set(re.findall(r'\b\w+\b', original.lower()))
    rew_words = set(re.findall(r'\b\w+\b', rewritten.lower()))
    if orig_words:
        metrics['diversity'] = len(rew_words - orig_words) / max(len(orig_words), 1)
    else:
        metrics['diversity'] = 0.0
    
    return metrics

def check_api_response(response: genai.types.GenerateContentResponse, context: str) -> Tuple[Optional[str], Optional[str], bool, bool]:
    """
    Проверяет ответ API и возвращает: (text, error, max_tokens, is_blocked)
    is_blocked=True означает блокировку контента (PROHIBITED_CONTENT или SAFETY)
    """
    text, error, max_tokens, is_blocked = None, None, False, False

    if response.prompt_feedback.block_reason:
        is_blocked = True
        block_reason = response.prompt_feedback.block_reason.name
        error = f"{context}: Промпт заблокирован ({block_reason})."
        logger.warning(f"{error} Попытка fallback.")
        return None, error, False, is_blocked

    if not response.candidates:
        try:
            text = response.text.strip()
            if text:
                logger.warning(f"{context}: Нет кандидатов, используется fallback текст ({count_chars(text)} симв.).")
                return text, None, False, False
            error = f"{context}: Нет кандидатов и пустой fallback текст."
            logger.error(error)
            return None, error, False, False
        except Exception as e:
            error = f"{context}: Ошибка получения текста: {e}."
            logger.error(error)
            return None, error, False, False

    candidate = response.candidates[0]
    finish_reason = getattr(candidate, 'finish_reason', None)
    finish_value = int(finish_reason.value) if finish_reason else 0

    if finish_value == 3:
        is_blocked = True
        error = f"{context}: Ответ заблокирован фильтром безопасности."
        logger.warning(f"{error} Попытка fallback.")
        return None, error, False, is_blocked
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
        return text, None, max_tokens, False
    except Exception as e:
        error = f"{context}: Ошибка извлечения текста: {e}."
        logger.error(error)
        return None, error, False, False

def find_split_point(text: str, start: int, target_end: int, min_len: int, max_len: int) -> int:
    """
    Улучшенная функция поиска точки разбиения с учетом семантических границ.
    Приоритет: абзацы > предложения > знаки препинания > пробелы.
    """
    text_len = len(text)
    ideal_end = min(text_len, max(start + min_len, min(target_end, start + max_len)))

    if ideal_end >= text_len:
        return text_len

    # Увеличиваем радиус поиска для лучшего качества разбиения
    radius = int((target_end - start) * max(SEARCH_RADIUS_FACTOR, 0.15))
    search_start = max(start + min_len, ideal_end - radius)
    search_end = min(text_len, ideal_end + radius, start + max_len)

    best_point = -1
    min_dist = float('inf')
    best_priority = float('inf')  # Чем меньше, тем выше приоритет

    # Используем улучшенные приоритеты разбиения
    for priority, seq in enumerate(SPLIT_PRIORITY_ENHANCED):
        pos = search_start
        while pos < search_end:
            try:
                idx = text.index(seq, pos, search_end) + len(seq)
                if idx > start + min_len:
                    dist = abs(idx - ideal_end)
                    # Учитываем и приоритет, и расстояние
                    score = dist + (priority * 1000)  # Приоритет важнее расстояния
                    if score < (min_dist + (best_priority * 1000)):
                        min_dist = dist
                        best_point = idx
                        best_priority = priority
                pos = idx + 1
            except ValueError:
                break
        # Если нашли хорошую точку с высоким приоритетом (абзац или предложение), используем её
        if best_point != -1 and best_priority <= 2:  # Абзац или предложение
            return best_point

    # Если не нашли хорошую точку, ищем пробел
    if best_point == -1:
        try:
            last_space = text.rindex(' ', search_start, min(ideal_end, start + max_len))
            if last_space > start:
                return last_space + 1
        except ValueError:
            pass

    # Если нашли точку, возвращаем её, иначе возвращаем идеальную
    return best_point if best_point != -1 else min(ideal_end, start + max_len)


def split_into_blocks(text: str, target_size: int) -> Optional[List[BlockInfo]]:
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

def create_rewrite_prompt(language: str, style: str, goal: str, block_text: str, prev_context: str, next_context: str, original_len: int) -> str:
    """
    Создает улучшенный промпт для переписывания с литературным контекстом.
    Между маркерами передается ТОЛЬКО текст блока, контекст указывается отдельно.
    """
    min_len = int(original_len * MIN_REWRITE_LENGTH_RATIO)
    max_len = int(original_len * MAX_REWRITE_LENGTH_RATIO)

    # Формируем текст с маркерами - только блок между маркерами
    text_with_markers = f"{START_MARKER}{block_text}{END_MARKER}"

    # Формируем контекст для понимания связности
    context_section = ""
    if prev_context or next_context:
        context_section = "\n\nContext for coherence (DO NOT include in output):\n"
        if prev_context:
            context_section += f"Previous text: ...{prev_context}\n"
        if next_context:
            context_section += f"Next text: {next_context}..."

    return f"""
LITERARY EDITORIAL TASK: Professional book rewriting and adaptation

This is a legitimate literary editing project. You are rewriting a segment from a book/manuscript to improve its quality, readability, and engagement while preserving the original narrative and meaning.

Task: Rewrite the marked text segment

Language: {language}
Style: {style}
Goal: {goal}
Target Length: ~{min_len}-{max_len} characters

CRITICAL REQUIREMENTS:
1. Rewrite ONLY the text between {START_MARKER} and {END_MARKER}
2. Preserve core meaning and all key information
3. Use semantic transformation - different expression, same meaning
4. Ensure lexical diversity - avoid repeating words from original
5. Maintain smooth transitions with surrounding context (if provided)
6. Do NOT repeat any sentences from the context provided below
7. Output ONLY the rewritten segment - no markers, no explanations

Text to rewrite:
{text_with_markers}{context_section}
"""

def create_fallback_prompt(language: str, style: str, goal: str, block_text: str, prev_context: str, next_context: str, original_len: int) -> str:
    """
    Создает альтернативный промпт для случаев блокировки контента.
    Использует более нейтральную формулировку и акцент на образовательных целях.
    """
    min_len = int(original_len * MIN_REWRITE_LENGTH_RATIO)
    max_len = int(original_len * MAX_REWRITE_LENGTH_RATIO)

    # Формируем текст с маркерами
    text_with_markers = f"{START_MARKER}{block_text}{END_MARKER}"

    # Формируем контекст
    context_section = ""
    if prev_context or next_context:
        context_section = "\n\nNarrative context:\n"
        if prev_context:
            context_section += f"Previous: ...{prev_context}\n"
        if next_context:
            context_section += f"Following: {next_context}..."

    return f"""
EDUCATIONAL WRITING EXERCISE: Text paraphrasing and stylistic improvement

You are working on an educational exercise to improve writing skills. The task is to paraphrase and enhance the text segment between the markers while maintaining the original meaning.

Target language: {language}
Writing style to apply: {style}
Improvement objective: {goal}
Length guideline: approximately {min_len}-{max_len} characters

Instructions:
- Paraphrase only the text between {START_MARKER} and {END_MARKER}
- Keep the same meaning and narrative elements
- Use different words and sentence structures
- Ensure natural flow with the surrounding context
- Return only the paraphrased text without markers

Source text:
{text_with_markers}{context_section}
"""

# --- Улучшенная функция валидации ---
def validate_rewritten_text(text: str, original: str, orig_len: int, prev_block: str, next_block: str, context: str) -> Tuple[bool, Optional[str], Optional[Dict[str, float]]]:
    """
    Улучшенная валидация переписанного текста с расширенными метриками качества.
    Маркеры START_MARKER и END_MARKER удаляются из 'text' перед проверкой контента.
    Возвращает: (is_valid, error_message, quality_metrics)
    """
    # Проверяем, не пустой ли текст при непустом оригинале (до удаления маркеров)
    if not text.strip() and original.strip():
        return False, f"{context}: Пустой текст при непустом оригинале.", None

    # Удаляем маркеры из текста *перед* дальнейшими проверками контента
    text_cleaned = text.replace(START_MARKER, "").replace(END_MARKER, "")

    # Вычисляем метрики качества
    quality_metrics = calculate_text_quality_metrics(original, text_cleaned)

    # --- Проверки выполняются на ОЧИЩЕННОМ тексте ('text_cleaned') ---

    # Улучшенная проверка на схожесть с оригиналом
    if original.strip() and text_cleaned.strip() and original != text_cleaned:
        similarity = quality_metrics['similarity']
        if similarity >= SIMILARITY_THRESHOLD:
            return False, f"{context}: Слишком похож на оригинал ({similarity:.3f}) после очистки маркеров.", quality_metrics
        
        # Дополнительная проверка: если схожесть очень высокая (>0.85), но не критическая, предупреждаем
        if similarity > 0.85:
            logger.warning(f"{context}: Высокая схожесть ({similarity:.3f}), но в пределах допустимого.")

    # Проверка длины с более гибкими границами
    if orig_len > 0:
        text_len_cleaned = count_chars(text_cleaned)
        min_len = orig_len * MIN_REWRITE_LENGTH_RATIO
        max_len = orig_len * MAX_REWRITE_LENGTH_RATIO
        if text_len_cleaned > max_len:
            return False, f"{context}: Слишком длинный ({text_len_cleaned} > {int(max_len)}) после очистки маркеров.", quality_metrics
        # Проверяем на слишком короткий текст, только если оригинал не был совсем коротким
        if text_len_cleaned < min_len and orig_len > 20:
            return False, f"{context}: Слишком короткий ({text_len_cleaned} < {int(min_len)}) после очистки маркеров.", quality_metrics

    # Улучшенная проверка на повторение предложений из контекста
    rewritten_sentences = [normalize_sentence(s) for s in split_into_sentences(text_cleaned)]

    if prev_block:
        prev_sentences = set(normalize_sentence(s) for s in split_into_sentences(prev_block) if has_more_than_n_words(s, 8))  # Уменьшили порог для более строгой проверки
        repeated_prev = [sent for sent in rewritten_sentences if sent in prev_sentences]
        if repeated_prev:
            logger.debug(f"Повтор предложений из предыдущего блока: {repeated_prev[:2]}...")  # Показываем только первые 2
            return False, f"{context}: Содержит предложение(я) из предыдущего блока.", quality_metrics

    if next_block:
        next_sentences = set(normalize_sentence(s) for s in split_into_sentences(next_block) if has_more_than_n_words(s, 8))
        repeated_next = [sent for sent in rewritten_sentences if sent in next_sentences]
        if repeated_next:
            logger.debug(f"Повтор предложений из следующего блока: {repeated_next[:2]}...")
            return False, f"{context}: Содержит предложение(я) из следующего блока.", quality_metrics

    # Дополнительная проверка: минимальное разнообразие
    if quality_metrics['diversity'] < 0.1 and orig_len > 50:
        logger.warning(f"{context}: Низкое разнообразие лексики ({quality_metrics['diversity']:.3f})")

    # Если все проверки пройдены
    return True, None, quality_metrics
# --- Конец улучшенной функции валидации ---


def calculate_adaptive_temperature(failed_attempts: int, quality_metrics: Optional[Dict[str, float]] = None) -> float:
    """
    Вычисляет адаптивную температуру на основе количества неудачных попыток и метрик качества.
    Больше попыток -> выше температура для большего разнообразия.
    """
    base_temp = ADAPTIVE_TEMPERATURE_BASE
    
    # Увеличиваем температуру с каждой неудачной попыткой
    attempt_factor = min(failed_attempts * 0.05, 0.2)  # Максимум +0.2
    
    # Если метрики показывают высокую схожесть, увеличиваем температуру
    similarity_factor = 0.0
    if quality_metrics and quality_metrics.get('similarity', 0) > 0.85:
        similarity_factor = (quality_metrics['similarity'] - 0.85) * 0.4  # До +0.06
    
    # Если низкое разнообразие, увеличиваем температуру
    diversity_factor = 0.0
    if quality_metrics and quality_metrics.get('diversity', 1) < 0.15:
        diversity_factor = (0.15 - quality_metrics['diversity']) * 0.3  # До +0.045
    
    final_temp = base_temp + attempt_factor + similarity_factor + diversity_factor
    return max(ADAPTIVE_TEMPERATURE_MIN, min(ADAPTIVE_TEMPERATURE_MAX, final_temp))

def call_gemini_rewrite_api(
    system_instruction: str,
    user_content: str,
    model_name: str,
    orig_len: int,
    original: str,
    prev_block: str,
    next_block: str,
    stop_event: threading.Event,
    failed_attempts: int = 0,
    previous_quality_metrics: Optional[Dict[str, float]] = None,
    language: str = "Русский",
    style: str = "",
    goal: str = ""
) -> Optional[str]:
    try:
        # Создаем модель без проверок - все модели включая с 'thinking' должны работать
        model = genai.GenerativeModel(
            model_name,
            system_instruction=system_instruction
        )
        logger.info(f"Инициализирована модель '{model_name}' с системной инструкцией.")
    except Exception as e:
        logger.error(f"Ошибка инициализации модели '{model_name}': {e}")
        return None

    # Адаптивная температура на основе предыдущих попыток
    adaptive_temp = calculate_adaptive_temperature(failed_attempts, previous_quality_metrics)
    
    generation_config = GenerationConfig(
        temperature=adaptive_temp, 
        top_p=0.95, 
        top_k=32, 
        max_output_tokens=OUTPUT_TOKEN_LIMIT
    )
    
    if failed_attempts > 0:
        logger.debug(f"Используется адаптивная температура: {adaptive_temp:.2f} (попыток: {failed_attempts})")
    safety_settings = {
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
    }

    # Обновляем температуру перед каждой попыткой на основе текущих failed_attempts
    current_failed_attempts = failed_attempts
    last_quality_metrics = previous_quality_metrics
    use_fallback = False
    fallback_user_content = None

    for attempt in range(MAX_RETRIES):
        # Проверяем флаг остановки перед каждой попыткой
        if stop_event and stop_event.is_set():
            logger.warning("Получен сигнал остановки во время попыток вызова API.")
            return None

        # Пересчитываем температуру для каждой попытки
        if attempt > 0:
            adaptive_temp = calculate_adaptive_temperature(current_failed_attempts + attempt, last_quality_metrics)
            generation_config = GenerationConfig(
                temperature=adaptive_temp,
                top_p=0.95,
                top_k=32,
                max_output_tokens=OUTPUT_TOKEN_LIMIT
            )
            logger.debug(f"Попытка {attempt + 1}: Адаптивная температура {adaptive_temp:.2f}")

        context = f"Попытка {attempt + 1}/{MAX_RETRIES}"

        # Если используем fallback, создаем альтернативный промпт один раз
        if use_fallback and fallback_user_content is None and language and style and goal:
            logger.info(f"{context}: Использую fallback промпт для обхода блокировки")
            fallback_user_content = create_fallback_prompt(language, style, goal, original, prev_block, next_block, orig_len)

        # Выбираем какой контент отправлять
        current_content = fallback_user_content if use_fallback and fallback_user_content else user_content

        logger.info(f"Вызов API: {context}{' (fallback)' if use_fallback else ''}")
        try:
            response = model.generate_content(
                contents= [{'role': 'user', 'parts': [{'text': current_content}]}],
                generation_config=generation_config,
                safety_settings=safety_settings
            )
            text, error, max_tokens, is_blocked = check_api_response(response, context)

            # Если контент заблокирован, активируем fallback для следующей попытки
            if is_blocked and not use_fallback:
                logger.warning(f"{context}: Контент заблокирован, следующая попытка будет использовать fallback")
                use_fallback = True
                continue

            if text is None:
                logger.error(f"{context}: Ошибка API: {error}")
                # Если даже fallback заблокирован, возможно блок слишком проблематичный
                if is_blocked and use_fallback:
                    logger.error(f"{context}: Fallback промпт также заблокирован. Пропуск блока.")
                    return None
            else:
                # Валидация теперь происходит с учетом удаления маркеров внутри validate_rewritten_text
                is_valid, validation_error, quality_metrics = validate_rewritten_text(
                    text, original, orig_len, prev_block, next_block, context
                )
                if is_valid:
                    # Возвращаем текст *после* удаления маркеров, так как он прошел валидацию
                    text_cleaned = text.replace(START_MARKER, "").replace(END_MARKER, "")
                    metrics_str = ""
                    if quality_metrics:
                        metrics_str = f" (схожесть: {quality_metrics['similarity']:.3f}, разнообразие: {quality_metrics['diversity']:.3f})"
                    logger.info(f"{context}: Успешно переписан и валидирован блок ({count_chars(text_cleaned)} симв. после очистки){metrics_str}.")
                    return text_cleaned # Возвращаем очищенный текст
                else:
                    logger.warning(f"{context}: {validation_error}")
                    # Сохраняем метрики для следующей попытки (адаптивная температура)
                    if quality_metrics:
                        last_quality_metrics = quality_metrics
        except Exception as e:
            error_msg = str(e)
            logger.error(f"{context}: Ошибка вызова API: {error_msg}")

            # Проверяем на ошибку квоты (429) и извлекаем время ожидания
            if "429" in error_msg and "retry_delay" in error_msg:
                try:
                    # Извлекаем время ожидания из сообщения об ошибке
                    import re
                    retry_match = re.search(r'retry_delay\s*{\s*seconds:\s*(\d+)', error_msg)
                    if retry_match:
                        retry_seconds = int(retry_match.group(1))
                        logger.warning(f"{context}: Достигнут лимит API. Ожидание {retry_seconds} секунд...")

                        # Проверяем остановку во время ожидания
                        for wait_second in range(retry_seconds):
                            if stop_event and stop_event.is_set():
                                logger.warning("Получен сигнал остановки во время ожидания лимита API.")
                                return None
                            time.sleep(1)

                        continue  # Пробуем снова после ожидания
                except Exception as parse_error:
                    logger.error(f"{context}: Не удалось извлечь время ожидания из ошибки: {parse_error}")

            # Если это не ошибка квоты или не удалось извлечь время, используем стандартную задержку
            if attempt < MAX_RETRIES - 1:
                # Проверяем остановку во время стандартной задержки
                for wait_second in range(RETRY_DELAY_SECONDS):
                    if stop_event and stop_event.is_set():
                        logger.warning("Получен сигнал остановки во время стандартной задержки.")
                        return None
                    time.sleep(1)

        # Если это не ошибка квоты, используем стандартную задержку между попытками
        if attempt < MAX_RETRIES - 1:
            for wait_second in range(RETRY_DELAY_SECONDS):
                if stop_event and stop_event.is_set():
                    logger.warning("Получен сигнал остановки во время задержки между попытками.")
                    return None
                time.sleep(1)

    logger.error("Исчерпаны попытки переписывания.")
    return None

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


def save_intermediate(filename: str, content: str, context: str = ""):
    temp_file = filename + ".tmp"
    try:
        with open(temp_file, 'w', encoding='utf-8') as f:
            f.write(content)
        os.replace(temp_file, filename)
        logger.info(f"{context}: Промежуточный файл сохранен: {filename}")
    except Exception as e:
        logger.warning(f"{context}: Ошибка сохранения: {e}")


def rewrite_process(params: Dict, progress_callback=None, stop_event=None):
    input_file = params['input_file']
    output_file = params['output_file']
    language = params['language']
    style = params['style']
    goal = params['goal']
    model_name = params['rewriter_model']
    resume = params['resume']
    save_interval = params['save_interval']

    output_dir = os.path.dirname(output_file) or '.'
    base_name = os.path.splitext(os.path.basename(output_file))[0]
    state_file = os.path.join(output_dir, base_name + STATE_SUFFIX)
    intermediate_file = os.path.join(output_dir, base_name + INTERMEDIATE_SUFFIX)

    logger.info(f"Начало переписывания: {input_file} -> {output_file}")
    logger.info(f"Язык: {language}, Стиль: {style[:50]}..., Цель: {goal[:50]}..., Модель: {model_name}")
    logger.info(f"Возобновление: {resume}, Интервал сохранения: {save_interval}")

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_text = f.read()
        if not original_text.strip():
            logger.error("Входной файл пуст или содержит только пробельные символы.")
            if progress_callback:
                progress_callback(0, 1)
            return
    except FileNotFoundError:
         logger.error(f"Входной файл не найден: {input_file}")
         if progress_callback:
             progress_callback(0, 1)
         return
    except Exception as e:
        logger.error(f"Ошибка чтения входного файла {input_file}: {e}", exc_info=True)
        if progress_callback:
            progress_callback(0, 1)
        return

    original_text_len = count_chars(original_text)
    logger.info(f"Длина исходного текста: {original_text_len} символов.")

    blocks: Optional[List[BlockInfo]] = None
    processed_idx = -1
    rewritten_text: Optional[str] = None
    state_loaded = False

    if resume:
        state = load_state(state_file)
        if state and 'original_blocks_data' in state:
            try:
                with open(intermediate_file, 'r', encoding='utf-8') as f:
                    rewritten_text = f.read()
                blocks = state['original_blocks_data']
                processed_idx = state.get('processed_block_index', -1)
                state_loaded = True
                logger.info(f"Состояние и промежуточный текст ({count_chars(rewritten_text)} симв.) загружены. Возобновление с блока {processed_idx + 2}.")
            except FileNotFoundError:
                logger.warning(f"Файл состояния найден, но промежуточный файл {intermediate_file} отсутствует. Начинаем заново.")
            except Exception as e:
                logger.warning(f"Ошибка чтения промежуточного файла {intermediate_file}: {e}. Начинаем заново.")

    if not state_loaded:
        logger.info("Инициализация нового процесса или перезапуск после ошибки загрузки.")
        rewritten_text = original_text
        blocks = split_into_blocks(original_text, BLOCK_TARGET_CHARS)
        if blocks:
            save_intermediate(intermediate_file, rewritten_text, "Инициализация")
            save_state(state_file, {
                'processed_block_index': -1,
                'original_blocks_data': blocks,
                'total_blocks': len(blocks),
                'timestamp': time.time()
            })
            processed_idx = -1
        else:
            logger.error("Не удалось разбить текст на блоки при инициализации.")
            if progress_callback:
                progress_callback(0, 1)
            return

    if not blocks or rewritten_text is None:
        logger.error("Критическая ошибка: отсутствуют блоки или текст для обработки.")
        if progress_callback:
            progress_callback(0, 1)
        return

    total_blocks = len(blocks)
    logger.info(f"Всего блоков для обработки: {total_blocks}")
    if progress_callback:
        progress_callback(processed_idx + 1, total_blocks)

    current_system_instruction = SYSTEM_INSTRUCTION_BASE.format(START_MARKER=START_MARKER, END_MARKER=END_MARKER, min_len="{min_len}", max_len="{max_len}") # Подставим маркеры

    for i in range(total_blocks):
        if stop_event and stop_event.is_set():
            logger.warning("Процесс остановлен пользователем.")
            break

        try:
            block = blocks[i]
        except IndexError:
            logger.error(f"Ошибка: Попытка доступа к блоку с индексом {i}, но всего блоков {total_blocks}.")
            break

        if i <= processed_idx:
            logger.debug(f"Блок {i+1}/{total_blocks} уже обработан, пропуск.")
            continue
        if block.get('processed', False):
            logger.debug(f"Блок {i+1}/{total_blocks} помечен как обработанный, пропуск.")
            continue
        if block.get('failed_attempts', 0) >= MAX_RETRIES:
            logger.warning(f"Блок {i+1}/{total_blocks} пропущен: превышен лимит ({MAX_RETRIES}) неудачных попыток.")
            continue

        start = block['start_char_index']
        end = block['end_char_index']
        current_rewritten_text_len = count_chars(rewritten_text)

        if not (0 <= start <= end <= current_rewritten_text_len):
             logger.error(f"Ошибка: Некорректные границы для блока {i+1}: start={start}, end={end}, text_len={current_rewritten_text_len}. Прерывание.")
             save_state(state_file, {
                 'processed_block_index': processed_idx,
                 'original_blocks_data': blocks,
                 'total_blocks': total_blocks,
                 'timestamp': time.time()
             })
             break

        logger.info(f"Обработка блока {i+1}/{total_blocks} [{start}:{end}] (Длина: {end-start} симв.)")

        block_text = rewritten_text[start:end]
        original_block_length = block.get('original_char_length', len(block_text))

        # Улучшенное получение контекста (предыдущий и следующий блоки)
        # Важно: Контекст берем из *текущего* состояния rewritten_text *до* модификации текущего блока
        # Используем расширенный контекст для лучшей связности (последние 2-3 предложения предыдущего блока)
        prev_block_text = ""
        if i > 0:
            prev_block_info = blocks[i-1]
            prev_start = prev_block_info['start_char_index']
            prev_end = prev_block_info['end_char_index']
            if 0 <= prev_start <= prev_end <= current_rewritten_text_len:
                full_prev_text = rewritten_text[prev_start:prev_end]
                # Берем последние 2-3 предложения для контекста (достаточно для связности)
                # prev_sentences = split_into_sentences(full_prev_text)
                # if len(prev_sentences) > 3:
                #     prev_block_text = ' '.join(prev_sentences[-3:])  # Последние 3 предложения
                # else:
                prev_block_text = full_prev_text  # Если предложений мало, берем весь блок
            else:
                 logger.warning(f"Некорректные границы для предыдущего блока {i}, используем пустой контекст.")

        next_block_text = ""
        if i < total_blocks - 1:
            next_block_info = blocks[i+1]
            next_start = next_block_info['start_char_index']
            next_end = next_block_info['end_char_index']
            safe_next_end = min(next_end, current_rewritten_text_len)
            if 0 <= next_start <= safe_next_end <= current_rewritten_text_len:
                full_next_text = rewritten_text[next_start:safe_next_end]
                # Берем первые 2-3 предложения для контекста
                # next_sentences = split_into_sentences(full_next_text)
                # if len(next_sentences) > 3:
                #     next_block_text = ' '.join(next_sentences[:3])  # Первые 3 предложения
                # else:
                next_block_text = full_next_text  # Если предложений мало, берем весь блок
            else:
                 logger.warning(f"Некорректные границы для следующего блока {i+2}, используем пустой контекст.")



        # Создаем промпт для пользователя (динамически, т.к. длина меняется)
        # Передаем только текст блока между маркерами, контекст отдельно
        min_len_api = int(original_block_length * MIN_REWRITE_LENGTH_RATIO)
        max_len_api = int(original_block_length * MAX_REWRITE_LENGTH_RATIO)
        user_input_content = create_rewrite_prompt(
            language, style, goal, block_text, prev_block_text, next_block_text, original_block_length
        )

        # Вызываем API с адаптивными параметрами
        block_failed_attempts = block.get('failed_attempts', 0)
        previous_quality = block.get('last_quality_metrics', None)
        
        new_text = call_gemini_rewrite_api(
            system_instruction=current_system_instruction.format(min_len=min_len_api, max_len=max_len_api), # Подставляем длину
            user_content=user_input_content,
            model_name=model_name,
            orig_len=original_block_length,
            original=block_text, # Оригинальный текст *этого* блока
            prev_block=prev_block_text, # Контекст до
            next_block=next_block_text,  # Контекст после
            stop_event=stop_event,
            failed_attempts=block_failed_attempts,
            previous_quality_metrics=previous_quality,
            language=language,
            style=style,
            goal=goal
        )

        # Если API вернул None из-за лимита, делаем дополнительную паузу перед следующим блоком
        if new_text is None:
            logger.warning(f"Блок {i+1}: Пропуск из-за ошибки API. Дополнительная пауза 10 секунд...")
            time.sleep(10)
            continue
        # `new_text` теперь уже приходит без маркеров, если валидация прошла успешно

        if new_text is not None: # Успешный переписанный и валидированный текст (уже без маркеров)
            new_text_len = count_chars(new_text)
            delta = new_text_len - len(block_text) # Изменение длины относительно *оригинального* текста блока
            logger.info(f"Блок {i+1} успешно переписан. Изменение длины: {delta} симв.")

            # Обновляем основной текст
            rewritten_text = rewritten_text[:start] + new_text + rewritten_text[end:]

            # Обновляем информацию о текущем блоке
            block['end_char_index'] = start + new_text_len # Новая конечная позиция
            # block['original_char_length'] остается прежней, чтобы сравнения длины шли с оригиналом
            block['processed'] = True
            block['failed_attempts'] = 0
            block['last_quality_metrics'] = None  # Сбрасываем метрики при успехе
            processed_idx = i

            # Сохраняем промежуточный результат
            save_intermediate(intermediate_file, rewritten_text, f"Блок {i+1}")

            # Корректируем границы последующих блоков, если длина изменилась
            if delta != 0:
                logger.debug(f"Сдвигаем границы последующих блоков на {delta}...")
                for j in range(i + 1, total_blocks):
                    blocks[j]['start_char_index'] += delta
                    blocks[j]['end_char_index'] += delta
                logger.debug("Сдвиг границ завершен.")

        else: # Переписывание не удалось
            block['failed_attempts'] += 1
            # Метрики качества будут сохранены в следующей попытке через параметр previous_quality_metrics
            logger.error(f"Блок {i+1} не был переписан после {block['failed_attempts']} попыток.")
            # Текст остается неизменным, границы не сдвигаются

        if progress_callback:
            progress_callback(i + 1, total_blocks)

        # Сохранение состояния
        if save_interval and ((new_text is not None and (i + 1) % save_interval == 0) or block['failed_attempts'] >= MAX_RETRIES):
             logger.info(f"Сохранение состояния на блоке {i+1}...")
             save_state(state_file, {
                 'processed_block_index': processed_idx,
                 'original_blocks_data': blocks, # Сохраняем обновленные границы
                 'total_blocks': total_blocks,
                 'timestamp': time.time()
             })

    processed_count = sum(1 for b in blocks if b.get('processed', False))
    failed_count = sum(1 for b in blocks if b.get('failed_attempts', 0) >= MAX_RETRIES and not b.get('processed', False))
    logger.info(f"Переписывание завершено. Обработано: {processed_count}/{total_blocks}. Пропущено из-за ошибок: {failed_count}.")

    # Финальное сохранение результата
    final_file_path = os.path.join(output_dir, base_name + FINAL_SUFFIX) # Убедимся что имя правильное
    save_intermediate(final_file_path, rewritten_text, "Финал")

    logger.info("Сохранение финального состояния...")
    save_state(state_file, {
        'processed_block_index': processed_idx,
        'original_blocks_data': blocks,
        'total_blocks': total_blocks,
        'timestamp': time.time()
    })

    logger.info(f"Финальный результат сохранен в: {final_file_path}")
    logger.info(f"Финальное состояние сохранено в: {state_file}")

    if progress_callback:
        progress_callback(processed_count, total_blocks)


# --- GUI Класс (без изменений) ---
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
        ttk.Entry(settings_frame, textvariable=self.input_var, width=60).grid(row=0, column=1, sticky=tk.EW) # Увеличена ширина
        ttk.Button(settings_frame, text="Обзор", command=self.browse_input).grid(row=0, column=2)

        ttk.Label(settings_frame, text="Выходной файл:").grid(row=1, column=0, sticky=tk.W)
        self.output_var = tk.StringVar()
        ttk.Entry(settings_frame, textvariable=self.output_var, width=60).grid(row=1, column=1, sticky=tk.EW) # Увеличена ширина
        ttk.Button(settings_frame, text="Обзор", command=self.browse_output).grid(row=1, column=2)

        ttk.Label(settings_frame, text="Язык:").grid(row=2, column=0, sticky=tk.W)
        self.lang_var = tk.StringVar(value="Русский")
        languages = [
            "Русский", "English", "Español", "Français", "Deutsch", "Italiano",
            "Português", "中文", "日本語", "한국어", "العربية", "हिन्दी",
            "Türkçe", "Polski", "Nederlands", "Svenska", "Norsk", "Dansk",
            "Suomi", "Čeština", "Slovenčina", "Magyar", "Română", "Български",
            "Українська", "Ελληνικά", "עברית", "ไทย", "Tiếng Việt", "Bahasa Indonesia"
        ]
        ttk.Combobox(settings_frame, textvariable=self.lang_var, values=languages, state="readonly", width=20).grid(row=2, column=1, sticky=tk.W)

        ttk.Label(settings_frame, text="Стиль:").grid(row=3, column=0, sticky=tk.NW)
        self.style_text = tk.Text(settings_frame, height=4, width=50)
        self.style_text.grid(row=3, column=1, columnspan=2, sticky=tk.EW)
        self.style_text.insert("1.0", "Увлекательный и живой повествовательный стиль, схожий с оригиналом, но с улучшенной динамикой и более богатой лексикой. Избегать канцеляризмов и излишней формальности.") # Пример

        ttk.Label(settings_frame, text="Цель:").grid(row=4, column=0, sticky=tk.NW)
        self.goal_text = tk.Text(settings_frame, height=4, width=50)
        self.goal_text.grid(row=4, column=1, columnspan=2, sticky=tk.EW)
        self.goal_text.insert("1.0", "Переписать сегмент, сохраняя основной смысл и сюжетную линию, но делая его более выразительным и интересным для современного читателя. Устранить возможные повторы и улучшить читаемость. Обеспечить естественные переходы к соседним блокам.") # Пример

        self.resume_var = tk.BooleanVar(value=True)
        ttk.Checkbutton(settings_frame, text="Возобновить", variable=self.resume_var).grid(row=5, column=0, sticky=tk.W)

        ttk.Label(settings_frame, text="Модель:").grid(row=6, column=0, sticky=tk.W)
        available_models = list_available_models()
        default_model = REWRITER_MODEL_DEFAULT if available_models and REWRITER_MODEL_DEFAULT in available_models else (available_models[0] if available_models else "gemini-1.5-flash")
        self.model_var = tk.StringVar(value=default_model)
        self.model_combobox = ttk.Combobox(settings_frame, textvariable=self.model_var, values=available_models, state="readonly", width=40) # Увеличена ширина
        self.model_combobox.grid(row=6, column=1, columnspan=2, sticky=tk.EW)

        settings_frame.grid_columnconfigure(1, weight=1) # Позволяет полю ввода растягиваться

        control_frame = ttk.Frame(main_frame)
        control_frame.pack(fill=tk.X, pady=5)

        self.start_btn = ttk.Button(control_frame, text="Старт", command=self.start_rewrite)
        self.start_btn.pack(side=tk.LEFT, padx=5)
        self.stop_btn = ttk.Button(control_frame, text="Стоп", command=self.stop_rewrite, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        self.status_var = tk.StringVar(value="Готов")
        ttk.Label(control_frame, textvariable=self.status_var, anchor=tk.E).pack(side=tk.RIGHT, padx=5)

        self.progress = ttk.Progressbar(control_frame, mode='determinate', length=300) # Уменьшена начальная длина
        self.progress.pack(fill=tk.X, expand=True, padx=5)

        log_frame = ttk.LabelFrame(main_frame, text="Лог")
        log_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 5))
        self.log_area = scrolledtext.ScrolledText(log_frame, state=tk.DISABLED, height=15) # Уменьшена высота
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        gui_handler = QueueHandler(log_queue)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(gui_handler)
        self.master.after(100, self.process_log_queue)

        # Добавляем обработчик закрытия окна
        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)

    def browse_input(self):
        file = filedialog.askopenfilename(filetypes=[("Текстовые файлы", "*.txt")])
        if file:
            self.input_var.set(file)
            if not self.output_var.get():
                base, ext = os.path.splitext(file)
                self.output_var.set(f"{base}{FINAL_SUFFIX}") # Авто-имя для выходного файла

    def browse_output(self):
        # Предлагаем имя по умолчанию на основе входного файла, если он выбран
        default_name = ""
        input_path = self.input_var.get()
        if input_path:
             default_name = f"{os.path.splitext(input_path)[0]}{FINAL_SUFFIX}"
        elif self.output_var.get(): # Если уже есть выходное имя, используем его
             default_name = self.output_var.get()

        file = filedialog.asksaveasfilename(defaultextension=".txt",
                                            filetypes=[("Текстовые файлы", "*.txt")],
                                            initialfile=os.path.basename(default_name) or "output_final_rewritten.txt",
                                            initialdir=os.path.dirname(default_name) or ".")
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
        # Проверяем состояние потока и обновляем кнопки/статус
        if self.rewrite_thread and not self.rewrite_thread.is_alive():
            if not self.stop_event.is_set(): # Если поток завершился сам
                 self.status_var.set("Завершено")
            else: # Если был остановлен
                 self.status_var.set("Остановлено")
            self.start_btn['state'] = tk.NORMAL
            self.stop_btn['state'] = tk.DISABLED
            self.rewrite_thread = None # Сбрасываем поток

        self.master.after(100, self.process_log_queue) # Повторяем проверку

    def update_progress(self, current: int, total: int):
        if total > 0:
            self.progress['maximum'] = total
            self.progress['value'] = current
            self.status_var.set(f"Обработка: {current}/{total} ({current/total*100:.1f}%)")
        else:
            self.progress['value'] = 0
            self.status_var.set("Инициализация...")


    def start_rewrite(self):
        params = {
            'input_file': self.input_var.get(),
            'output_file': self.output_var.get(),
            'language': self.lang_var.get(),
            'style': self.style_text.get("1.0", tk.END).strip(),
            'goal': self.goal_text.get("1.0", tk.END).strip(),
            'rewriter_model': self.model_var.get(),
            'resume': self.resume_var.get(),
            'save_interval': 1 # Получаем интервал из GUI
        }
        if not all([params['input_file'], params['output_file'], params['style'], params['goal']]):
            messagebox.showerror("Ошибка", "Заполните пути к файлам, стиль и цель переписывания.")
            return
        if not os.path.exists(params['input_file']):
             messagebox.showerror("Ошибка", f"Входной файл не найден:\n{params['input_file']}")
             return

        try:
            configure_gemini()
        except ValueError as e:
             messagebox.showerror("Ошибка конфигурации", f"Не удалось настроить Gemini API:\n{e}\n\nПроверьте API-ключ (GOOGLE_API_KEY) в .env файле.")
             return
        except Exception as e:
            messagebox.showerror("Ошибка конфигурации", f"Непредвиденная ошибка при настройке Gemini API:\n{e}")
            return

        self.stop_event.clear()
        self.start_btn['state'] = tk.DISABLED
        self.stop_btn['state'] = tk.NORMAL
        self.status_var.set("Запуск...")
        self.progress['value'] = 0 # Сбрасываем прогресс
        # Очищаем лог перед новым запуском (опционально)
        # self.log_area.configure(state=tk.NORMAL)
        # self.log_area.delete('1.0', tk.END)
        # self.log_area.configure(state=tk.DISABLED)

        self.rewrite_thread = threading.Thread(target=rewrite_process, args=(params, self.update_progress_threadsafe, self.stop_event), daemon=True)
        self.rewrite_thread.start()

    # Безопасная для потоков версия update_progress
    def update_progress_threadsafe(self, current, total):
        self.master.after(0, self.update_progress, current, total)


    def stop_rewrite(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
            logger.warning("Получен сигнал остановки...")
            self.status_var.set("Остановка...")
            self.stop_event.set()
            self.stop_btn['state'] = tk.DISABLED # Деактивируем кнопку, пока процесс не остановится

    def on_closing(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
            if messagebox.askyesno("Выход", "Процесс переписывания активен. Остановить его и выйти?"):
                self.stop_rewrite()
                self.master.after(200, self.master.destroy())
            else:
                return
        else:
            self.master.destroy()


if __name__ == "__main__":
    log_file = "rewriter.log"
    # Настройка логгирования в файл с ротацией
    try:
        file_handler = RotatingFileHandler(log_file, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(logging.INFO) # Устанавливаем уровень INFO для файла
    except Exception as e:
        print(f"Не удалось настроить логгирование в файл: {e}")
        logger.error(f"Не удалось настроить логгирование в файл: {e}")


    root = tk.Tk()
    try:
        sv_ttk.set_theme("dark") # Попробуем установить темную тему
    except Exception as e:
        print(f"Не удалось установить тему sv_ttk: {e}")
        # Используем стандартную тему ttk, если sv_ttk не удалась
        pass
    app = BookRewriterApp(root)
    root.mainloop()