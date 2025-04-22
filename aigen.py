import google.generativeai as genai
import os
import time
import json
import logging
from dotenv import load_dotenv
from typing import List, Dict, Optional, Tuple, TypedDict, Sequence
import tkinter as tk
from tkinter import ttk, filedialog, scrolledtext, messagebox
import threading
import queue
import difflib
from logging.handlers import RotatingFileHandler
import sv_ttk

log_queue = queue.Queue()
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

try:
    load_dotenv()
    GEMINI_API_KEY = os.getenv("GOOGLE_API_KEY")
    if not GEMINI_API_KEY:
        logging.warning("Переменная окружения GOOGLE_API_KEY не установлена или пуста. Проверьте .env файл.")
except Exception as e:
    logging.error(f"Ошибка при загрузке .env файла: {e}")
    GEMINI_API_KEY = None

AVAILABLE_MODELS = ["gemini-2.0-pro-exp",
                    "gemini-2.0-flash", "gemini-2.0-flash-exp",
                    "gemini-2.0-flash-lite-001",
                    "gemini-2.0-flash-live",
                    "gemini-1.5-pro", "gemini-1.5-pro-001", "gemini-1.5-pro-002",
                    "gemini-1.5-flash","gemini-1.5-flash-001", "gemini-1.5-flash-002",
                    "learnlm-1.5-pro-experimental",
                    "learnlm-2.0-flash-experimental",
                    "gemini-2.0-exp",
                    "gemini-2.5-pro-preview-03-25", "gemini-2.5-flash-preview-04-17"
]
REWRITER_MODEL_DEFAULT = "gemini-2.0-flash-exp"

START_MARKER = "<|~START_REWRITE~|>"
END_MARKER = "<|~END_REWRITE~|>"

BLOCK_TARGET_CHARS_REWRITE = 8000
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

GENERATION_CONFIG_REWRITER = genai.types.GenerationConfig(
    temperature=0.3,
    top_p=0.95,
    max_output_tokens=OUTPUT_TOKEN_LIMIT
)

STATE_FILENAME_SUFFIX = "_rewrite_state.json"
INTERMEDIATE_SUFFIX = "_intermediate.txt"
FINAL_OUTPUT_SUFFIX = "_final_rewritten.txt"

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
    if not GEMINI_API_KEY:
        raise ValueError("Переменная окружения GOOGLE_API_KEY не установлена или пуста.")
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        _ = list(genai.list_models())
        logging.info("Клиент Gemini успешно сконфигурирован и ключ API проверен.")
        return True
    except Exception as e:
        logging.error(f"Не удалось сконфигурировать или проверить ключ Gemini API: {e}", exc_info=True)
        raise ValueError(f"Ошибка конфигурации/проверки Gemini: {e}")

def count_chars(text: str) -> int:
    return len(text) if isinstance(text, str) else 0


def check_api_response_safety_and_finish(response: genai.types.GenerateContentResponse, context_msg: str) -> Tuple[Optional[str], Optional[str], bool]:
    extracted_text: Optional[str] = None
    error_message: Optional[str] = None
    was_max_tokens = False

    if response.prompt_feedback.block_reason:
        reason = response.prompt_feedback.block_reason.name
        ratings_str = str(getattr(response.prompt_feedback, 'safety_ratings', 'N/A'))
        error_message = f"{context_msg}: Промпт заблокирован ({reason}). Рейтинги: {ratings_str}"
        logging.error(error_message)
        return None, error_message, False

    if not response.candidates:
        try:
            fallback_text = response.text.strip()
            if fallback_text:
                logging.warning(f"{context_msg}: Кандидатов нет, но fallback response.text найден ({count_chars(fallback_text)} симв.). Используем его.")
                error_message = f"{context_msg}: В ответе нет кандидатов, но использован fallback response.text."
                return fallback_text, error_message, False
            else:
                error_message = f"{context_msg}: В ответе нет кандидатов и fallback response.text пуст."
                logging.error(error_message)
                return None, error_message, False
        except (ValueError, AttributeError) as e_fallback:
            error_message = f"{context_msg}: В ответе нет кандидатов, и не удалось получить fallback response.text ({e_fallback})."
            logging.error(error_message)
            return None, error_message, False

    try:
        candidate = response.candidates[0]
    except (IndexError, TypeError) as e:
        error_message = f"{context_msg}: Ошибка доступа к candidates[0] ({e}), хотя проверка 'not response.candidates' пройдена."
        logging.error(error_message)
        try:
            fallback_text = response.text.strip()
            if fallback_text:
                logging.warning(f"{context_msg}: Ошибка доступа к candidates[0], но fallback response.text найден ({count_chars(fallback_text)} симв.). Используем его.")
                return fallback_text, error_message, False
            else:
                return None, error_message, False
        except (ValueError, AttributeError) as e_fallback2:
             logging.debug(f"Не удалось получить response.text при ошибке доступа к candidates[0]: {e_fallback2}")
             return None, error_message, False

    finish_reason_value = 0
    finish_reason_str = "UNKNOWN"
    try:
        reason_enum = getattr(candidate, 'finish_reason', None)
        if reason_enum is not None:
            finish_reason_value = int(reason_enum.value)
            try:
                finish_reason_str = reason_enum.name
            except AttributeError:
                 finish_reason_str = f"ENUM_VAL_{finish_reason_value}"
        else:
             logging.warning(f"{context_msg}: Атрибут 'finish_reason' отсутствует у кандидата.")
    except (AttributeError, ValueError, TypeError) as e_reason:
        logging.warning(f"{context_msg}: Ошибка получения finish_reason: {e_reason}. Value={finish_reason_value}, Str={finish_reason_str}")

    logging.debug(f"  {context_msg}: Причина завершения = {finish_reason_str} (Value={finish_reason_value})")

    if finish_reason_value == 3:
        ratings_str = str(getattr(candidate, 'safety_ratings', 'N/A'))
        error_message = f"{context_msg}: Ответ заблокирован фильтром безопасности ({finish_reason_str}). Рейтинги: {ratings_str}"
        logging.error(error_message)
        return None, error_message, False
    elif finish_reason_value == 4:
        error_message = f"{context_msg}: Ответ заблокирован из-за цитирования ({finish_reason_str})."
        logging.error(error_message)
        return None, error_message, False

    if finish_reason_value == 2:
        logging.warning(f"{context_msg}: Ответ может быть усечен (MAX_TOKENS).")
        was_max_tokens = True

    if finish_reason_value not in [1, 2, 3, 4]:
         other_reason_msg = f"{context_msg}: Нестандартная причина завершения: {finish_reason_str} (Value={finish_reason_value})."
         logging.warning(other_reason_msg)
         error_message = other_reason_msg

    try:
        if hasattr(candidate, 'content') and candidate.content and hasattr(candidate.content, 'parts') and candidate.content.parts:
             extracted_text = "".join(part.text for part in candidate.content.parts if hasattr(part, 'text')).strip()
             logging.debug(f"  {context_msg}: Извлечен текст из 'candidate.content.parts' (Длина: {count_chars(extracted_text)}).")
        elif hasattr(response, 'text') and response.text is not None:
            fallback_text = response.text.strip()
            if fallback_text:
                 extracted_text = fallback_text
                 logging.debug(f"  {context_msg}: Извлечен текст из 'response.text' (Длина: {count_chars(extracted_text)}).")
            else:
                 extracted_text = ""
        else:
             extracted_text = ""

        if not extracted_text:
            if finish_reason_value == 1 and not was_max_tokens:
                empty_text_warning = f"{context_msg}: Причина STOP, но извлеченный текст пуст."
                logging.warning(empty_text_warning)
                return "", (error_message or empty_text_warning), False
            else:
                empty_text_error = f"{context_msg}: Извлеченный текст пуст (Причина: {finish_reason_str})."
                logging.warning(empty_text_error)
                return None, (error_message or empty_text_error), was_max_tokens

        if was_max_tokens:
            max_tokens_warning = f"{context_msg}: Ответ усечен (MAX_TOKENS). Результат может быть неполным."
            if error_message:
                error_message += f" {max_tokens_warning}"
            else:
                error_message = max_tokens_warning

        return extracted_text, error_message, was_max_tokens

    except (AttributeError, ValueError, TypeError) as e:
        error_message = f"{context_msg}: Ошибка доступа к атрибутам содержимого ответа: {type(e).__name__} - {e}"
        logging.error(error_message, exc_info=True)
        return None, error_message, False
    except Exception as e_unexpected:
         error_message = f"{context_msg}: Неожиданная ошибка при извлечении текста: {type(e_unexpected).__name__} - {e_unexpected}"
         logging.error(error_message, exc_info=True)
         return None, error_message, False

def find_closest_split_point(
    text: str,
    start_index: int,
    target_end_index: int,
    min_len: int,
    max_len: int,
    split_sequences: Sequence[str] = SPLIT_PRIORITY
    ) -> int:

    text_len = len(text)
    ideal_end = min(text_len, target_end_index)
    ideal_end = max(ideal_end, start_index + min_len)
    ideal_end = min(ideal_end, start_index + max_len)

    if ideal_end >= text_len:
        logging.debug(f"  Split: Reached end of text. Split at {text_len}")
        return text_len

    search_radius = int((target_end_index - start_index) * SEARCH_RADIUS_FACTOR)
    search_start = max(start_index + min_len, ideal_end - search_radius)
    search_end = min(text_len, ideal_end + search_radius)
    search_end = min(search_end, start_index + max_len)

    logging.debug(f"  Split Search: Target={ideal_end}, MinLen={min_len}, MaxLen={max_len} -> "
                  f"Range=[{search_start}, {search_end}], Radius={search_radius}")

    best_split_point = -1
    min_distance = float('inf')

    for seq in split_sequences:
        seq_len = len(seq)
        found_indices = []
        start_search = search_start
        while True:
            try:
                idx = text.index(seq, start_search, search_end)
                found_indices.append(idx + seq_len)
                start_search = idx + 1
            except ValueError:
                break

        if found_indices:
            logging.debug(f"  Split Search: Found '{seq.replace(chr(10), r'\\n')}' at indices: {found_indices}")
            for idx in found_indices:
                if idx > start_index + min_len:
                    distance = abs(idx - ideal_end)
                    if distance < min_distance:
                        min_distance = distance
                        best_split_point = idx
                        logging.debug(f"    New best split for '{seq.replace(chr(10), r'\\n')}': {best_split_point} (Dist: {distance})")
                    elif distance == min_distance and idx < best_split_point:
                         best_split_point = idx
                         logging.debug(f"    Tie distance for '{seq.replace(chr(10), r'\\n')}', preferring earlier: {best_split_point}")

            if best_split_point != -1:
                 logging.info(f"  Best Split Found: Using '{seq.replace(chr(10), r'\\n')}' at index {best_split_point} (closest to {ideal_end})")
                 return best_split_point

    logging.warning(f"  Split Search: No priority sequence found in range [{search_start}, {search_end}]. Using fallback.")

    fallback_search_end = min(ideal_end, start_index + max_len)
    fallback_search_start = max(start_index + min_len, fallback_search_end - search_radius * 2)
    try:
        last_space_index = text.rindex(' ', fallback_search_start, fallback_search_end)
        if last_space_index > start_index:
            logging.debug(f"  Fallback Split: Using last space at {last_space_index + 1}")
            return last_space_index + 1
    except ValueError:
        logging.debug("  Fallback Split: No space found in fallback range.")

    final_split = min(ideal_end, start_index + max_len)
    if final_split <= start_index:
        final_split = min(start_index + min_len, text_len)
        logging.warning(f"  Forcing minimum progress split at {final_split} as ideal/max end was <= start.")

    logging.warning(f"  Fallback Split: Resorting to clamped ideal/max index: {final_split}")
    return final_split

def propose_blocks_locally(
    full_text: str,
    target_chars_guideline: int
    ) -> Optional[List[BlockInfo]]:
    logging.info("Предложение блоков локально...")
    original_text_length = count_chars(full_text)
    if original_text_length == 0:
        logging.warning("Входной текст пуст, невозможно предложить блоки.")
        return None

    blocks: List[BlockInfo] = []
    current_char_index: int = 0
    block_counter: int = 0

    min_block_len = int(target_chars_guideline * MIN_BLOCK_LEN_FACTOR)
    max_block_len = int(target_chars_guideline * MAX_BLOCK_LEN_FACTOR)
    if min_block_len < 10: min_block_len = 10
    if max_block_len <= min_block_len : max_block_len = min_block_len + target_chars_guideline

    while current_char_index < original_text_length:
        logging.debug(f"Proposing block {block_counter + 1} starting at {current_char_index}")
        target_end_index = current_char_index + target_chars_guideline

        actual_end_index = find_closest_split_point(
            text=full_text,
            start_index=current_char_index,
            target_end_index=target_end_index,
            min_len=min_block_len,
            max_len=max_block_len
        )

        if actual_end_index <= current_char_index:
             logging.error(f"Ошибка разбиения: Не удалось найти точку разделения после индекса {current_char_index}. "
                           f"Принудительное разделение в конце текста ({original_text_length}) для завершения.")
             actual_end_index = original_text_length

        block_length = actual_end_index - current_char_index
        block_info: BlockInfo = {
            'block_index': block_counter,
            'start_char_index': current_char_index,
            'end_char_index': actual_end_index,
            'original_char_length': block_length,
            'processed': False,
            'failed_attempts': 0
        }
        blocks.append(block_info)
        logging.debug(f"  -> Proposed block {block_counter + 1}: [{current_char_index}:{actual_end_index}] Length: {block_length}")

        current_char_index = actual_end_index
        block_counter += 1

        if block_counter > original_text_length / 10 + 100:
             logging.critical("КРИТИЧЕСКАЯ ОШИБКА: Слишком много блоков создано, возможно бесконечный цикл в propose_blocks_locally. Прерывание.")
             return None

    logging.info(f"Локальное предложение блоков завершено. Создано {len(blocks)} блоков.")

    if blocks:
        last_block_end = blocks[-1]['end_char_index']
        if last_block_end != original_text_length:
            logging.warning(f"Несоответствие длины после локального разбиения: Последний end_index={last_block_end}, Длина текста={original_text_length}. Корректировка последнего блока.")
            blocks[-1]['end_char_index'] = original_text_length
            blocks[-1]['original_char_length'] = blocks[-1]['end_char_index'] - blocks[-1]['start_char_index']

        for i in range(len(blocks)):
             if i > 0 and blocks[i]['start_char_index'] != blocks[i-1]['end_char_index']:
                  logging.error(f"ОШИБКА ПЕРЕКРЫТИЯ/РАЗРЫВА обнаружена после локального разбиения между блоками {i-1} и {i}. "
                                f"End={blocks[i-1]['end_char_index']}, Start={blocks[i]['start_char_index']}")

    return blocks

def build_rewrite_prompt(target_language: str, target_style: str, rewriting_goal: str, full_context_with_markers: str, original_segment_length: int) -> str:
    min_len_target = int(original_segment_length * MIN_REWRITE_LENGTH_RATIO)
    max_len_target = int(original_segment_length * MAX_REWRITE_LENGTH_RATIO)

    prompt = f"""**Роль:** Внимательный ИИ-редактор, сфокусированный на качестве и точности.
**Задача:** Перепиши ТОЛЬКО сегмент текста, заключенный между `{START_MARKER}` и `{END_MARKER}`. Используй окружающий текст (ДО и ПОСЛЕ маркеров) ИСКЛЮЧИТЕЛЬНО как контекст для стиля и сюжета, но НЕ переписывай его.
**Параметры:** 
*   **Язык:** {target_language}
*   **Стиль:** {target_style}
*   **Цель:** {rewriting_goal}
*   **Ориентир по длине:** Исходный сегмент ~{original_segment_length} символов. Старайся сделать переписанный сегмент сопоставимой длины (например, **в диапазоне {min_len_target}-{max_len_target} символов**, если это не противоречит цели и стилю). **Избегай как чрезмерной краткости, так и излишней многословности.**

**Строгие Инструкции:**
1.  **ФОКУС:** Переписывай **ИСКЛЮЧИТЕЛЬНО** текст между `{START_MARKER}` и `{END_MARKER}`.
2.  **КАЧЕСТВО:** Следуй параметрам Языка, Стиля и Цели. Сохраняй ключевой смысл, сюжет, персонажей и факты оригинала, если Цель не требует изменений. Обеспечь плавные переходы с контекстом.
3.  **ЧИСТЫЙ ВЫВОД:**
    *   Ответ должен содержать **ТОЛЬКО переписанный текст сегмента**.
    *   **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** включать маркеры `{START_MARKER}` или `{END_MARKER}`.
    *   **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** включать любой текст из контекста (ДО или ПОСЛЕ маркеров).
    *   **КАТЕГОРИЧЕСКИ ЗАПРЕЩЕНО** добавлять любые комментарии, заголовки, пояснения или мета-текст. Ответ - это только чистый переписанный сегмент.
4.  **ДЛИНА:** Придерживайся ориентира по длине, сохраняя при этом качество и смысл.

**Текст с маркерами (контекст + сегмент для переписывания):**
--- НАЧАЛО КОНТЕКСТА ---
{full_context_with_markers}
--- КОНЕЦ КОНТЕКСТА ---

**Инструкция:** Перепиши **ТОЛЬКО** отмеченный сегмент ({START_MARKER}...{END_MARKER}). Ответ — **только** новый текст для этого сегмента, без маркеров и контекста, ориентируясь на длину ~{min_len_target}-{max_len_target} символов.
"""
    return prompt

def call_gemini_rewrite_api(
    prompt: str,
    model_name: str,
    original_block_length: int,
    original_block_content: str
    ) -> Optional[str]:
    try:
        model = genai.GenerativeModel(model_name)
    except Exception as e:
        logging.error(f"Не удалось инициализировать GenerativeModel '{model_name}' для переписывания: {e}", exc_info=True)
        return None

    retries = 0
    last_error_message = "Неизвестная ошибка перед началом цикла."

    while retries < MAX_RETRIES:
        context_msg = f"Переписывание блока (Попытка {retries + 1}/{MAX_RETRIES})"
        logging.info(f"  Вызов Gemini API для {context_msg}...")
        try:
            response = model.generate_content(
                prompt,
                safety_settings=SAFETY_SETTINGS,
                generation_config=GENERATION_CONFIG_REWRITER
            )

            rewritten_text, api_error_message, was_max_tokens = check_api_response_safety_and_finish(response, context_msg)
            last_error_message = api_error_message if api_error_message else "Нет сообщения об ошибке от API."

            if rewritten_text is not None:
                logging.info(f"  {context_msg}: Вызов API вернул текст ({count_chars(rewritten_text)} симв.).")

                validation_passed = True
                validation_error = None

                if not rewritten_text.strip() and original_block_content.strip():
                     validation_passed = False
                     validation_error = f"{context_msg}: API вернул пустой текст при непустом оригинале."

                elif START_MARKER in rewritten_text or END_MARKER in rewritten_text:
                    validation_passed = False
                    validation_error = f"{context_msg}: Переписанный текст ошибочно содержит маркеры '{START_MARKER}' или '{END_MARKER}'."

                elif original_block_content and rewritten_text:
                    if original_block_content.strip() and rewritten_text.strip() and original_block_content != rewritten_text:
                        matcher = difflib.SequenceMatcher(None, original_block_content, rewritten_text, autojunk=False)
                        similarity_ratio = matcher.ratio()
                        logging.debug(f"  Проверка схожести: {similarity_ratio:.3f}")
                        if similarity_ratio >= SIMILARITY_THRESHOLD:
                            validation_passed = False
                            validation_error = f"{context_msg}: Текст слишком похож на оригинал (Схожесть: {similarity_ratio:.2f} >= {SIMILARITY_THRESHOLD})."
                    elif original_block_content == rewritten_text:
                          logging.debug("  Текст идентичен оригиналу, пропуск проверки схожести (валидно).")

                if validation_passed and original_block_length > 0:
                    rewritten_len = count_chars(rewritten_text)
                    min_allowed_len = original_block_length * MIN_REWRITE_LENGTH_RATIO if original_block_length > 50 else original_block_length * 0.5
                    max_allowed_len = original_block_length * MAX_REWRITE_LENGTH_RATIO + 10 if original_block_length > 50 else original_block_length * 2.0 + 10

                    logging.debug(f"  Проверка длины: Оригинал={original_block_length}, Переписано={rewritten_len}, Мин={min_allowed_len:.0f}, Макс={max_allowed_len:.0f}")

                    if rewritten_len > max_allowed_len:
                        validation_passed = False
                        validation_error = f"{context_msg}: Текст слишком длинный ({rewritten_len} > {max_allowed_len:.0f} симв., >{MAX_REWRITE_LENGTH_RATIO*100:.0f}% от {original_block_length})."
                    elif rewritten_len < min_allowed_len:
                        if original_block_length > 20 or (rewritten_len == 0 and original_block_length > 5):
                             validation_passed = False
                             validation_error = f"{context_msg}: Текст слишком короткий ({rewritten_len} < {min_allowed_len:.0f} симв., <{MIN_REWRITE_LENGTH_RATIO*100:.0f}% от {original_block_length})."
                        else:
                             logging.debug(f"  Пропуск проверки мин. длины для очень короткого блока (оригинал={original_block_length}, переписано={rewritten_len})")

                if validation_passed:
                    logging.debug(f"  {context_msg}: Валидация переписанного блока пройдена.")
                    if was_max_tokens:
                         logging.warning(f"  {context_msg}: Ответ мог быть усечен (MAX_TOKENS), но прошел валидацию.")
                    return rewritten_text
                else:
                    last_error_message = validation_error or f"{context_msg}: Неизвестная ошибка валидации."
                    logging.warning(f"  {last_error_message} Отбрасываем результат.")

            elif api_error_message:
                 logging.error(f"  {context_msg}: Ошибка вызова API: {api_error_message}")
            else:
                 last_error_message = f"{context_msg}: API не вернул текст и не сообщил об ошибке (неожиданно)."
                 logging.error(last_error_message)

        except Exception as e:
            last_error_message = f"{context_msg}: Неожиданное исключение во время вызова API или обработки: {type(e).__name__} - {e}"
            logging.error(last_error_message, exc_info=True)

        retries += 1
        if retries < MAX_RETRIES:
            logging.warning(f"  Ошибка переписывания блока (попытка {retries}/{MAX_RETRIES}): {last_error_message}")
            logging.warning(f"  Повтор через {RETRY_DELAY_SECONDS} сек...")
            time.sleep(RETRY_DELAY_SECONDS)
        else:
            logging.error(f"Достигнуто максимальное количество попыток ({MAX_RETRIES}) для этого блока.")
            logging.error(f"Последняя ошибка: {last_error_message}")
            return None

    logging.error("Неожиданно завершен цикл call_gemini_rewrite_api.")
    return None

def save_state(filename: str, data: Dict):
    temp_filename = filename + ".tmp"
    try:
        output_dir = os.path.dirname(filename) or '.'
        os.makedirs(output_dir, exist_ok=True)

        with open(temp_filename, 'w', encoding='utf-8') as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

        os.replace(temp_filename, filename)
        logging.info(f"Состояние успешно сохранено в {filename}")
    except Exception as e:
        logging.error(f"Ошибка сохранения состояния в {filename}: {e}", exc_info=True)
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e_rem:
             logging.error(f"Не удалось удалить временный файл состояния {temp_filename}: {e_rem}")

def load_state(filename: str) -> Optional[Dict]:
    state = None
    valid_state_loaded = False

    if os.path.exists(filename):
        logging.info(f"Попытка загрузки состояния из основного файла: {filename}")
        try:
            with open(filename, 'r', encoding='utf-8') as f:
                state = json.load(f)
            if validate_state_data(state, filename):
                logging.info(f"Состояние успешно загружено и валидировано из {filename}")
                valid_state_loaded = True
            else:
                logging.warning(f"Основной файл состояния {filename} не прошел валидацию.")
                state = None
        except (json.JSONDecodeError, OSError, Exception) as e:
            logging.warning(f"Ошибка загрузки/валидации основного файла состояния {filename}: {e}. Попытка загрузить временный файл...")
            state = None

    if not valid_state_loaded:
        temp_filename = filename + ".tmp"
        if os.path.exists(temp_filename):
            logging.info(f"Попытка загрузки из временного файла состояния: {temp_filename}")
            try:
                with open(temp_filename, 'r', encoding='utf-8') as f_temp:
                    state = json.load(f_temp)
                if validate_state_data(state, temp_filename):
                    logging.info(f"Состояние успешно загружено и валидировано из временного файла {temp_filename}")
                    valid_state_loaded = True
                    try:
                        os.replace(temp_filename, filename)
                        logging.info(f"Основной файл состояния {filename} восстановлен из временного.")
                    except OSError as e_replace:
                        logging.error(f"Не удалось восстановить основной файл состояния из {temp_filename}: {e_replace}")
                else:
                    logging.error(f"Временный файл состояния {temp_filename} также не прошел валидацию.")
                    state = None
            except (json.JSONDecodeError, OSError, Exception) as e_temp:
                logging.error(f"Ошибка загрузки или валидации временного файла состояния {temp_filename}: {e_temp}", exc_info=True)
                state = None
        else:
            if state is None and os.path.exists(filename):
                 logging.error(f"Основной файл состояния {filename} поврежден/невалиден, временный файл {temp_filename} отсутствует.")
            elif not os.path.exists(filename):
                 logging.info(f"Файл состояния {filename} не найден (ни основной, ни временный).")

    return state if valid_state_loaded else None

def validate_state_data(state: Dict, filename: str) -> bool:
    try:
        required_keys = ['processed_block_index', 'original_blocks_data', 'total_blocks']
        if not all(key in state for key in required_keys):
            missing = [k for k in required_keys if k not in state]
            raise KeyError(f"Отсутствуют обязательные ключи: {missing}")

        if not isinstance(state.get('processed_block_index'), int):
            raise TypeError(f"'processed_block_index' должен быть int, получен {type(state.get('processed_block_index'))}")
        if not isinstance(state.get('original_blocks_data'), list):
             raise TypeError(f"'original_blocks_data' должен быть list, получен {type(state.get('original_blocks_data'))}")
        if not isinstance(state.get('total_blocks'), int):
             raise TypeError(f"'total_blocks' должен быть int, получен {type(state.get('total_blocks'))}")

        actual_len = len(state['original_blocks_data'])
        if state['total_blocks'] != actual_len:
            logging.warning(f"Несоответствие в файле состояния '{filename}': 'total_blocks' ({state['total_blocks']}) != длина 'original_blocks_data' ({actual_len}). Используем длину списка.")
            state['total_blocks'] = actual_len

        total_blocks = state['total_blocks']
        processed_index = state['processed_block_index']
        if total_blocks == 0:
            if processed_index != -1:
                 raise ValueError(f"Невалидное значение 'processed_block_index': {processed_index} (при total_blocks=0, ожидалось -1)")
        elif not (-1 <= processed_index < total_blocks):
             raise ValueError(f"Невалидное значение 'processed_block_index': {processed_index} (допустимый диапазон: -1..{total_blocks - 1})")

        if state['original_blocks_data']:
             first_block = state['original_blocks_data'][0]
             if isinstance(first_block, dict):
                 block_keys = ['block_index', 'start_char_index', 'end_char_index', 'original_char_length', 'processed', 'failed_attempts']
                 if not all(key in first_block for key in block_keys):
                     missing = [k for k in block_keys if k not in first_block]
                     logging.warning(f"Первый блок в '{filename}' не содержит ожидаемых ключей (отсутствуют: {missing}). Структура может быть нарушена.")
             else:
                  logging.warning(f"Первый элемент в 'original_blocks_data' в '{filename}' не является словарем (тип: {type(first_block)}).")

        logging.debug(f"Файл состояния '{filename}' прошел базовую проверку.")
        return True

    except (TypeError, KeyError, ValueError) as e:
        logging.error(f"Ошибка валидации данных состояния из '{filename}': {e}", exc_info=False)
        return False
    except Exception as e_unexp:
         logging.error(f"Неожиданная ошибка валидации состояния из '{filename}': {e_unexp}", exc_info=True)
         return False

def save_intermediate_file(filename: str, content: str, context_msg: str = ""):
    temp_filename = filename + ".tmp"
    try:
        with open(temp_filename, 'w', encoding='utf-8') as f_inter:
            f_inter.write(content)
        os.replace(temp_filename, filename)
        if context_msg:
            logging.info(f"  {context_msg}: Промежуточный файл сохранен/обновлен: {filename}")
        else:
            logging.info(f"Промежуточный файл сохранен: {filename}")
    except Exception as e:
        logging.warning(f"ПРЕДУПРЕЖДЕНИЕ ({context_msg}): Не удалось сохранить/обновить промежуточный файл '{filename}': {e}", exc_info=False)
        try:
            if os.path.exists(temp_filename):
                os.remove(temp_filename)
        except Exception as e_rem:
            logging.error(f"Не удалось удалить временный промежуточный файл {temp_filename}: {e_rem}")

def run_rewrite_process(params: Dict, progress_callback=None, stop_event=None):
    input_file = params['input_file']
    output_file = params['output_file']
    target_language = params['language']
    target_style = params['style']
    rewriting_goal = params['goal']
    rewriter_model = params['rewriter_model']
    resume = params['resume']
    save_interval = params['save_interval']

    output_dir = os.path.dirname(output_file) or '.'
    output_base, _ = os.path.splitext(os.path.basename(output_file))
    final_output_file = output_file

    state_file = os.path.join(output_dir, output_base + STATE_FILENAME_SUFFIX)
    intermediate_file = os.path.join(output_dir, output_base + INTERMEDIATE_SUFFIX)

    logging.info(f"--- Параметры Запуска ---")
    logging.info(f"Входной файл: {input_file}")
    logging.info(f"Выходной файл: {final_output_file}")
    logging.info(f"Файл состояния: {state_file}")
    logging.info(f"Промежуточный файл: {intermediate_file}")
    logging.info(f"Язык: {target_language}, Стиль: '{target_style[:50]}...', Цель: '{rewriting_goal[:50]}...'")
    logging.info(f"Модель Переписывания: '{rewriter_model}' (Предложение блоков: Локальное)")
    logging.info(f"Опции: Возобновить={resume}, Интервал сохр.={save_interval}")
    logging.info(f"Целевая длина блока (локально): {BLOCK_TARGET_CHARS_REWRITE}")
    logging.info(f"Порог схожести: {SIMILARITY_THRESHOLD*100:.1f}%")
    logging.info(f"--- Конец Параметров ---")

    try:
        os.makedirs(output_dir, exist_ok=True)
        logging.info(f"Выходная директория '{output_dir}' существует или создана.")
    except OSError as e:
         logging.error(f"Не удалось создать выходную директорию '{output_dir}': {e}")
         if tk.Toplevel.winfo_exists(root):
            messagebox.showerror("Ошибка директории", f"Невозможно создать/получить доступ к выходной директории:\n{output_dir}\n\nОшибка: {e}")
         return

    original_book_text: Optional[str] = None
    current_rewritten_book_text: Optional[str] = None
    original_blocks_data: List[BlockInfo] = []
    processed_block_index: int = -1
    total_blocks = 0
    state_loaded = False

    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            original_book_text = f.read()
        if not original_book_text:
            logging.error(f"Входной файл '{input_file}' пуст.")
            if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка файла", f"Входной файл пуст:\n{input_file}")
            return
        original_text_length = count_chars(original_book_text)
        logging.info(f"Прочитан входной файл: {input_file} ({original_text_length} символов)")
    except FileNotFoundError:
        logging.error(f"Входной файл не найден: {input_file}")
        if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка файла", f"Входной файл не найден:\n{input_file}")
        return
    except Exception as e:
        logging.error(f"Ошибка чтения входного файла '{input_file}': {e}", exc_info=True)
        if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка файла", f"Ошибка чтения входного файла:\n{input_file}\n\nОшибка: {e}")
        return

    if resume:
        logging.info(f"Попытка возобновления из файла состояния: {state_file}")
        state = load_state(state_file)
        if state:
            logging.info("Файл состояния успешно загружен и валидирован.")
            try:
                processed_block_index = state['processed_block_index']
                original_blocks_data = state['original_blocks_data']
                total_blocks = state['total_blocks']

                if not original_blocks_data:
                    raise ValueError("Блоки данных в файле состояния пусты.")

                if os.path.exists(intermediate_file):
                    try:
                        with open(intermediate_file, 'r', encoding='utf-8') as f_inter:
                            current_rewritten_book_text = f_inter.read()
                        logging.info(f"Загружен промежуточный текст ({count_chars(current_rewritten_book_text)} симв.) из {intermediate_file}")
                        state_loaded = True
                    except Exception as e_inter_load:
                         logging.error(f"Ошибка загрузки промежуточного файла '{intermediate_file}': {e_inter_load}. Возобновление невозможно.")
                         state_loaded = False
                else:
                     logging.error(f"Невозможно возобновить: Промежуточный файл '{intermediate_file}' отсутствует или недоступен, хотя файл состояния '{state_file}' найден.")
                     state_loaded = False

                if state_loaded:
                    logging.info(f"Возобновление успешно. Всего блоков: {total_blocks}. Следующий блок для обработки: {processed_block_index + 2} (индекс {processed_block_index + 1}).")
                else:
                    logging.warning("Не удалось загрузить промежуточный текст. Перезапуск процесса.")
                    resume = False
                    processed_block_index = -1
                    original_blocks_data = []
                    current_rewritten_book_text = None
                    total_blocks = 0

            except (KeyError, ValueError, TypeError, IndexError, Exception) as e_state:
                logging.error(f"Ошибка данных в файле состояния '{state_file}' при возобновлении: {e_state}. Перезапуск.", exc_info=True)
                resume = False
                state_loaded = False
                processed_block_index = -1
                original_blocks_data = []
                current_rewritten_book_text = None
                total_blocks = 0
        else:
            logging.warning(f"Файл состояния '{state_file}' не найден, невалиден или не удалось загрузить. Перезапуск.")
            resume = False
            state_loaded = False

    if not state_loaded:
        logging.info("Запуск нового процесса или перезапуск после неудачного возобновления.")
        current_rewritten_book_text = original_book_text
        processed_block_index = -1
        original_blocks_data = []
        total_blocks = 0

        logging.info(f"Создание/перезапись начального промежуточного файла: {intermediate_file}")
        save_intermediate_file(intermediate_file, current_rewritten_book_text, "Начальное создание")

        logging.info("Запуск локального предложения блоков...")
        proposed_blocks = propose_blocks_locally(
            original_book_text,
            BLOCK_TARGET_CHARS_REWRITE
        )

        if proposed_blocks:
            original_blocks_data = proposed_blocks
            total_blocks = len(original_blocks_data)
            logging.info(f"Инициализация завершена. Всего блоков для обработки: {total_blocks}")
            logging.info(f"Сохранение начального состояния после определения блоков...")
            initial_state_data = {
                'processed_block_index': -1,
                'original_blocks_data': original_blocks_data,
                'total_blocks': total_blocks,
                'timestamp': time.time(),
                'last_processed_block_index_in_loop': -1,
                'status_at_save': 'initialized',
                'params': {k: v for k, v in params.items() if k not in ['input_file', 'output_file', 'style', 'goal']}
            }
            save_state(state_file, initial_state_data)

        else:
            error_msg = "КРИТИЧЕСКАЯ ОШИБКА: Не удалось создать структуру блоков локально. Обработка невозможна."
            logging.critical(error_msg)
            if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка инициализации", error_msg + "\n\nПроверьте лог на возможные ошибки в тексте или логике разбиения.")
            return

    if not original_blocks_data or total_blocks <= 0:
        error_msg = "КРИТИЧЕСКАЯ ОШИБКА: Отсутствуют данные о блоках для начала переписывания (после инициализации/загрузки)."
        logging.critical(error_msg)
        if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка выполнения", error_msg)
        return
    if current_rewritten_book_text is None:
        error_msg = "КРИТИЧЕСКАЯ ОШИБКА: Текст для переписывания не инициализирован (current_rewritten_book_text is None) перед основным циклом."
        logging.critical(error_msg)
        if tk.Toplevel.winfo_exists(root): messagebox.showerror("Критическая ошибка", error_msg + "\n\nВнутренняя ошибка программы.")
        return

    logging.info(f"\n{'='*15} Начало основного цикла переписывания {'='*15}")
    logging.info(f"Всего блоков: {total_blocks}. Начинаем с блока {processed_block_index + 2} (индекс {processed_block_index + 1}).")

    if progress_callback:
        progress_callback(processed_block_index + 1, total_blocks)

    critical_error_occurred = False

    for i in range(total_blocks):
        if stop_event and stop_event.is_set():
            logging.warning(f"Получен сигнал остановки перед обработкой блока {i+1}.")
            break

        try:
            current_block_info = original_blocks_data[i]
            if current_block_info['block_index'] != i:
                 logging.warning(f"Несоответствие индекса блока: в данных {current_block_info['block_index']}, ожидался индекс цикла {i}. Используем данные блока как есть.")
        except IndexError:
            logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Индекс {i} вне диапазона original_blocks_data (длина {len(original_blocks_data)}). Прерывание.")
            critical_error_occurred = True
            break
        except KeyError as e:
             logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Отсутствует ключ '{e}' в данных блока {i}. Прерывание.")
             critical_error_occurred = True
             break
        except TypeError as e:
             logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА: Ожидался словарь, но получен {type(current_block_info)} для блока {i}. Данные повреждены? Прерывание. Ошибка: {e}")
             critical_error_occurred = True
             break

        if i <= processed_block_index:
            logging.debug(f"Пропуск уже обработанного блока {i+1}/{total_blocks} (индекс {i} <= processed {processed_block_index}).")
            if progress_callback: progress_callback(i + 1, total_blocks)
            continue

        if current_block_info.get('processed', False):
            logging.warning(f"Пропуск блока {i+1}, помеченного 'processed=True', хотя индекс цикла {i} > processed_block_index ({processed_block_index}). Возможно, ошибка в файле состояния или логике. Обновляем processed_block_index.")
            processed_block_index = i
            if progress_callback: progress_callback(i + 1, total_blocks)
            continue

        if current_block_info.get('failed_attempts', 0) >= MAX_RETRIES:
            logging.warning(f"Пропуск блока {i+1} из-за превышения лимита неудачных попыток ({current_block_info.get('failed_attempts', 0)} >= {MAX_RETRIES}).")
            original_blocks_data[i]['processed'] = False
            if progress_callback: progress_callback(i + 1, total_blocks)
            continue

        start_index = current_block_info['start_char_index']
        end_index = current_block_info['end_char_index']
        original_len_from_data = current_block_info.get('original_char_length', end_index - start_index)
        if original_len_from_data <= 0 and (end_index - start_index > 0):
            logging.warning(f"  Длина блока {i+1} из данных ({original_len_from_data}) некорректна. Используем расчетную {end_index - start_index}.")
            original_len_from_data = end_index - start_index

        logging.info(f"\n>>> Обработка блока {i+1}/{total_blocks} (Индекс {i}) | Символы [{start_index}:{end_index}] | Длина (исходная): {original_len_from_data} <<<")

        current_text_len = count_chars(current_rewritten_book_text)

        if not (0 <= start_index <= current_text_len):
            logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА: Начальный индекс блока {i+1} ({start_index}) вне диапазона текущего текста (0..{current_text_len}). Рассинхронизация индексов. Прерывание.")
            critical_error_occurred = True
            break

        adjusted_end_index = min(end_index, current_text_len)
        if adjusted_end_index != end_index:
            logging.warning(f"  Конечный индекс блока {i+1} ({end_index}) скорректирован на {adjusted_end_index} из-за текущей длины текста ({current_text_len}). Возможен дрейф индексов.")

        if adjusted_end_index < start_index:
             logging.warning(f"  Предупреждение: Скорректированный end_index ({adjusted_end_index}) < start_index ({start_index}) для блока {i+1}. Блок будет пустым при извлечении.")
             adjusted_end_index = start_index

        if i > 0:
             try:
                prev_block_end_index_stored = original_blocks_data[i-1]['end_char_index']
                if start_index != prev_block_end_index_stored:
                    gap = start_index - prev_block_end_index_stored
                    logging.warning(f"  Обнаружен разрыв/перекрытие перед блоком {i+1} на основе сохраненных индексов: start_index ({start_index}) != end_index ({prev_block_end_index_stored}) предыдущего. Разница: {gap} симв.")
             except IndexError:
                 logging.error(f"  Не удалось получить данные предыдущего блока ({i-1}) для проверки перекрытия.")
             except KeyError:
                 logging.error(f"  Отсутствует ключ 'end_char_index' у предыдущего блока ({i-1}).")

        try:
            block_content_to_rewrite = current_rewritten_book_text[start_index:adjusted_end_index]
            actual_extracted_len = len(block_content_to_rewrite)
            block_len_for_api = actual_extracted_len
            if abs(original_len_from_data - actual_extracted_len) > 5:
                 logging.warning(f"  Длина блока {i+1} по данным ({original_len_from_data}) не совпадает с фактической длиной извлеченного текста ({actual_extracted_len}). Используем фактическую ({actual_extracted_len}) для API.")
            logging.debug(f"  Содержимое для переписывания (длина {block_len_for_api}, начало): '{block_content_to_rewrite[:100].replace(chr(10),' ')}...'")
        except IndexError:
             logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА ИНДЕКСАЦИИ при извлечении текста блока {i+1} [{start_index}:{adjusted_end_index}] из текста длиной {current_text_len}. Прерывание.")
             critical_error_occurred = True
             break

        try:
            text_for_api = (
                current_rewritten_book_text[:start_index]
                + START_MARKER
                + block_content_to_rewrite
                + END_MARKER
                + current_rewritten_book_text[adjusted_end_index:]
            )
        except Exception as e_concat:
            logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА при конкатенации текста для API блока {i+1}: {e_concat}. Прерывание.", exc_info=True)
            critical_error_occurred = True
            break

        rewrite_prompt = build_rewrite_prompt(
            target_language, target_style, rewriting_goal, text_for_api,
            block_len_for_api
        )

        rewritten_block_text = call_gemini_rewrite_api(
            prompt=rewrite_prompt,
            model_name=rewriter_model,
            original_block_length=block_len_for_api,
            original_block_content=block_content_to_rewrite
        )

        block_processed_successfully = False
        if rewritten_block_text is not None:
            rewritten_len = count_chars(rewritten_block_text)
            delta = rewritten_len - block_len_for_api

            logging.info(f"  Блок {i+1} успешно переписан и прошел валидацию. Исходная длина (извлеч.): {block_len_for_api}, Новая длина: {rewritten_len}, Дельта: {delta:+}")
            logging.debug(f"  Переписанное содержимое (начало): '{rewritten_block_text[:100].replace(chr(10),' ')}...'")

            try:
                current_rewritten_book_text = (
                    current_rewritten_book_text[:start_index]
                    + rewritten_block_text
                    + current_rewritten_book_text[adjusted_end_index:]
                )
                block_processed_successfully = True
                save_intermediate_file(intermediate_file, current_rewritten_book_text, f"После блока {i+1}")
            except Exception as e_replace:
                 logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА при замене текста для блока {i+1}: {e_replace}. Прерывание.", exc_info=True)
                 block_processed_successfully = False
                 critical_error_occurred = True

            if block_processed_successfully:
                try:
                    new_end_index = start_index + rewritten_len
                    original_blocks_data[i]['end_char_index'] = new_end_index
                    original_blocks_data[i]['processed'] = True
                    original_blocks_data[i]['failed_attempts'] = 0
                    processed_block_index = i

                    if delta != 0:
                         logging.debug(f"  Обновление индексов для {total_blocks - (i + 1)} последующих блоков на delta={delta}...")
                         for j in range(i + 1, total_blocks):
                             original_blocks_data[j]['start_char_index'] += delta
                             original_blocks_data[j]['end_char_index'] += delta
                         logging.debug("  Обновление индексов последующих блоков завершено.")
                    else:
                         logging.debug("  Длина блока не изменилась, индексы последующих блоков не требуют обновления.")

                except IndexError as e_idx_update:
                     logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА (IndexError) при обновлении индексов блока {i} или последующих: {e_idx_update}. Прерывание.", exc_info=False)
                     critical_error_occurred = True
                except KeyError as e_key_update:
                     logging.critical(f"  КРИТИЧЕСКАЯ ОШИБКА (KeyError) при обновлении данных блока {i} или последующих: {e_key_update}. Прерывание.", exc_info=False)
                     critical_error_occurred = True
                except Exception as e_update:
                     logging.critical(f"  КРИТИЧЕСКАЯ НЕОЖИДАННАЯ ОШИБКА при обновлении данных блоков: {e_update}. Прерывание.", exc_info=True)
                     critical_error_occurred = True
        else:
            logging.error(f"  Не удалось переписать блок {i+1} (после {MAX_RETRIES} попыток или из-за ошибки/валидации). Оригинал ОСТАВЛЕН в тексте.")
            block_processed_successfully = False
            try:
                 current_attempts = original_blocks_data[i].get('failed_attempts', 0)
                 original_blocks_data[i]['failed_attempts'] = current_attempts + 1
                 original_blocks_data[i]['processed'] = False
                 logging.info(f"  Увеличено число неудачных попыток для блока {i+1} до {original_blocks_data[i]['failed_attempts']}")
            except (IndexError, KeyError, TypeError) as e:
                 logging.error(f"  Ошибка обновления статуса неудачи для блока {i}: {e}")

        if progress_callback:
            progress_callback(i + 1, total_blocks)

        should_save_state = False
        save_reason = ""
        if save_interval > 0 and ((i + 1) % save_interval == 0):
            should_save_state = True
            save_reason = f"(интервал {save_interval})"
        elif not block_processed_successfully and original_blocks_data[i].get('failed_attempts', 0) < MAX_RETRIES:
            should_save_state = True
            save_reason = "(после ошибки блока)"
        elif i == total_blocks - 1:
            should_save_state = True
            save_reason = "(финальное после цикла)"

        if should_save_state:
            logging.info(f"Сохранение промежуточного состояния {save_reason} после попытки блока {i+1}...")
            state_data = {
                'processed_block_index': processed_block_index,
                'original_blocks_data': original_blocks_data,
                'total_blocks': total_blocks,
                'timestamp': time.time(),
                'last_processed_block_index_in_loop': i,
                'status_at_save': 'success' if block_processed_successfully else ('failed_or_skipped' if i <= processed_block_index else 'failed_attempt'),
                'params': {k: v for k, v in params.items() if k not in ['input_file', 'output_file', 'style', 'goal']}
            }
            save_state(state_file, state_data)

        if stop_event and stop_event.is_set():
            logging.warning(f"Сигнал остановки обработан после завершения действий для блока {i+1}.")
            break

        if critical_error_occurred:
             logging.critical("Критическая ошибка обнаружена во время обработки блока. Прерывание основного цикла.")
             break

    logging.info(f"\n{'='*15} Основной цикл переписывания завершен {'='*15}")

    final_processed_count = sum(1 for b in original_blocks_data if b.get('processed'))
    final_failed_count = sum(1 for idx, b in enumerate(original_blocks_data)
                              if not b.get('processed') and (b.get('failed_attempts', 0) >= MAX_RETRIES or (critical_error_occurred and idx == i) ))
    last_loop_index = i if 'i' in locals() and not critical_error_occurred else processed_block_index if critical_error_occurred else total_blocks -1
    final_skipped_or_pending_count = sum(1 for idx, b in enumerate(original_blocks_data)
                                         if idx > processed_block_index and not b.get('processed') and b.get('failed_attempts', 0) < MAX_RETRIES)

    logging.info(f"--- Итоговая Статистика ---")
    logging.info(f"  Всего блоков: {total_blocks}")
    logging.info(f"  Успешно обработано (processed=True): {final_processed_count}")
    if final_failed_count > 0:
        logging.warning(f"  Не удалось обработать (макс. попыток / крит. ошибка): {final_failed_count}")
    if final_skipped_or_pending_count > 0:
        status_reason = ""
        if critical_error_occurred: status_reason = "(из-за критической ошибки)"
        elif stop_event and stop_event.is_set(): status_reason = "(из-за остановки пользователем)"
        elif final_processed_count + final_failed_count < total_blocks: status_reason = "(не завершено)"
        logging.warning(f"  Пропущено / Ожидает обработки: {final_skipped_or_pending_count} {status_reason}")

    final_status = "unknown"
    if critical_error_occurred: final_status = "критическая ошибка"
    elif stop_event and stop_event.is_set(): final_status = "прервано пользователем"
    elif final_processed_count == total_blocks: final_status = "завершено успешно"
    elif final_processed_count + final_failed_count == total_blocks: final_status = "завершено (с ошибками обработки)"
    elif final_skipped_or_pending_count > 0 : final_status = f"не завершено ({final_skipped_or_pending_count} блоков осталось)"
    else: final_status = "завершено (неопределенный статус)"

    logging.info(f"Финальный Статус: {final_status}")

    if final_output_file:
        should_save_final = (processed_block_index >= 0) or critical_error_occurred or (stop_event and stop_event.is_set()) or (not resume and not state_loaded)
        if should_save_final and current_rewritten_book_text is not None:
            final_text_len = count_chars(current_rewritten_book_text)
            logging.info(f"Сохранение итогового текста (длина {final_text_len}) в: {final_output_file}")
            try:
                temp_final_filename = final_output_file + ".final.tmp"
                with open(temp_final_filename, 'w', encoding='utf-8') as f_final:
                    f_final.write(current_rewritten_book_text)
                os.replace(temp_final_filename, final_output_file)
                logging.info("Итоговый текст успешно сохранен.")
            except Exception as e:
                logging.error(f"КРИТИЧЕСКАЯ ОШИБКА сохранения итогового файла '{final_output_file}': {e}", exc_info=True)
                if tk.Toplevel.winfo_exists(root): messagebox.showerror("Ошибка сохранения", f"Не удалось сохранить итоговый файл:\n{final_output_file}\n\nОшибка: {e}")
                try:
                    if os.path.exists(temp_final_filename): os.remove(temp_final_filename)
                except Exception: pass
        elif current_rewritten_book_text is None:
             logging.error("Итоговый текст отсутствует (None), финальный файл НЕ сохранен.")
        else:
            logging.warning(f"Финальный файл '{final_output_file}' не сохраняется. Причина: Не было успешных обработок или изменений с момента последней загрузки/инициализации.")
    else:
        logging.warning("Имя итогового файла не указано. Итоговый текст НЕ сохранен.")

    logging.info("Сохранение финального файла состояния...")
    final_state_data = {
        'processed_block_index': processed_block_index,
        'original_blocks_data': original_blocks_data,
        'total_blocks': total_blocks,
        'timestamp': time.time(),
        'final_status': final_status,
        'stats': {
            'processed': final_processed_count,
            'failed': final_failed_count,
            'skipped_or_pending': final_skipped_or_pending_count,
        },
        'params': {k: v for k, v in params.items() if k not in ['input_file', 'output_file', 'style', 'goal']}
    }
    save_state(state_file, final_state_data)

    logging.info(f"--- Процесс переписывания завершен (Статус: {final_status}) ---")

class BookRewriterApp:
    def __init__(self, master):
        self.master = master
        master.title("AI Book Rewriter v1.8.0 (Gemini + Local Split + Resume)")
        master.geometry("950x750")
        master.minsize(800, 600)

        self.master.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.rewrite_thread = None
        self.stop_event = threading.Event()

        style = ttk.Style()
        style.configure("TLabel", padding=5)
        style.configure("TButton", padding=5)
        style.configure("TEntry", padding=(5, 3), relief=tk.FLAT)
        style.configure("TCombobox", padding=5)
        style.configure("TCheckbutton", padding=5)
        style.configure("TFrame", padding=10)
        style.configure("TLabelframe", padding=10, relief=tk.GROOVE)
        style.configure("TLabelframe.Label", font=("TkDefaultFont", 10, "bold"))
        style.configure("Accent.TButton", font=("TkDefaultFont", 10, "bold"))

        try:
            is_dark_theme = "dark" in style.theme_use().lower()
            fallback_text_bg = "#333333" if is_dark_theme else "#FFFFFF"
            fallback_text_fg = "#FFFFFF" if is_dark_theme else "#000000"
            fallback_border = "#555555" if is_dark_theme else "#ABABAB"
            fallback_insert = fallback_text_fg
            fallback_select_bg = "#0078D7"
            fallback_select_fg = "#FFFFFF"

            text_bg = style.lookup('TEntry', 'fieldbackground') or fallback_text_bg
            text_fg = style.lookup('TEntry', 'foreground') or fallback_text_fg
            text_border = style.lookup('TFrame', 'background') or fallback_border
            insert_color = style.lookup('TEntry', 'insertcolor') or fallback_insert
            select_bg = style.lookup('TEntry', 'selectbackground') or fallback_select_bg
            select_fg = style.lookup('TEntry', 'selectforeground') or fallback_select_fg
            self.placeholder_color = "grey" if not is_dark_theme else "#999999"

            logging.debug(f"Тема: {style.theme_use()}. Цвета для Text: BG={text_bg}, FG={text_fg}, Border={text_border}, Insert={insert_color}, SelBG={select_bg}, SelFG={select_fg}")

        except Exception as e:
            logging.error(f"Ошибка получения стилей ttk: {e}. Используются запасные цвета.")
            text_bg, text_fg, text_border, insert_color, select_bg, select_fg = (
                "#FFFFFF", "#000000", "#ABABAB", "#000000", "#0078D7", "#FFFFFF"
            )
            self.placeholder_color = "grey"

        main_frame = ttk.Frame(master, padding="10 10 10 10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        settings_frame = ttk.LabelFrame(main_frame, text="Настройки переписывания")
        settings_frame.pack(pady=10, padx=5, fill=tk.X)
        settings_frame.columnconfigure(1, weight=1)

        ttk.Label(settings_frame, text="Входной файл (.txt):*").grid(row=0, column=0, sticky=tk.W, padx=5, pady=3)
        self.input_file_var = tk.StringVar()
        self.input_entry = ttk.Entry(settings_frame, textvariable=self.input_file_var, width=80)
        self.input_entry.grid(row=0, column=1, sticky=tk.EW, padx=5, pady=3)
        self.browse_input_btn = ttk.Button(settings_frame, text="Обзор...", command=self.browse_input)
        self.browse_input_btn.grid(row=0, column=2, padx=5, pady=3)

        ttk.Label(settings_frame, text="Выходной файл (.txt):*").grid(row=1, column=0, sticky=tk.W, padx=5, pady=3)
        self.output_file_var = tk.StringVar()
        self.output_entry = ttk.Entry(settings_frame, textvariable=self.output_file_var, width=80)
        self.output_entry.grid(row=1, column=1, sticky=tk.EW, padx=5, pady=3)
        self.browse_output_btn = ttk.Button(settings_frame, text="Обзор...", command=self.browse_output)
        self.browse_output_btn.grid(row=1, column=2, padx=5, pady=3)

        ttk.Label(settings_frame, text="Целевой язык:*").grid(row=2, column=0, sticky=tk.W, padx=5, pady=3)
        self.language_var = tk.StringVar(value="Русский")
        languages = ["Русский", "English", "Deutsch", "Français", "Español", "中文", "日本語", "Italiano", "Português", "Українська", "Polski"]
        self.language_combo = ttk.Combobox(settings_frame, textvariable=self.language_var, values=languages, state="readonly", width=18)
        self.language_combo.grid(row=2, column=1, sticky=tk.W, padx=5, pady=3)

        ttk.Label(settings_frame, text="Желаемый стиль:*").grid(row=3, column=0, sticky=tk.NW, padx=5, pady=(5,0))
        self.style_text = tk.Text(settings_frame, height=4, width=80, wrap=tk.WORD,
                                  relief=tk.SOLID, borderwidth=1,
                                  bg=text_bg, fg=text_fg, insertbackground=insert_color,
                                  highlightthickness=1, highlightbackground=text_border,
                                  selectbackground=select_bg, selectforeground=select_fg)
        self.style_text.grid(row=3, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=(5, 5))
        self.style_placeholder = "Пример: Переписать в стиле современного фэнтези, с богатыми описаниями..."

        ttk.Label(settings_frame, text="Цель переписывания:*").grid(row=4, column=0, sticky=tk.NW, padx=5, pady=(5,0))
        self.goal_text = tk.Text(settings_frame, height=4, width=80, wrap=tk.WORD,
                                 relief=tk.SOLID, borderwidth=1,
                                 bg=text_bg, fg=text_fg, insertbackground=insert_color,
                                 highlightthickness=1, highlightbackground=text_border,
                                 selectbackground=select_bg, selectforeground=select_fg)
        self.goal_text.grid(row=4, column=1, columnspan=2, sticky=tk.EW, padx=5, pady=(5, 5))
        self.goal_placeholder = "Пример: Адаптировать текст для молодой аудитории (16-20 лет), упростив сложные обороты..."

        self.setup_placeholder(self.style_text, self.style_placeholder, text_fg)
        self.setup_placeholder(self.goal_text, self.goal_placeholder, text_fg)

        adv_frame = ttk.LabelFrame(settings_frame, text="Расширенные опции")
        adv_frame.grid(row=5, column=0, columnspan=3, sticky=tk.EW, padx=5, pady=(10, 5))
        adv_frame.columnconfigure(1, weight=1)

        adv_info_label_text = (f"Порог схожести: {SIMILARITY_THRESHOLD*100:.0f}% | "
                               f"Целевая длина блока: {BLOCK_TARGET_CHARS_REWRITE} симв.")
        adv_info_label = ttk.Label(adv_frame, text=adv_info_label_text, font=("TkDefaultFont", 8, "italic"))
        adv_info_label.grid(row=0, column=0, columnspan=3, padx=5, pady=(0,5), sticky=tk.W)

        self.resume_var = tk.BooleanVar(value=True)
        self.resume_check = ttk.Checkbutton(adv_frame, text="Возобновить с последнего состояния", variable=self.resume_var)
        self.resume_check.grid(row=1, column=0, padx=5, pady=5, sticky=tk.W)

        ttk.Label(adv_frame, text="Интервал сохр. (блоков, 0=в конце):").grid(row=2, column=0, padx=5, pady=5, sticky=tk.W)
        self.save_interval_var = tk.IntVar(value=1)
        self.save_interval_spin = ttk.Spinbox(adv_frame, from_=0, to=100, textvariable=self.save_interval_var, width=6, state="readonly")
        self.save_interval_spin.grid(row=2, column=1, padx=(5,15), pady=5, sticky=tk.W)

        ttk.Label(adv_frame, text="Модель писателя:").grid(row=3, column=0, padx=5, pady=5, sticky=tk.W)
        self.rewriter_model_var = tk.StringVar(value=REWRITER_MODEL_DEFAULT)
        self.rewriter_combo = ttk.Combobox(adv_frame, textvariable=self.rewriter_model_var, values=AVAILABLE_MODELS, state="readonly", width=25)
        self.rewriter_combo.grid(row=3, column=1, columnspan=2, padx=5, pady=5, sticky=tk.EW)

        control_frame = ttk.Frame(main_frame, padding="5 0 5 0")
        control_frame.pack(pady=5, padx=5, fill=tk.X)

        self.start_button = ttk.Button(control_frame, text="Начать переписывание", command=self.start_rewrite, style="Accent.TButton")
        self.start_button.pack(side=tk.LEFT, padx=(0, 5), ipady=2)

        self.stop_button = ttk.Button(control_frame, text="Стоп", command=self.stop_rewrite, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=5, ipady=2)

        self.status_label_var = tk.StringVar(value="Готов.")
        self.status_label = ttk.Label(control_frame, textvariable=self.status_label_var, anchor=tk.E)
        self.status_label.pack(side=tk.RIGHT, padx=5)

        self.progressbar = ttk.Progressbar(control_frame, orient=tk.HORIZONTAL, length=200, mode='determinate')
        self.progressbar.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(10, 5), ipady=3)

        log_frame = ttk.LabelFrame(main_frame, text="Лог выполнения")
        log_frame.pack(pady=(5, 0), padx=5, fill=tk.BOTH, expand=True)

        self.log_area = scrolledtext.ScrolledText(log_frame, height=15,
                                                 width=100, wrap=tk.WORD, state=tk.DISABLED,
                                                 relief=tk.SOLID, borderwidth=1,
                                                 bg=text_bg, fg=text_fg, insertbackground=insert_color,
                                                 highlightthickness=1, highlightbackground=text_border,
                                                 selectbackground=select_bg, selectforeground=select_fg)
        self.log_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        gui_handler = QueueHandler(log_queue)
        gui_handler.setFormatter(logging.Formatter('%(asctime)s [%(levelname)s] %(message)s', datefmt='%H:%M:%S'))
        logger.addHandler(gui_handler)
        gui_handler.setLevel(logging.INFO)

        self.master.after(100, self.process_log_queue)

        ttk.Label(settings_frame, text="* Обязательные поля", font=("TkDefaultFont", 8, "italic")).grid(row=0, column=3, sticky=tk.E, padx=5)

    def setup_placeholder(self, widget: tk.Text, placeholder: str, normal_fg_color: str):
        self.normal_text_color = normal_fg_color
        widget.bind("<FocusIn>", lambda args, w=widget, p=placeholder: self.clear_placeholder(w, p))
        widget.bind("<FocusOut>", lambda args, w=widget, p=placeholder: self.restore_placeholder(w, p))
        self.restore_placeholder(widget, placeholder)

    def clear_placeholder(self, widget: tk.Text, placeholder: str):
        if widget.cget('foreground') == self.placeholder_color:
             current_text = widget.get("1.0", tk.END).strip()
             if current_text == placeholder:
                 widget.delete("1.0", tk.END)
                 widget.configure(foreground=self.normal_text_color)

    def restore_placeholder(self, widget: tk.Text, placeholder: str):
        current_text = widget.get("1.0", tk.END).strip()
        if not current_text:
            widget.delete("1.0", tk.END)
            widget.insert("1.0", placeholder)
            widget.configure(foreground=self.placeholder_color)

    def browse_input(self):
        filename = filedialog.askopenfilename(
            title="Выберите входной файл",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )
        if filename:
            self.input_file_var.set(filename)
            if not self.output_file_var.get():
                 base, ext = os.path.splitext(filename)
                 suggested_output = f"{base}{FINAL_OUTPUT_SUFFIX}"
                 self.output_file_var.set(suggested_output)

    def browse_output(self):
        initial_name = self.output_file_var.get()
        if not initial_name and self.input_file_var.get():
             base, _ = os.path.splitext(self.input_file_var.get())
             initial_name = f"{base}{FINAL_OUTPUT_SUFFIX}"

        initial_file = os.path.basename(initial_name or f"rewritten_output{FINAL_OUTPUT_SUFFIX}")
        initial_dir = os.path.dirname(initial_name or "")

        filename = filedialog.asksaveasfilename(
            title="Сохранить выходной файл как",
            defaultextension=".txt",
            initialfile=initial_file,
            initialdir=initial_dir or ".",
            filetypes=(("Текстовые файлы", "*.txt"), ("Все файлы", "*.*"))
        )
        if filename:
            self.output_file_var.set(filename)

    def process_log_queue(self):
        try:
            while True:
                record = log_queue.get_nowait()
                self.log_area.configure(state=tk.NORMAL)
                self.log_area.insert(tk.END, record + '\n')
                self.log_area.configure(state=tk.DISABLED)
                self.log_area.yview(tk.END)
        except queue.Empty:
            pass
        finally:
            if self.master.winfo_exists():
                self.master.after(100, self.process_log_queue)

    def update_progress(self, current_block_num, total_blocks):
        if not self.master.winfo_exists(): return

        if total_blocks > 0:
            current_block_num = max(0, min(current_block_num, total_blocks))
            self.progressbar['maximum'] = total_blocks
            self.progressbar['value'] = current_block_num
            percentage = (current_block_num / total_blocks) * 100 if total_blocks > 0 else 0
            status_text = f"Обработка: {current_block_num}/{total_blocks} ({percentage:.1f}%)"
            self.status_label_var.set(status_text)
        else:
             self.progressbar['value'] = 0
             self.progressbar['maximum'] = 1
             if current_block_num == 0 and self.rewrite_thread is not None and self.rewrite_thread.is_alive():
                 status_text = "Определение блоков..."
             else:
                status_text = "Инициализация..." if self.rewrite_thread is not None and self.rewrite_thread.is_alive() else "Готов."
             self.status_label_var.set(status_text)

        self.master.update_idletasks()

    def update_progress_threadsafe(self, current_block_num, total_blocks):
        if self.master.winfo_exists():
            self.master.after(0, self.update_progress, current_block_num, total_blocks)

    def set_ui_state(self, running: bool):
        if not self.master.winfo_exists(): return

        button_entry_check_state = tk.DISABLED if running else tk.NORMAL
        start_button_state = tk.DISABLED if running else tk.NORMAL
        stop_button_state = tk.NORMAL if running else tk.DISABLED
        combo_spin_state = tk.DISABLED if running else 'readonly'
        text_widget_state = tk.DISABLED if running else tk.NORMAL

        self.input_entry.configure(state=button_entry_check_state)
        self.output_entry.configure(state=button_entry_check_state)
        self.browse_input_btn.configure(state=button_entry_check_state)
        self.browse_output_btn.configure(state=button_entry_check_state)
        self.language_combo.configure(state=combo_spin_state)
        self.style_text.configure(state=text_widget_state)
        self.goal_text.configure(state=text_widget_state)
        self.resume_check.configure(state=button_entry_check_state)
        self.save_interval_spin.configure(state=combo_spin_state)
        self.rewriter_combo.configure(state=combo_spin_state)
        self.start_button.configure(state=start_button_state)
        self.stop_button.configure(state=stop_button_state)

        if not running:
            if hasattr(self, 'style_placeholder'):
                self.restore_placeholder(self.style_text, self.style_placeholder)
            if hasattr(self, 'goal_placeholder'):
                self.restore_placeholder(self.goal_text, self.goal_placeholder)

    def start_rewrite(self):
        input_file = self.input_file_var.get().strip()
        output_file = self.output_file_var.get().strip()
        style = self.style_text.get("1.0", tk.END).strip()
        goal = self.goal_text.get("1.0", tk.END).strip()

        if hasattr(self, 'style_placeholder') and style == self.style_placeholder:
            style = ""
        if hasattr(self, 'goal_placeholder') and goal == self.goal_placeholder:
            goal = ""

        errors = []
        if not input_file:
            errors.append("- Входной файл не указан.")
        elif not os.path.exists(input_file):
            errors.append(f"- Входной файл не найден:\n  {input_file}")
        elif not os.path.isfile(input_file):
            errors.append(f"- Указанный входной путь не является файлом:\n  {input_file}")

        if not output_file:
            errors.append("- Выходной файл не указан.")
        else:
             output_dir = os.path.dirname(output_file) or '.'
             try:
                 os.makedirs(output_dir, exist_ok=True)
                 test_file = os.path.join(output_dir, f".__write_test_{os.getpid()}.tmp")
                 with open(test_file, "w") as f:
                     f.write("test")
                 os.remove(test_file)
             except OSError as e:
                  errors.append(f"- Выходная директория недоступна для записи:\n  '{output_dir}'\n  Ошибка: {e}")
             except Exception as e:
                 errors.append(f"- Ошибка при проверке/создании выходной директории:\n  '{output_dir}'\n  Ошибка: {e}")

             if not errors and os.path.exists(output_file):
                  if not os.access(output_file, os.W_OK):
                       errors.append(f"- Нет прав на перезапись существующего выходного файла:\n  {output_file}")

        if not style:
            errors.append("- Желаемый стиль не указан.")
        if not goal:
            errors.append("- Цель переписывания не указана.")

        if errors:
            messagebox.showerror("Ошибка ввода", "Пожалуйста, исправьте ошибки:\n\n" + "\n".join(errors))
            return

        try:
            configure_gemini()
        except ValueError as e:
             messagebox.showerror("Ошибка API Key", f"Не удалось сконфигурировать Gemini API:\n{e}\n\nПроверьте переменную окружения GOOGLE_API_KEY или файл .env.")
             return
        except Exception as e:
             messagebox.showerror("Ошибка конфигурации Gemini", f"Непредвиденная ошибка при конфигурации Gemini:\n{e}")
             logger.error(f"Неожиданная ошибка конфигурации Gemini: {e}", exc_info=True)
             return

        params = {
            'input_file': input_file,
            'output_file': output_file,
            'language': self.language_var.get(),
            'style': style,
            'goal': goal,
            'rewriter_model': self.rewriter_model_var.get(),
            'resume': self.resume_var.get(),
            'save_interval': self.save_interval_var.get(),
        }

        self.log_area.configure(state=tk.NORMAL)
        self.log_area.delete('1.0', tk.END)
        self.log_area.configure(state=tk.DISABLED)

        self.stop_event.clear()
        self.set_ui_state(running=True)
        self.status_label_var.set("Запуск...")
        self.progressbar['value'] = 0
        self.progressbar['maximum'] = 1
        self.progressbar['mode'] = 'indeterminate'
        self.master.update_idletasks()

        logger.info("="*20 + " Запуск новой задачи переписывания " + "="*20)
        log_params = {k:v for k,v in params.items() if k not in ['style', 'goal']}
        try:
            logger.info(f"Параметры: {json.dumps(log_params, indent=2, ensure_ascii=False, default=str)}")
        except TypeError:
            logger.info(f"Параметры (fallback): {log_params}")
        logger.info(f"Стиль (начало): {style[:150]}...")
        logger.info(f"Цель (начало): {goal[:150]}...")

        self.rewrite_thread = threading.Thread(
            target=self.run_rewrite_thread_wrapper,
            args=(params,),
            daemon=True
        )
        self.rewrite_thread.start()
        self.master.after(500, lambda: setattr(self.progressbar, 'mode', 'determinate'))

    def run_rewrite_thread_wrapper(self, params):
        start_time = time.time()
        try:
            run_rewrite_process(params, self.update_progress_threadsafe, self.stop_event)
        except Exception as e:
            error_message_full = f"КРИТИЧЕСКАЯ НЕОБРАБОТАННАЯ ОШИБКА в потоке: {type(e).__name__}: {e}"
            logging.critical(error_message_full, exc_info=True)
            error_message_for_box = f"Произошла непредвиденная критическая ошибка:\n{type(e).__name__}: {e}\n\nПроцесс остановлен. Проверьте лог."
            if self.master.winfo_exists():
                self.master.after(0, lambda msg=error_message_for_box: messagebox.showerror(
                    "Критическая ошибка потока", msg
                ))
        finally:
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"--- Поток переписывания завершил работу за {duration:.2f} сек. ---")
            if self.master.winfo_exists():
                self.master.after(0, self.on_rewrite_finish)

    def on_rewrite_finish(self):
        if not self.master.winfo_exists():
             logging.info("Переписывание завершено, но окно GUI было закрыто.")
             return

        self.set_ui_state(running=False)

        final_progress = self.progressbar['value']
        max_progress = self.progressbar['maximum']
        final_status_message = "Статус неопределен."

        if self.stop_event.is_set():
             final_status_message = "Остановлено пользователем."
             messagebox.showwarning("Остановлено", "Процесс переписывания был остановлен.")
        elif max_progress > 0 and final_progress >= max_progress :
             final_status_message = "Завершено успешно!"
             messagebox.showinfo("Завершено", "Переписывание книги успешно завершено!")
        elif max_progress <= 1 and final_progress == 0 :
             final_status_message = "Ошибка инициализации или нет блоков."
             messagebox.showerror("Ошибка", "Процесс завершился без обработки блоков. Проверьте лог.")
        else:
             final_status_message = "Завершено (не полностью)."
             messagebox.showwarning("Завершено с проблемами", "Переписывание завершено, но не все блоки могли быть обработаны.\nПроверьте лог и финальный статус в файле состояния.")

        self.status_label_var.set(final_status_message)
        if "успешно" in final_status_message.lower() and max_progress > 0:
             self.progressbar['value'] = max_progress
        self.rewrite_thread = None

    def stop_rewrite(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
            logging.warning("Запрос на остановку процесса...")
            self.status_label_var.set("Остановка...")
            self.master.update_idletasks()
            self.stop_event.set()
            self.stop_button.configure(state=tk.DISABLED)
        else:
            logging.info("Запрос на остановку, но процесс не запущен или уже завершается.")
            self.stop_button.configure(state=tk.DISABLED)

    def on_closing(self):
        if self.rewrite_thread and self.rewrite_thread.is_alive():
             if messagebox.askyesno("Подтверждение выхода",
                                    "Процесс переписывания еще выполняется.\n"
                                    "Остановить его и выйти?\n\n"
                                    "Промежуточный результат и состояние будут сохранены (если возможно)."):
                 self.stop_rewrite()
                 self.master.after(200, self.master.destroy)
                 logging.info("Окно закрывается, отправлен сигнал остановки процессу.")
             else:
                 return
        else:
             logging.info("Окно закрывается.")
             self.master.destroy()

if __name__ == "__main__":
    log_filename = "book_rewriter_app.log"
    log_level_file = logging.DEBUG
    log_level_gui = logging.INFO

    try:
        file_handler = RotatingFileHandler(log_filename, maxBytes=5*1024*1024, backupCount=5, encoding='utf-8', delay=True)
        file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] %(filename)s:%(lineno)d - %(message)s'))
        logger.addHandler(file_handler)
        logger.setLevel(log_level_file)
        logging.info("\n" + "="*30 + f" Запуск приложения Book Rewriter {time.strftime('%Y-%m-%d %H:%M:%S')} " + "="*30)
    except ImportError:
         logging.warning("Модуль logging.handlers не найден, используется простой FileHandler без ротации.")
         try:
             file_handler = logging.FileHandler(log_filename, encoding='utf-8', mode='a')
             file_handler.setFormatter(logging.Formatter('%(asctime)s - %(levelname)s - [%(threadName)s] %(filename)s:%(lineno)d - %(message)s'))
             logger.addHandler(file_handler)
             logger.setLevel(log_level_file)
             logging.info("\n" + "="*30 + f" Запуск приложения Book Rewriter {time.strftime('%Y-%m-%d %H:%M:%S')} " + "="*30)
         except Exception as e_log:
              print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить логирование в файл: {e_log}")
    except Exception as e_log_setup:
        print(f"КРИТИЧЕСКАЯ ОШИБКА: Не удалось настроить логирование: {e_log_setup}")

    root = tk.Tk()
    sv_ttk.set_theme("dark")
    try:
        app = BookRewriterApp(root)
        root.mainloop()
    except Exception as e_main:
         logging.critical(f"КРИТИЧЕСКАЯ ОШИБКА в главном цикле Tkinter или инициализации GUI: {e_main}", exc_info=True)
         try:
             if tk.Toplevel.winfo_exists(root):
                 messagebox.showerror("Критическая ошибка приложения", f"Произошла критическая ошибка:\n{e_main}\n\nПриложение будет закрыто. См. лог-файл '{log_filename}'.")
         except Exception as e_msgbox:
             print(f"CRITICAL ERROR (cannot show messagebox: {e_msgbox}): {e_main}. See log file '{log_filename}'.")
    finally:
        logging.info("="*30 + " Приложение завершено " + "="*30 + "\n")