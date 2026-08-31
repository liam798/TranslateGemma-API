import ast
import concurrent.futures
import os
import queue
import re
import threading
import time
from collections import deque
from dataclasses import dataclass
from typing import Deque, Dict, List, Optional, Tuple, Union

import torch
from fastapi import FastAPI, HTTPException, Query, Request
from jinja2.exceptions import UndefinedError
from pydantic import BaseModel, Field
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
from transformers.utils import logging as transformers_logging


transformers_logging.set_verbosity_error()

MODEL_DIR = os.environ.get("TRANSLATE_GEMMA_MODEL_DIR", "/data/translate-gemma/model")
API_KEY = os.environ.get("TRANSLATE_GEMMA_API_KEY", "")
MAX_NEW_TOKENS = int(os.environ.get("TRANSLATE_GEMMA_MAX_NEW_TOKENS", "2048"))
CHUNK_CHARS = int(os.environ.get("TRANSLATE_GEMMA_CHUNK_CHARS", "900"))
CHUNK_MAX_NEW_TOKENS = int(os.environ.get("TRANSLATE_GEMMA_CHUNK_MAX_NEW_TOKENS", "768"))
MAX_BATCH_SIZE = int(os.environ.get("TRANSLATE_GEMMA_MAX_BATCH_SIZE", "4"))
LONG_BATCH_SIZE = int(os.environ.get("TRANSLATE_GEMMA_LONG_BATCH_SIZE", "2"))
LOCK_WAIT_SECONDS = float(os.environ.get("TRANSLATE_GEMMA_LOCK_WAIT_SECONDS", "3"))
BATCH_WAIT_SECONDS = float(os.environ.get("TRANSLATE_GEMMA_BATCH_WAIT_SECONDS", "0.02"))
BATCH_MIN_WAIT_SECONDS = float(os.environ.get("TRANSLATE_GEMMA_BATCH_MIN_WAIT_SECONDS", "0.005"))
BATCH_MAX_CHARS = int(os.environ.get("TRANSLATE_GEMMA_BATCH_MAX_CHARS", "6000"))
QUEUE_MAX_SIZE = int(os.environ.get("TRANSLATE_GEMMA_QUEUE_MAX_SIZE", "256"))
QUEUE_RESULT_TIMEOUT_SECONDS = float(os.environ.get("TRANSLATE_GEMMA_QUEUE_RESULT_TIMEOUT_SECONDS", "120"))

FALLBACK_LANGUAGES: Dict[str, str] = {
    "en": "English",
    "zh-Hans": "Chinese",
    "zh-Hant": "Chinese",
    "zh-TW": "Chinese",
    "zh": "Chinese",
    "es": "Spanish",
    "ja": "Japanese",
    "vi": "Vietnamese",
}
SUPPORTED_LANGUAGES: Dict[str, str] = dict(FALLBACK_LANGUAGES)

MODEL_LANGUAGE_ALIASES: Dict[str, str] = {
    "zh": "zh-Hans",
    "zh-CHS": "zh-Hans",
    "zh-CN": "zh-Hans",
    "zh-CHT": "zh-TW",
    "fil": "tl",
}

PROTECTED_TOKEN_RE = re.compile(r"(\{\{[^{}\n]{1,80}\}\}|<[^<>\n]{1,80}>|https?://\S+)")


class GoogleTranslateV2Request(BaseModel):
    q: Optional[Union[str, List[str]]] = Field(None, description="Text to translate")
    source: Optional[str] = Field(None, description="Source language code")
    target: Optional[str] = Field(None, description="Target language code")
    text: Optional[Union[str, List[str]]] = Field(None, description="CustomAPI text field")
    from_: Optional[str] = Field(None, alias="from", description="CustomAPI source language code")
    to: Optional[str] = Field(None, description="CustomAPI target language code")
    format: str = Field("text", description="Compatibility field")
    key: str = Field("", description="Google-style API key")
    api_key: str = Field("", description="Compatibility API key")
    platform: Optional[str] = Field(None, description="Compatibility field")


app = FastAPI(title="TranslateGemma Translation API")

processor = None
model = None
generate_lock = threading.Lock()
worker_started = False
worker_local = threading.local()
stats_lock = threading.Lock()
stats: Dict[str, Union[int, float]] = {
    "requests_total": 0,
    "jobs_total": 0,
    "batches_total": 0,
    "batch_items_total": 0,
    "batch_chars_total": 0,
    "generate_seconds_total": 0.0,
    "errors_total": 0,
    "queue_full_total": 0,
    "queue_timeout_total": 0,
}


@dataclass
class TranslationJob:
    text: str
    source: str
    target: str
    future: concurrent.futures.Future
    created_at: float


translation_queue = queue.Queue(maxsize=QUEUE_MAX_SIZE)


def increment_stat(name: str, amount: Union[int, float] = 1) -> None:
    with stats_lock:
        stats[name] = stats.get(name, 0) + amount


def snapshot_stats() -> Dict[str, Union[int, float]]:
    with stats_lock:
        current = dict(stats)
    current["queue_size"] = translation_queue.qsize()
    return current


def normalize_lang(code: str) -> str:
    return SUPPORTED_LANGUAGES.get(code, code)


def normalize_model_lang(code: str) -> str:
    return MODEL_LANGUAGE_ALIASES.get(code, code)


def parse_template_languages(chat_template: str) -> Dict[str, str]:
    lines = chat_template.splitlines()
    start = None
    end = None
    for index, line in enumerate(lines):
        if line.strip().startswith("{%- set languages = {"):
            start = index
            continue
        if start is not None and line.strip() == "-%}":
            end = index
            break

    if start is None or end is None:
        return {}

    block = "\n".join(lines[start:end])
    block = block.replace("{%- set languages = ", "", 1).strip()
    parsed = ast.literal_eval(block)
    if not isinstance(parsed, dict):
        return {}

    languages: Dict[str, str] = {}
    for code, name in parsed.items():
        if isinstance(code, str) and isinstance(name, str):
            languages[code] = name
    return languages


def ensure_supported_language(code: str) -> str:
    normalized = normalize_model_lang(code)
    if normalized not in SUPPORTED_LANGUAGES:
        raise HTTPException(status_code=400, detail=f"Unsupported language for TranslateGemma: {code}")
    return normalized


def detect_source_lang(text: str, target: str) -> str:
    if re.search(r"[\u3040-\u30ff]", text):
        return "ja"
    if re.search(r"[\u4e00-\u9fff]", text):
        return "zh-Hans"

    lowered = f" {text.lower()} "
    spanish_markers = (
        " el ",
        " la ",
        " los ",
        " las ",
        " de ",
        " que ",
        " una ",
        " para ",
        " hola ",
        " gracias ",
    )
    if re.search(r"[áéíóúñü¿¡]", lowered) or any(marker in lowered for marker in spanish_markers):
        return "es"

    if target != "en":
        return "en"
    return "en"


def require_api_key(api_key: str) -> None:
    if not API_KEY:
        raise HTTPException(status_code=500, detail="Server API key is not configured")
    if api_key != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid API key")


def require_any_api_key(*api_keys: Optional[str]) -> None:
    for api_key in api_keys:
        if api_key:
            require_api_key(api_key)
            return
    require_api_key("")


def get_request_api_key(request: Request) -> str:
    return request.headers.get("x-api-key", "") or request.headers.get("authorization", "").removeprefix("Bearer ").strip()


def load_model() -> None:
    global processor, model, SUPPORTED_LANGUAGES
    if model is not None:
        return

    quantization_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16,
    )

    processor = AutoProcessor.from_pretrained(MODEL_DIR, local_files_only=True)
    template_languages = parse_template_languages(getattr(processor, "chat_template", "") or "")
    if template_languages:
        SUPPORTED_LANGUAGES = template_languages
    model = AutoModelForImageTextToText.from_pretrained(
        MODEL_DIR,
        local_files_only=True,
        device_map="auto",
        torch_dtype=torch.bfloat16,
        quantization_config=quantization_config,
    )
    model.eval()


@app.on_event("startup")
def startup() -> None:
    load_model()
    start_batch_worker()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok", "model": "google/translategemma-4b-it"}


@app.get("/metrics")
def metrics() -> Dict[str, Union[int, float]]:
    current = snapshot_stats()
    batches = int(current.get("batches_total", 0))
    if batches:
        current["batch_items_avg"] = round(float(current.get("batch_items_total", 0)) / batches, 3)
        current["batch_chars_avg"] = round(float(current.get("batch_chars_total", 0)) / batches, 3)
        current["generate_seconds_avg"] = round(float(current.get("generate_seconds_total", 0.0)) / batches, 3)
    else:
        current["batch_items_avg"] = 0
        current["batch_chars_avg"] = 0
        current["generate_seconds_avg"] = 0
    return current


@app.get("/languages")
def languages() -> List[Dict[str, object]]:
    codes = sorted(SUPPORTED_LANGUAGES)
    return [
        {
            "code": code,
            "name": SUPPORTED_LANGUAGES[code],
            "targets": [target for target in codes if target != code],
        }
        for code in codes
    ]


def estimate_max_new_tokens(text: str) -> int:
    if len(text) <= CHUNK_CHARS:
        return min(MAX_NEW_TOKENS, max(128, min(CHUNK_MAX_NEW_TOKENS, len(text) + 96)))
    return MAX_NEW_TOKENS


def batched(items: List[str], batch_size: int) -> List[List[str]]:
    return [items[index : index + batch_size] for index in range(0, len(items), max(1, batch_size))]


def split_long_text(text: str, max_chars: int = CHUNK_CHARS) -> List[str]:
    if len(text) <= max_chars:
        return [text]

    chunks: List[str] = []
    current = ""
    parts = re.split(r"(\n{2,})", text)

    def flush_current() -> None:
        nonlocal current
        if current:
            chunks.append(current)
            current = ""

    def append_piece(piece: str) -> None:
        nonlocal current
        if not piece:
            return
        if len(piece) > max_chars:
            flush_current()
            sentences = re.split(r"(?<=[.!?。！？])\s+", piece)
            sentence_buffer = ""
            for sentence in sentences:
                if len(sentence) > max_chars:
                    if sentence_buffer:
                        chunks.append(sentence_buffer)
                        sentence_buffer = ""
                    for index in range(0, len(sentence), max_chars):
                        chunks.append(sentence[index : index + max_chars])
                    continue
                if sentence_buffer and len(sentence_buffer) + 1 + len(sentence) > max_chars:
                    chunks.append(sentence_buffer)
                    sentence_buffer = sentence
                else:
                    sentence_buffer = f"{sentence_buffer} {sentence}".strip() if sentence_buffer else sentence
            if sentence_buffer:
                chunks.append(sentence_buffer)
            return
        if current and len(current) + len(piece) > max_chars:
            chunks.append(current)
            current = piece
        else:
            current += piece

    for part in parts:
        append_piece(part)
    flush_current()
    return chunks or [text]


def protect_tokens(text: str) -> Tuple[str, Dict[str, str]]:
    replacements: Dict[str, str] = {}

    def replace(match: re.Match[str]) -> str:
        placeholder = f"ZXQKEEP{len(replacements)}QXZ"
        replacements[placeholder] = match.group(0)
        return placeholder

    return PROTECTED_TOKEN_RE.sub(replace, text), replacements


def restore_tokens(text: str, replacements: Dict[str, str]) -> str:
    restored = text
    for placeholder, original in replacements.items():
        restored = restored.replace(placeholder, original)
        restored = restored.replace(placeholder.lower(), original)

    missing_tokens = [original for original in replacements.values() if original not in restored]
    structural_tokens = [
        token
        for token in missing_tokens
        if token.startswith("<") or (token.startswith("{{") and token.endswith("}}"))
    ]
    if structural_tokens:
        restored = "\n".join(structural_tokens + [restored])
    return restored


def start_batch_worker() -> None:
    global worker_started
    if worker_started:
        return
    worker_started = True
    thread = threading.Thread(target=batch_worker_loop, name="translate-gemma-batcher", daemon=True)
    thread.start()


def length_bucket(text: str) -> str:
    size = len(text)
    if size <= 200:
        return "short"
    if size <= CHUNK_CHARS:
        return "medium"
    return "long"


def bucket_priority(bucket: str) -> int:
    if bucket == "short":
        return 0
    if bucket == "medium":
        return 1
    return 2


def batch_key(job: TranslationJob) -> Tuple[str, str, str]:
    return job.source, job.target, length_bucket(job.text)


def drain_ready_jobs(pending_jobs: Deque[TranslationJob], wait_seconds: float = 0.0) -> None:
    deadline = time.monotonic() + wait_seconds
    while True:
        timeout = max(0.0, deadline - time.monotonic())
        try:
            if timeout > 0:
                pending_jobs.append(translation_queue.get(timeout=timeout))
            else:
                pending_jobs.append(translation_queue.get_nowait())
        except queue.Empty:
            break
        if timeout <= 0:
            continue


def pop_next_pending(pending_jobs: Deque[TranslationJob]) -> TranslationJob:
    best_index = 0
    best_score: Tuple[int, float] = (bucket_priority(length_bucket(pending_jobs[0].text)), pending_jobs[0].created_at)
    for index, job in enumerate(pending_jobs):
        score = (bucket_priority(length_bucket(job.text)), job.created_at)
        if score < best_score:
            best_index = index
            best_score = score

    for _ in range(best_index):
        pending_jobs.append(pending_jobs.popleft())
    return pending_jobs.popleft()


def choose_batch_wait(first_job: TranslationJob, pending_count: int) -> float:
    queued_count = translation_queue.qsize() + pending_count
    if queued_count >= MAX_BATCH_SIZE:
        return BATCH_MIN_WAIT_SECONDS
    if length_bucket(first_job.text) == "long":
        return BATCH_MIN_WAIT_SECONDS
    return BATCH_WAIT_SECONDS


def pop_compatible_pending(
    pending_jobs: Deque[TranslationJob],
    key: Tuple[str, str, str],
    batch_chars: int,
) -> Optional[TranslationJob]:
    for _ in range(len(pending_jobs)):
        job = pending_jobs.popleft()
        if batch_key(job) == key and batch_chars + len(job.text) <= BATCH_MAX_CHARS:
            return job
        pending_jobs.append(job)
    return None


def collect_batch(first_job: TranslationJob, pending_jobs: Deque[TranslationJob]) -> List[TranslationJob]:
    jobs = [first_job]
    batch_chars = len(first_job.text)
    key = batch_key(first_job)
    deadline = time.monotonic() + choose_batch_wait(first_job, len(pending_jobs))
    while len(jobs) < MAX_BATCH_SIZE:
        next_job = pop_compatible_pending(pending_jobs, key, batch_chars)
        if next_job is None:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                break
            try:
                next_job = translation_queue.get(timeout=remaining)
            except queue.Empty:
                break
            if batch_key(next_job) != key or batch_chars + len(next_job.text) > BATCH_MAX_CHARS:
                pending_jobs.append(next_job)
                continue
        jobs.append(next_job)
        batch_chars += len(next_job.text)
    return jobs


def batch_worker_loop() -> None:
    worker_local.in_worker = True
    pending_jobs: Deque[TranslationJob] = deque()
    while True:
        if not pending_jobs:
            pending_jobs.append(translation_queue.get())
            drain_ready_jobs(pending_jobs, BATCH_MIN_WAIT_SECONDS)
        else:
            drain_ready_jobs(pending_jobs)
        first_job = pop_next_pending(pending_jobs)
        jobs = collect_batch(first_job, pending_jobs)
        source, target, _ = batch_key(first_job)
        start_time = time.monotonic()
        try:
            results = translate_batch_direct([job.text for job in jobs], source, target)
            elapsed = time.monotonic() - start_time
            with stats_lock:
                stats["batches_total"] = stats.get("batches_total", 0) + 1
                stats["batch_items_total"] = stats.get("batch_items_total", 0) + len(jobs)
                stats["batch_chars_total"] = stats.get("batch_chars_total", 0) + sum(len(job.text) for job in jobs)
                stats["generate_seconds_total"] = stats.get("generate_seconds_total", 0.0) + elapsed
            for job, result in zip(jobs, results):
                job.future.set_result(result)
        except Exception as exc:
            increment_stat("errors_total")
            for job in jobs:
                job.future.set_exception(exc)
        finally:
            for _ in jobs:
                translation_queue.task_done()


def submit_translation_jobs(texts: List[str], source: str, target: str) -> List[str]:
    jobs: List[TranslationJob] = []
    increment_stat("requests_total")
    for text in texts:
        future = concurrent.futures.Future()
        job = TranslationJob(text=text, source=source, target=target, future=future, created_at=time.monotonic())
        try:
            translation_queue.put(job, timeout=LOCK_WAIT_SECONDS)
        except queue.Full as exc:
            increment_stat("queue_full_total")
            raise HTTPException(status_code=503, detail="TranslateGemma queue is full, retry later") from exc
        jobs.append(job)
    increment_stat("jobs_total", len(jobs))

    results: List[str] = []
    for job in jobs:
        try:
            results.append(job.future.result(timeout=QUEUE_RESULT_TIMEOUT_SECONDS))
        except concurrent.futures.TimeoutError as exc:
            increment_stat("queue_timeout_total")
            raise HTTPException(status_code=503, detail="TranslateGemma queue wait timed out, retry later") from exc
    return results


def build_message(text: str, source: str, target: str) -> List[Dict[str, object]]:
    return [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "source_lang_code": source,
                    "target_lang_code": target,
                    "text": text,
                }
            ],
        }
    ]


def translate_batch_once(texts: List[str], source: str, target: str) -> List[str]:
    if getattr(worker_local, "in_worker", False):
        return translate_batch_direct(texts, source, target)
    return submit_translation_jobs(texts, source, target)


def translate_batch_direct(texts: List[str], source: str, target: str) -> List[str]:
    source_code = ensure_supported_language(source)
    target_code = ensure_supported_language(target)
    protected: List[str] = []
    replacement_maps: List[Dict[str, str]] = []
    for text in texts:
        protected_text, replacements = protect_tokens(text)
        protected.append(protected_text)
        replacement_maps.append(replacements)

    messages = [build_message(text, source_code, target_code) for text in protected]

    try:
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
            processor_kwargs={"padding": True},
        ).to(model.device)
    except UndefinedError as exc:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported source or target language for TranslateGemma: {source_code}->{target_code}",
        ) from exc

    input_len = inputs["input_ids"].shape[-1]
    max_new_tokens = max(estimate_max_new_tokens(text) for text in texts)
    lock_acquired = generate_lock.acquire(timeout=1)
    if not lock_acquired:
        raise HTTPException(status_code=503, detail="TranslateGemma internal worker is busy, retry later")
    try:
        with torch.inference_mode():
            outputs = model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                do_sample=False,
                use_cache=True,
            )
    finally:
        generate_lock.release()

    generated = outputs[:, input_len:]
    translated = processor.batch_decode(generated, skip_special_tokens=True)
    return [restore_tokens(text.strip(), replacement_maps[index]) for index, text in enumerate(translated)]


def translate_one(text: str, source: str, target: str) -> str:
    return translate_batch_once([text], source, target)[0]


def translate_text(text: str, source: str, target: str) -> str:
    chunks = split_long_text(text)
    if len(chunks) == 1:
        return translate_one(text, source, target)
    translated_chunks: List[str] = []
    for chunk_batch in batched(chunks, LONG_BATCH_SIZE):
        translated_chunks.extend(translate_batch_once(chunk_batch, source, target))
    return "\n\n".join(translated_chunks)


def translate_many(
    raw_texts: Union[str, List[str]],
    raw_source: Optional[str],
    raw_target: Optional[str],
) -> Tuple[List[str], List[str], List[str]]:
    if raw_target is None:
        raise HTTPException(status_code=400, detail="Invalid request: missing target or to parameter")

    source_code = normalize_model_lang(raw_source or "auto")
    target_code = ensure_supported_language(raw_target)
    texts = raw_texts if isinstance(raw_texts, list) else [raw_texts]

    translated: List[str] = []
    detected_sources = []
    pending_by_source: Dict[str, List[Tuple[int, str]]] = {}
    for index, text in enumerate(texts):
        detected_source = detect_source_lang(text, target_code) if source_code == "auto" else ensure_supported_language(source_code)
        detected_sources.append(detected_source)
        pending_by_source.setdefault(detected_source, []).append((index, text))

    translated = [""] * len(texts)
    for detected_source, indexed_texts in pending_by_source.items():
        current_batch: List[Tuple[int, str]] = []
        for index, text in indexed_texts:
            if len(split_long_text(text)) > 1:
                for batch_index, batch_text in current_batch:
                    translated[batch_index] = translate_text(batch_text, detected_source, target_code)
                current_batch = []
                translated[index] = translate_text(text, detected_source, target_code)
                continue

            current_batch.append((index, text))
            if len(current_batch) >= MAX_BATCH_SIZE:
                batch_indexes = [batch_index for batch_index, _ in current_batch]
                batch_texts = [batch_text for _, batch_text in current_batch]
                for batch_index, translated_text in zip(batch_indexes, translate_batch_once(batch_texts, detected_source, target_code)):
                    translated[batch_index] = translated_text
                current_batch = []

        if current_batch:
            batch_indexes = [batch_index for batch_index, _ in current_batch]
            batch_texts = [batch_text for _, batch_text in current_batch]
            for batch_index, translated_text in zip(batch_indexes, translate_batch_once(batch_texts, detected_source, target_code)):
                translated[batch_index] = translated_text

    return texts, translated, detected_sources


@app.post("/language/translate/v2")
def google_translate_v2_post(req: GoogleTranslateV2Request, request: Request) -> Dict[str, object]:
    require_any_api_key(req.key, req.api_key, get_request_api_key(request))
    raw_texts = req.q if req.q is not None else req.text
    raw_source = req.source if req.source is not None else req.from_
    raw_target = req.target if req.target is not None else req.to
    if raw_texts is None:
        raise HTTPException(status_code=400, detail="Invalid request: missing q or text parameter")

    _, translated, detected_sources = translate_many(raw_texts, raw_source, raw_target)
    translations = []
    for index, translated_text in enumerate(translated):
        translations.append(
            {
                "translatedText": translated_text,
                "detectedSourceLanguage": detected_sources[index],
            }
        )
    return {"data": {"translations": translations}}


@app.get("/language/translate/v2")
def google_translate_v2_get(
    request: Request,
    q: List[str] = Query(...),
    target: str = Query(...),
    source: Optional[str] = Query(None),
    key: str = Query(""),
    api_key: str = Query(""),
    format: str = Query("text"),
) -> Dict[str, object]:
    del format
    require_any_api_key(key, api_key, get_request_api_key(request))
    _, translated, detected_sources = translate_many(q, source, target)
    translations = []
    for index, translated_text in enumerate(translated):
        translations.append(
            {
                "translatedText": translated_text,
                "detectedSourceLanguage": detected_sources[index],
            }
        )
    return {"data": {"translations": translations}}
