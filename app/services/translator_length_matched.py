import time
import torch
from app.utils.logger import logger

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = None
model = None

# Duration-aware translation settings
WORDS_PER_SEC = 2.8      # natural English speech rate used to size each line
NUM_CANDIDATES = 8       # translations generated per segment
SCORE_TOLERANCE = 0.6    # how much worse than the best translation we accept
                         # (average log-probability) in exchange for a better fit


def load_model():
    global tokenizer
    global model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        logger.info("NLLB model loaded successfully.")


def _generate(text, max_length, num_return):
    inputs = tokenizer(text, return_tensors="pt")
    with torch.inference_mode():
        out = model.generate(
            **inputs,
            forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
            max_length=max_length,
            num_beams=max(num_return, 5),
            num_return_sequences=num_return,
            output_scores=True,
            return_dict_in_generate=True,
        )
    texts = tokenizer.batch_decode(out.sequences, skip_special_tokens=True)
    scores = out.sequences_scores.tolist()
    return texts, scores


def translate_to_fit(text, slot_seconds):
    """
    Translate one segment and pick, among the model's best candidates, the
    one whose word count best matches what the speaker can say in the same
    time. Meaning stays close to the best translation (SCORE_TOLERANCE).
    """
    texts, scores = _generate(text, max_length=256, num_return=NUM_CANDIDATES)

    target_words = max(1, round(slot_seconds * WORDS_PER_SEC))
    best_score = max(scores)

    seen = set()
    candidates = []
    for t, s in zip(texts, scores):
        t = t.strip()
        if not t or t in seen:
            continue
        seen.add(t)
        if s >= best_score - SCORE_TOLERANCE:
            candidates.append((t, s))

    # closest to the target length, ties broken by model score
    chosen = min(
        candidates,
        key=lambda c: (abs(len(c[0].split()) - target_words), -c[1]),
    )[0]

    logger.info(
        f"Length-matched translation: slot={slot_seconds:.2f}s "
        f"target~{target_words} words, got {len(chosen.split())}: {chosen!r}"
    )
    return chosen


def translate(text, segments=None):
    load_model()
    logger.info("Translation started using NLLB model.")

    try:
        start_time = time.time()
        tokenizer.src_lang = "urd_Arab"

        # Full text (kept for the transcript / UI)
        full, _ = _generate(text, max_length=512, num_return=1)
        translated_text = full[0]

        translated_segments = []
        if segments:
            for segment in segments:
                slot = float(segment["end"]) - float(segment["start"])
                translated_segments.append({
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": translate_to_fit(segment["text"], slot),
                })

        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Translation completed successfully ({processing_time}s)")

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise

    return {
        "translated_text": translated_text,
        "translated_segments": translated_segments,
        "source_language": "Urdu",
        "target_language": "English",
        "model": MODEL_NAME,
        "processing_time": processing_time,
    }
