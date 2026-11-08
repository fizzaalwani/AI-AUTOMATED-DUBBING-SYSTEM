# import time
# import torch
# from app.utils.logger import logger

# from transformers import AutoTokenizer
# from transformers import AutoModelForSeq2SeqLM

# MODEL_NAME = "facebook/nllb-200-distilled-600M"

# tokenizer = None
# model = None

# def load_model():
#     global tokenizer
#     global model

#     if tokenizer is None or model is None:
#         tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
#         model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
#         logger.info("NLLB model loaded successfully.")

# def translate(text, segments=None):
#     load_model()
#     logger.info("Translation started using NLLB model.")

#     try:
#         start_time = time.time()
#         tokenizer.src_lang = "urd_Arab"
#         inputs = tokenizer(text,return_tensors="pt")

#         with torch.inference_mode():
#             translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids( "eng_Latn"),
#                 max_length=512, num_beams=5)

#         translated_text = tokenizer.batch_decode( translated_tokens, skip_special_tokens=True)[0]

#         translated_segments = []
#         if segments:
#             for segment in segments:
#                 inputs = tokenizer(segment["text"], return_tensors="pt")
#                 with torch.inference_mode():
#                     translated_tokens = model.generate(
#                         **inputs,
#                         forced_bos_token_id=tokenizer.convert_tokens_to_ids("eng_Latn"),
#                         max_length=256,
#                         num_beams=5
#                     )

#                 translated_segment = tokenizer.batch_decode(translated_tokens, skip_special_tokens=True)[0]
#                 translated_segments.append({
#                     "start": segment["start"],
#                     "end": segment["end"],
#                     "text": translated_segment
#                 })
#         processing_time = round( time.time() - start_time, 2)
#         logger.info(f"Translation completed successfully ({processing_time}s)")

#     except Exception as e:
#         logger.error(f"Error during translation: {e}")
#         raise

#     return {
#         "translated_text": translated_text,
#         "translated_segments": translated_segments,
#         "source_language": "Urdu",
#         "target_language": "English",
#         "model": MODEL_NAME,
#         "processing_time": processing_time
#     }

"""
Free Urdu -> English translation.

Provider order:
  1. Google Translate through the `deep-translator` package (free, no API
     key, needs internet). Much better than NLLB-600M on conversational
     Urdu: tenses, idioms, "لگتا" = "feel", mixed Urdu/English, etc.
  2. NLLB (local, offline) if Google is unreachable.

Setup:
    pip install deep-translator
Optional (use a bigger, better local model for the NLLB fallback):
    set NLLB_MODEL=facebook/nllb-200-distilled-1.3B

Note: deep-translator uses Google's public web endpoint, which is fine for
a project / demo but not an official, supported API for production use.
"""
import gc
import os
import time

from app.utils.logger import logger

NLLB_MODEL = os.getenv("NLLB_MODEL", "facebook/nllb-200-distilled-600M")
MODEL_NAME = NLLB_MODEL

WORDS_PER_SEC = 2.8
FIT_TOLERANCE = 0.35      # keep the best translation if within +-35% of target
SCORE_TOLERANCE = 0.25    # otherwise accept a slightly worse one, no more

tokenizer = None
model = None


# ----------------------------------------------------------------- Google
def _google_translate(texts):
    """Google allows ~5 requests/second, so go one at a time with a short
    pause, and back off if it still says "too many requests"."""
    from deep_translator import GoogleTranslator

    translator = GoogleTranslator(source="ur", target="en")
    out = []
    for t in texts:
        for attempt in range(5):
            try:
                out.append((translator.translate(t) or "").strip())
                break
            except Exception:
                if attempt == 4:
                    raise
                time.sleep(2 * (attempt + 1))
        time.sleep(0.5)
    return out


# ------------------------------------------------------------------- NLLB
def load_model():
    global tokenizer, model
    if tokenizer is None or model is None:
        from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
        import torch
        tokenizer = AutoTokenizer.from_pretrained(NLLB_MODEL)
        # bfloat16 halves memory (the 1.3B model needs ~5 GB in float32,
        # ~2.7 GB in bfloat16) - important with your paging-file limit.
        dtype = torch.bfloat16 if os.getenv("NLLB_BF16", "1") == "1" else torch.float32
        model = AutoModelForSeq2SeqLM.from_pretrained(
            NLLB_MODEL, torch_dtype=dtype, low_cpu_mem_usage=True
        )
        logger.info(f"NLLB model loaded successfully ({NLLB_MODEL}).")


def unload_model():
    global tokenizer, model
    tokenizer = None
    model = None
    gc.collect()


def _nllb_generate(text, max_length, num_return):
    import torch
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
    return (tokenizer.batch_decode(out.sequences, skip_special_tokens=True),
            out.sequences_scores.tolist())


def _nllb_fit(text, slot):
    """Best translation; only swap to a different candidate if the best one
    is far too long / short for the slot AND the swap costs little accuracy."""
    texts, scores = _nllb_generate(text, 256, 8)
    target = max(1, round(slot * WORDS_PER_SEC))
    best = texts[0].strip()
    if abs(len(best.split()) - target) <= max(1, FIT_TOLERANCE * target):
        return best
    top = max(scores)
    pool = [(t.strip(), s) for t, s in zip(texts, scores)
            if t.strip() and s >= top - SCORE_TOLERANCE]
    return min(pool, key=lambda c: (abs(len(c[0].split()) - target), -c[1]))[0]


def _nllb_translate(text, segments):
    load_model()
    tokenizer.src_lang = "urd_Arab"
    full = _nllb_generate(text, 512, 1)[0][0]
    segs = [{
        "start": s["start"], "end": s["end"],
        "text": _nllb_fit(s["text"], float(s["end"]) - float(s["start"])),
    } for s in (segments or [])]
    return full, segs, NLLB_MODEL


# ------------------------------------------------------------------ main
def translate(text, segments=None):
    logger.info("Translation started.")
    start_time = time.time()

    try:
        segments = segments or []
        lines = [s["text"] for s in segments]
        full, *seg_lines = _google_translate([text] + lines)
        translated_segments = [
            {"start": s["start"], "end": s["end"], "text": t}
            for s, t in zip(segments, seg_lines)
        ]
        used = "Google Translate (deep-translator)"
    except Exception as e:
        logger.warning(f"Google translation unavailable ({e}); using NLLB.")
        full, translated_segments, used = _nllb_translate(text, segments)

    for s, t in zip(segments, translated_segments):
        logger.info(f"Urdu: {s['text']!r} -> English: {t['text']!r}")

    processing_time = round(time.time() - start_time, 2)
    logger.info(f"Translation completed successfully ({processing_time}s)")

    return {
        "translated_text": full,
        "translated_segments": translated_segments,
        "source_language": "Urdu",
        "target_language": "English",
        "model": used,
        "processing_time": processing_time,
    }