import time
import torch
from app.utils.logger import logger

from transformers import AutoTokenizer
from transformers import AutoModelForSeq2SeqLM

MODEL_NAME = "facebook/nllb-200-distilled-600M"

tokenizer = None
model = None

def load_model():
    global tokenizer
    global model

    if tokenizer is None or model is None:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
        logger.info("NLLB model loaded successfully.")

def translate(text):
    load_model()
    logger.info("Translation started using NLLB model.")

    try:
        start_time = time.time()
        tokenizer.src_lang = "urd_Arab"
        inputs = tokenizer(text,return_tensors="pt")

        with torch.inference_mode():
            translated_tokens = model.generate(**inputs, forced_bos_token_id=tokenizer.convert_tokens_to_ids( "eng_Latn"),
                max_length=512, num_beams=5)

        translated_text = tokenizer.batch_decode( translated_tokens, skip_special_tokens=True)[0]
        processing_time = round( time.time() - start_time, 2)
        logger.info(f"Translation completed successfully ({processing_time}s)")

    except Exception as e:
        logger.error(f"Error during translation: {e}")
        raise

    return {
        "translated_text": translated_text,
        "source_language": "Urdu",
        "target_language": "English",
        "model": MODEL_NAME,
        "processing_time": processing_time
    }