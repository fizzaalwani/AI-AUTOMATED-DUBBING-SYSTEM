import subprocess
import shutil
import time
from app.utils.logger import logger

MODEL_NAME = "F5TTS_v1_Base"

def generate_speech( video_id, reference_audio, reference_text, generation_text, output_dir):
    logger.info(f"Starting voice generation for video_id={video_id}")

    start_time = time.time()
    cli_path = shutil.which("f5-tts_infer-cli")
    generated_audio = output_dir / f"{video_id}_generated.wav"

    command = [
        cli_path,
        "--model",
        MODEL_NAME,
        "--ref_audio",
        str(reference_audio),
        "--ref_text",
        reference_text,
        "--gen_text",
        generation_text,
        "--output_file",
        str(generated_audio)
    ]

    try:
        logger.info(f"Running F5-TTS model: {MODEL_NAME}")
        subprocess.run(command, check=True)

        processing_time = round( time.time() - start_time, 2)
        logger.info(f"Voice generated successfully: {generated_audio} ({processing_time}s)")

    except subprocess.CalledProcessError as e:
        logger.error(f"Voice generation failed: {e}")
        raise RuntimeError(f"Voice generation failed: {e}")
    
    return {
        "generated_audio": str(generated_audio),
        "tts_model": MODEL_NAME,
        "processing_time": processing_time
    }
