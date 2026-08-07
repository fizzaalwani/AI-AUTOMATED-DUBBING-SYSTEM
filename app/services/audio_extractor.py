import subprocess
import time
from app.utils.logger import logger

def extract_audio(video_path, video_id, output_dir):
    logger.info(f"Audio extraction started for video_id={video_id}")
    start_time = time.time()
    audio_filename = f"{video_id}.wav"
    audio_path = output_dir / audio_filename

    command = [
        "ffmpeg",
        "-i",
        str(video_path),
        "-vn",
        "-acodec",
        "pcm_s16le",
        "-ar",
        "16000",
        "-ac",
        "1",
        "-y",
        str(audio_path)
    ]
    try:
        subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Audio extracted successfully: {audio_path} ({processing_time}s)")

    except subprocess.CalledProcessError as e:
        logger.error(f"Error occurred while extracting audio for video_id={video_id}: {e}")
        raise

    return {"audio_path": audio_path, "processing_time": processing_time}