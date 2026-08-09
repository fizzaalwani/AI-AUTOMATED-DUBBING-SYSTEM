import subprocess
import time
from pathlib import Path
from app.utils.logger import logger

def dub_video(video_path, dubbed_audio_path, output_dir):
    logger.info("Dubbed video generation started.")
    start_time = time.time()
    output_video = output_dir / f"{Path(video_path).stem}_dubbed.mp4"

    command = [
        "ffmpeg",
        "-i", str(video_path),
        "-i", str(dubbed_audio_path),
        "-map", "0:v:0",
        "-map", "1:a:0",
        "-c:v", "copy",
        "-c:a", "aac",
        "-y",
        str(output_video)
    ]

    try:
        logger.info("Merging generated audio with original video using FFmpeg.")
        subprocess.run(command, check=True)
        processing_time = round(time.time() - start_time, 2)
        logger.info(f"Dubbed video generated successfully: {output_video} ({processing_time}s)")

    except subprocess.CalledProcessError as e:
            logger.error(f"Video dubbing failed: {e}")
            raise
    
    return {"dubbed_video": output_video, "processing_time": processing_time}
