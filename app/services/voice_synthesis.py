
import re
from pathlib import Path
import subprocess
import time
from app.utils.logger import logger

MODEL_NAME = "F5TTS_v1_Base"

REF_MIN_SECONDS = 5.0
REF_MAX_SECONDS = 12.0

# English speech rate used to size each clip.
WORDS_PER_SEC_FAST = 3.5   # fastest we allow (floor on duration)
WORDS_PER_SEC_SLOW = 1.8   # slowest we allow (never stretch past this)

SEED = 1234     # same seed for every line keeps the voice consistent
NFE_STEP = 16   # F5 default is 32; 16 is ~2x faster on CPU with small quality loss

_f5 = None


def get_f5():
    """Load F5-TTS once. The CLI reloaded the model for every segment
    (~6 minutes each in your logs); the Python API loads it a single time."""
    global _f5
    if _f5 is None:
        from f5_tts.api import F5TTS
        _f5 = F5TTS(model=MODEL_NAME)
        logger.info("F5-TTS model loaded successfully.")
    return _f5


def get_duration(path):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration",
         "-of", "default=noprint_wrappers=1:nokey=1", str(path)],
        check=True, capture_output=True, text=True,
    )
    return float(r.stdout.strip())


def build_reference_clip(reference_audio, reference_segments, out_path):
    """Cut a 5-12 s reference clip and return (path, matching text)."""
    chosen, total = [], 0.0

    def speech_rate(seg):
        dur = float(seg["end"]) - float(seg["start"])
        return len(seg["text"].split()) / dur if dur > 0 else 0

    # Skip segments that are mostly silence (a couple of words spread over
    # several seconds) - they make a poor voice reference.
    candidates = [s for s in reference_segments if speech_rate(s) >= 1.0]
    if not candidates:
        candidates = list(reference_segments)

    for seg in candidates:
        dur = float(seg["end"]) - float(seg["start"])
        if dur <= 0:
            continue
        if not chosen:
            chosen, total = [seg], dur
        elif total + dur <= REF_MAX_SECONDS and (
            float(seg["start"]) - float(chosen[-1]["end"]) < 1.5
        ):
            chosen.append(seg)
            total += dur
        else:
            break
        if total >= REF_MIN_SECONDS:
            break

    if not chosen:
        raise RuntimeError("No usable segments for reference audio.")

    start = float(chosen[0]["start"])
    length = min(float(chosen[-1]["end"]) - start, REF_MAX_SECONDS)
    text = " ".join(s["text"].strip() for s in chosen)

    subprocess.run(
        ["ffmpeg", "-v", "error", "-ss", f"{start:.3f}", "-t", f"{length:.3f}",
         "-i", str(reference_audio), "-ac", "1",
         # F5-TTS itself cuts silences > 1 s out of the reference. Doing it
         # here means the reference length we measure is the length it uses
         # (otherwise fix_duration is off by the amount it removed).
         "-af", "silenceremove=stop_periods=-1:stop_duration=0.4:"
                "stop_threshold=-40dB",
         "-y", str(out_path)],
        check=True,
    )
    logger.info(f"Reference clip: {start:.2f}s +{length:.1f}s")
    return out_path, text


def count_syllables(text):
    """Rough English syllable count (vowel groups): a better proxy for how
    long a line takes to say than the number of words."""
    total = 0
    for w in re.findall(r"[a-zA-Z']+", text.lower()):
        n = len(re.findall(r"[aeiouy]+", w))
        if w.endswith("e") and n > 1 and not w.endswith(("le", "ee")):
            n -= 1
        total += max(n, 1)
    return max(total, 1)


# How far a line's speaking rate may stray from the scene's typical rate.
# Small band = no sudden jumps between very slow and very fast lines.
RATE_BAND = 0.12


def plan_durations(translated_segments):
    """
    Pick a duration for every line so the voice keeps a steady pace.

    Each line would ideally fill its slot, but doing that exactly makes the
    rate swing (2 to 3.5 words/s) between lines. Instead:
      1. find the scene's typical rate (median over all lines),
      2. let each line deviate from it by at most +-RATE_BAND,
      3. smooth neighbouring lines so the pace changes gradually.
    A line that is too long for its slot spills into the pause after it;
    one that is too short just ends a little early.
    """
    items = []
    for seg in translated_segments:
        text = seg["text"].strip()
        slot = float(seg["end"]) - float(seg["start"])
        if text and slot > 0:
            items.append((count_syllables(text), slot))
        else:
            items.append(None)

    rates = sorted(s / d for s, d in (i for i in items if i))
    if not rates:
        return [0.8] * len(items)
    typical = rates[len(rates) // 2]
    lo, hi = typical * (1 - RATE_BAND), typical * (1 + RATE_BAND)

    wanted = [min(max(i[0] / i[1], lo), hi) if i else typical for i in items]

    # light smoothing across neighbours (keeps changes gradual)
    smooth = []
    for k, r in enumerate(wanted):
        prev_r = wanted[k - 1] if k > 0 else r
        next_r = wanted[k + 1] if k + 1 < len(wanted) else r
        smooth.append(0.25 * prev_r + 0.5 * r + 0.25 * next_r)

    return [
        max(i[0] / r, 0.8) if i else 0.8
        for i, r in zip(items, smooth)
    ]


def target_duration(text, slot):
    """Duration to ask F5-TTS for: the slot, bounded by a natural speech rate."""
    words = max(len(text.split()), 1)
    fastest = words / WORDS_PER_SEC_FAST
    slowest = words / WORDS_PER_SEC_SLOW
    return max(min(slot, slowest), fastest, 0.8)


def generate_speech(
    video_id,
    reference_audio,
    reference_text,
    reference_segments,
    translated_segments,
    output_dir,
):
    logger.info(f"Starting voice generation for video_id={video_id}")
    start_time = time.time()

    output_dir = Path(output_dir)
    segments_dir = output_dir / f"{video_id}_voice_segments"
    segments_dir.mkdir(parents=True, exist_ok=True)

    ref_clip, ref_text = build_reference_clip(
        reference_audio, reference_segments, segments_dir / "reference.wav"
    )
    f5 = get_f5()

    # F5-TTS trims silence from the reference before using it, so its real
    # length is shorter than the clip we cut. fix_duration counts the REAL
    # reference length, so using the raw clip length made every line about
    # 3 s too long (then it was sped up to fit). Measure the processed clip.
    from f5_tts.infer.utils_infer import preprocess_ref_audio_text
    processed_ref, _ = preprocess_ref_audio_text(
        str(ref_clip), ref_text, show_info=lambda *a, **k: None
    )
    ref_dur = get_duration(processed_ref)
    logger.info(f"Reference length: cut={get_duration(ref_clip):.2f}s, "
                f"used by F5-TTS={ref_dur:.2f}s")
    planned = plan_durations(translated_segments)
    generated_segments = []

    for index, segment in enumerate(translated_segments):
        text = segment["text"].strip()
        if not text:
            continue

        slot = float(segment["end"]) - float(segment["start"])
        gen_dur = planned[index]
        out_file = segments_dir / f"segment_{index}.wav"

        logger.info(
            f"Segment {index}: slot={slot:.2f}s -> requesting {gen_dur:.2f}s "
            f"for {text!r}"
        )

        # fix_duration = reference + generated audio, in seconds.
        f5.infer(
            ref_file=str(ref_clip),
            ref_text=ref_text,
            gen_text=text,
            file_wave=str(out_file),
            fix_duration=ref_dur + gen_dur,
            remove_silence=False,
            nfe_step=NFE_STEP,
            seed=SEED,
        )
        logger.info(
            f"Segment {index}: requested {gen_dur:.2f}s, "
            f"generated {get_duration(out_file):.2f}s"
        )

        generated_segments.append({
            "index": index,
            "start": segment["start"],
            "end": segment["end"],
            "text": text,
            "audio_path": str(out_file),
        })

    processing_time = round(time.time() - start_time, 2)
    logger.info(f"Voice generation completed ({processing_time}s)")

    return {
        "generated_segments": generated_segments,
        "tts_model": MODEL_NAME,
        "processing_time": processing_time,
    }