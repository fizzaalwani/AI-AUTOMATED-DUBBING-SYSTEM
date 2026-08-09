from uuid import UUID
from datetime import datetime
from pydantic import BaseModel, ConfigDict

#video schemas
class VideoCreate(BaseModel):
    original_filename: str
    stored_filename: str

class VideoResponse(BaseModel):
    video_id: UUID
    original_filename: str
    stored_filename: str
    audio_path: str
    status: str
    processing_time: float
    uploaded_at: datetime
    model_config = ConfigDict(from_attributes=True)

class VideoUploadResponse(BaseModel):
    message: str
    video: VideoResponse

#transcript schemas
class TranscriptCreate(BaseModel):
    whisper_model: str
    language: str
    transcript: str
    processing_time: float

class TranscriptResponse(BaseModel):
    transcript_id: UUID
    video_id: UUID
    whisper_model: str
    language: str
    transcript: str
    segments: list
    processing_time: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

#translation schemas
class TranslationCreate(BaseModel):
    translation_model: str
    model_version: str
    source_language: str
    target_language: str
    translated_text: str
    processing_time: float

class TranslationResponse(BaseModel):
    translation_id: UUID
    video_id: UUID
    translation_model: str
    source_language: str
    target_language: str
    translated_text: str
    translated_segments: list
    translated_segments: list
    processing_time: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)

#voice generation schemas
class VoiceSynthesisCreate(BaseModel):
    tts_model: str
    voice_clone_model: str
    model_version: str
    reference_audio_path: str
    generated_audio_path: str
    processing_time: float

class VoiceSynthesisResponse(BaseModel):
    voice_id: UUID
    video_id: UUID
    tts_model: str
    voice_clone_model: str
    generated_text: str
    reference_audio_path: str
    generated_audio_path: str
    dubbed_video_path: str
    processing_time: float
    created_at: datetime
    model_config = ConfigDict(from_attributes=True)
