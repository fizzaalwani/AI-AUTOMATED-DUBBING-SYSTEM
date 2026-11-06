# import uuid
# from sqlalchemy import Column, Float, String, Text, DateTime, ForeignKey
# from sqlalchemy.dialects.postgresql import UUID
# from sqlalchemy import JSON
# from sqlalchemy.orm import relationship
# from sqlalchemy.sql import func
# from app.database.database import Base

# class Video(Base):
#     __tablename__ = "videos"

#     video_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     original_filename = Column(String(255), nullable=False)
#     stored_filename = Column(String(255), nullable=False)
#     audio_path = Column(String(500))
#     status = Column(String(50), default="Uploaded")
#     uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

#     transcript = relationship("Transcript", back_populates="video", uselist=False, cascade="all, delete-orphan")
#     translation = relationship("Translation", back_populates="video", uselist=False, cascade="all, delete-orphan")
#     voice_synthesis = relationship("VoiceSynthesis", back_populates="video", uselist=False, cascade="all, delete-orphan")

# class Transcript(Base):
#     __tablename__ = "transcripts"

#     transcript_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
#     whisper_model = Column(String(100), nullable=False)
#     language = Column(String(20))
#     transcript = Column(Text, nullable=False)
#     segments = Column(JSON, nullable=True)
#     processing_time = Column(Float)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())

#     video = relationship("Video", back_populates="transcript")

# class Translation(Base):
#     __tablename__ = "translations"

#     translation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
#     translation_model = Column(String(100), nullable=False)
#     source_language = Column(String(20))
#     target_language = Column(String(20))
#     translated_text = Column(Text)
#     translated_segments = Column(JSON, nullable=True)
#     processing_time = Column(Float)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())

#     video = relationship("Video", back_populates="translation")

# class VoiceSynthesis(Base):
#     __tablename__ = "voice_synthesis"

#     voice_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
#     video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
#     tts_model = Column(String(100), nullable=False)
#     voice_clone_model = Column(String(100))
#     generated_text = Column(Text)
#     reference_audio_path = Column(String(500))
#     generated_audio_path = Column(String(500))
#     dubbed_video_path = Column(String(500))
#     processing_time = Column(Float)
#     created_at = Column(DateTime(timezone=True), server_default=func.now())

#     video = relationship("Video", back_populates="voice_synthesis")
import uuid
from sqlalchemy import Column, Float, String, Text, DateTime, ForeignKey
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy import JSON
from sqlalchemy.orm import relationship
from sqlalchemy.sql import func
from app.database.database import Base

class Video(Base):
    __tablename__ = "videos"

    video_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    original_filename = Column(String(255), nullable=False)
    stored_filename = Column(String(255), nullable=False)
    audio_path = Column(String(500))
    status = Column(String(50), default="Uploaded")
    uploaded_at = Column(DateTime(timezone=True), server_default=func.now())

    transcript = relationship("Transcript", back_populates="video", uselist=False, cascade="all, delete-orphan")
    translation = relationship("Translation", back_populates="video", uselist=False, cascade="all, delete-orphan")
    voice_synthesis = relationship("VoiceSynthesis", back_populates="video", uselist=False, cascade="all, delete-orphan")

class Transcript(Base):
    __tablename__ = "transcripts"

    transcript_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
    whisper_model = Column(String(100), nullable=False)
    language = Column(String(20))
    transcript = Column(Text, nullable=False)
    # list of {"start", "end", "text"} (later also "speaker")
    segments = Column(JSON, nullable=True)
    processing_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="transcript")

class Translation(Base):
    __tablename__ = "translations"

    translation_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
    translation_model = Column(String(100), nullable=False)
    source_language = Column(String(20))
    target_language = Column(String(20))
    translated_text = Column(Text)
    translated_segments = Column(JSON, nullable=True)
    processing_time = Column(Float)
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="translation")

class VoiceSynthesis(Base):
    __tablename__ = "voice_synthesis"

    voice_id = Column(UUID(as_uuid=True), primary_key=True, default=uuid.uuid4)
    video_id = Column(UUID(as_uuid=True), ForeignKey("videos.video_id", ondelete="CASCADE"), nullable=False)
    tts_model = Column(String(100), nullable=False)
    voice_clone_model = Column(String(100))
    generated_text = Column(Text)
    reference_audio_path = Column(String(500))
    generated_audio_path = Column(String(500))   # final mix (dub + background)
    dubbed_video_path = Column(String(500))
    processing_time = Column(Float)              # voice generation time
    video_processing_time = Column(Float)        # NEW: final video merge time
    created_at = Column(DateTime(timezone=True), server_default=func.now())

    video = relationship("Video", back_populates="voice_synthesis")