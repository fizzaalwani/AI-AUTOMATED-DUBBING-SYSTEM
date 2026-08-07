from fastapi import APIRouter
from fastapi import UploadFile
from fastapi import File
from fastapi import Depends
from fastapi import HTTPException

from sqlalchemy.orm import Session

from app.database.database import get_db
from app.services.pipeline import process_pipeline
from app.utils.logger import logger


router = APIRouter( prefix="/pipeline", tags=["Pipeline"])

@router.post("/process")
def pipeline(file: UploadFile = File(...), db: Session = Depends(get_db)):
    logger.info("Pipeline request received.")
    try:
        result = process_pipeline(file, db)
        logger.info("Pipeline completed successfully.")
        return result
    except Exception as e:
        logger.error(f"Error occurred while processing pipeline: {str(e)}")
        db.rollback()
        raise HTTPException(status_code=500, detail=str(e))