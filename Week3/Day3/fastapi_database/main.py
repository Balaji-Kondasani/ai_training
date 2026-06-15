from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_model_raster_from_canvas  # Wait, standard SQLAlchemy elements are Column, Integer, String, Float, create_engine
from sqlalchemy import Column, Integer, String, Float, create_engine
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Generator

# --- SQLAlchemy Setup ---
SQLALCHEMY_DATABASE_URL = "sqlite:///./prediction_audit.db"

# Create Engine and SessionLocal
# `connect_args={"check_same_thread": False}` is needed only for SQLite
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Base class for DB models
Base = declarative_base()

# --- DB Model ---
class DBModelPredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True)
    input_features = Column(String)  # Stored as string representation
    prediction = Column(Integer)
    probability = Column(Float)

# Create database tables
Base.metadata.create_all(bind=engine)

# --- FastAPI App ---
app = FastAPI(
    title="Day 17: FastAPI Database Integration (SQLAlchemy & SQLite)",
    description="Shows how to persist API transactions (like prediction logs) in a SQLite database using SQLAlchemy ORM.",
    version="1.0.0"
)

# --- Pydantic Schemas ---
class PredictionLogCreate(BaseModel):
    model_name: str = Field(..., description="Name of the model run")
    input_features: str = Field(..., description="String representing raw features")
    prediction: int = Field(..., description="Model prediction class")
    probability: float = Field(..., description="Model prediction probability")

class PredictionLogResponse(BaseModel):
    id: int
    model_name: str
    input_features: str
    prediction: int
    probability: float

    class Config:
        from_attributes = True

# --- DB Session Dependency ---
def get_db() -> Generator[Session, None, None]:
    """
    Dependency that yields a database session and closes it once the request terminates.
    """
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoints ---

@app.post("/logs", response_model=PredictionLogResponse, status_code=status.HTTP_201_CREATED)
def create_log(log_in: PredictionLogCreate, db: Session = Depends(get_db)):
    """
    Inserts a new prediction log into the SQLite database.
    """
    db_log = DBModelPredictionLog(
        model_name=log_in.model_name,
        input_features=log_in.input_features,
        prediction=log_in.prediction,
        probability=log_in.probability
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@app.get("/logs", response_model=List[PredictionLogResponse])
def read_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    """
    Retrieves logged predictions from the SQLite database with pagination support.
    """
    logs = db.query(DBModelPredictionLog).offset(skip).limit(limit).all()
    return logs

if __name__ == "__main__":
    import uvicorn
    # Start app
    uvicorn.run(app, host="127.0.0.1", port=8000)
