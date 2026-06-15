from fastapi import FastAPI, Depends, HTTPException, status
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from pydantic import BaseModel, Field
from typing import List, Generator

# Import models from our separate file
from .models import Base, PredictionLog

# Database configuration
SQLALCHEMY_DATABASE_URL = "sqlite:///./prediction_audit.db"
engine = create_engine(SQLALCHEMY_DATABASE_URL, connect_args={"check_same_thread": False})
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

app = FastAPI(
    title="Day 18: Database Migrations with Alembic",
    description="Implements DB transaction logging using SQLAlchemy models where tables are fully managed via Alembic migrations.",
    version="1.0.0"
)

# --- Pydantic Schemas ---
class PredictionLogCreate(BaseModel):
    model_name: str = Field(..., description="Name of the model run")
    input_features: str = Field(..., description="Raw input features")
    prediction: int = Field(..., description="Predicted output class")
    probability: float = Field(..., description="Confidence probability")
    status: str = Field(default="completed", description="Execution status")

class PredictionLogResponse(BaseModel):
    id: int
    model_name: str
    input_features: str
    prediction: int
    probability: float
    status: str

    class Config:
        from_attributes = True

# --- DB Dependency ---
def get_db() -> Generator[Session, None, None]:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()

# --- Endpoints ---
@app.post("/logs", response_model=PredictionLogResponse, status_code=status.HTTP_201_CREATED)
def create_log(log_in: PredictionLogCreate, db: Session = Depends(get_db)):
    """
    Saves a log record in the database.
    Note: Base.metadata.create_all(bind=engine) is NOT called on startup.
    The database tables are set up using Alembic migration files instead.
    """
    db_log = PredictionLog(
        model_name=log_in.model_name,
        input_features=log_in.input_features,
        prediction=log_in.prediction,
        probability=log_in.probability,
        status=log_in.status
    )
    db.add(db_log)
    db.commit()
    db.refresh(db_log)
    return db_log

@app.get("/logs", response_model=List[PredictionLogResponse])
def read_logs(skip: int = 0, limit: int = 100, db: Session = Depends(get_db)):
    logs = db.query(PredictionLog).offset(skip).limit(limit).all()
    return logs

if __name__ == "__main__":
    import uvicorn
    # Start server
    uvicorn.run(app, host="127.0.0.1", port=8000)
