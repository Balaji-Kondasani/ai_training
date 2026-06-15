from sqlalchemy import Column, Integer, String, Float
from sqlalchemy.ext.declarative import declarative_base

Base = declarative_base()

class PredictionLog(Base):
    __tablename__ = "prediction_logs"
    
    id = Column(Integer, primary_key=True, index=True)
    model_name = Column(String, index=True, nullable=False)
    input_features = Column(String, nullable=False)
    prediction = Column(Integer, nullable=False)
    probability = Column(Float, nullable=False)
    
    # We add a 'status' column to demonstrate table schema changes and migration history
    status = Column(String, default="completed", nullable=True)
