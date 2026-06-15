import time
import asyncio
from fastapi import FastAPI, BackgroundTasks

app = FastAPI(
    title="Day 15: FastAPI Background Tasks & Async Code",
    description="Covers how to run slow background processes without blocking client responses, using async/await.",
    version="1.0.0"
)

# Simulated training log file
LOG_FILE = "background_train_log.txt"

def simulate_heavy_training(model_name: str, epochs: int):
    """
    A standard CPU-bound or slow synchronous function that simulates model training.
    Since it runs in BackgroundTasks, FastAPI will execute it in a threadpool so it doesn't block the main event loop.
    """
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Started training {model_name} for {epochs} epochs...\n")
        
    for epoch in range(1, epochs + 1):
        time.sleep(1)  # Simulate 1 second of computation per epoch
        with open(LOG_FILE, "a") as f:
            f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {model_name} - Epoch {epoch}/{epochs} complete\n")
            
    with open(LOG_FILE, "a") as f:
        f.write(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] Training complete for {model_name}!\n\n")

@app.post("/train-bg")
async def trigger_training(model_name: str, epochs: int, background_tasks: BackgroundTasks):
    """
    Triggers model training in the background.
    The client receives a response immediately, while the training runs in the background.
    """
    # Queue the heavy training function to run in the background
    background_tasks.add_task(simulate_heavy_training, model_name, epochs)
    
    return {
        "status": "Accepted",
        "message": f"Training job for '{model_name}' has been queued in the background.",
        "check_logs_at": "/logs"
    }

@app.get("/logs")
async def read_logs():
    """
    Asynchronously reads the training log file using non-blocking I/O.
    """
    import os
    if not os.path.exists(LOG_FILE):
        return {"logs": "No background training logs found yet. Trigger a job first."}
        
    # Read logs (for simple demonstration, we read using standard open, in production you could use aiofiles)
    with open(LOG_FILE, "r") as f:
        lines = f.readlines()
        
    return {
        "logs": lines[-15:]  # Return the last 15 log lines
    }

if __name__ == "__main__":
    import uvicorn
    # Test triggering background task:
    # curl -X POST "http://127.0.0.1:8000/train-bg?model_name=RandomForest&epochs=5"
    uvicorn.run(app, host="127.0.0.1", port=8000)
