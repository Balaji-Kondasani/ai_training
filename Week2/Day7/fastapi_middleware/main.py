import time
from fastapi import FastAPI, Depends, Header, HTTPException, Request
from typing import Dict, Generator

app = FastAPI(
    title="Day 12: FastAPI Dependency Injection & Middleware",
    description="Demonstrates how to write reusable dependencies and intercept requests with custom middleware.",
    version="1.0.0"
)

# --- Custom Middleware ---
# Middleware intercepts requests before they hit the endpoints, and updates responses after they execute.
@app.middleware("http")
async def add_process_time_header(request: Request, call_next):
    start_time = time.time()
    
    # Process the request and get response
    response = await call_next(request)
    
    process_time = time.time() - start_time
    # Add custom header showing processing duration
    response.headers["X-Process-Time"] = f"{process_time:.6f}s"
    print(f"Request {request.url.path} processed in {process_time:.6f} seconds")
    return response

# --- Mock DB Connection dependency ---
def get_db() -> Generator[Dict[str, str], None, None]:
    """
    Simulated Database session manager.
    FastAPI handles setting up the session and tearing it down after execution.
    """
    db_session = {"connection": "active", "db": "ml_metadata_store"}
    print("[DB Dependency] Session opened")
    try:
        yield db_session
    finally:
        print("[DB Dependency] Session closed")

# --- Header verification dependency ---
def verify_api_key(x_api_key: str = Header(..., description="API Key for accessing this route")):
    """
    Checks if a required header is present and valid.
    """
    if x_api_key != "secret-ml-key":
        raise HTTPException(
            status_code=403, 
            detail="Forbidden: Invalid X-API-Key header"
        )
    return x_api_key

# --- Endpoints using Dependencies ---

@app.get("/db-status")
def read_db_status(db: dict = Depends(get_db)):
    """
    Retrieves status using the get_db dependency.
    """
    return {
        "status": "connected",
        "database_info": db
    }

@app.get("/secure-data", dependencies=[Depends(verify_api_key)])
def read_secure_data():
    """
    Secures this endpoint using verify_api_key dependency.
    If the X-API-Key header is missing or incorrect, it returns HTTP 403 before executing this function.
    """
    return {
        "secret_data": "Random Forest is less prone to overfitting than Decision Trees."
    }

if __name__ == "__main__":
    import uvicorn
    # Test curl for secure-data:
    # curl -H "X-API-Key: secret-ml-key" http://127.0.0.1:8000/secure-data
    uvicorn.run(app, host="127.0.0.1", port=8000)
