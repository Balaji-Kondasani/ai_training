import os
from fastapi import FastAPI, UploadFile, File, HTTPException, status
from fastapi.staticfiles import StaticFiles

app = FastAPI(
    title="Day 16: FastAPI File Uploads & Static Files",
    description="Covers how to accept file uploads (e.g., dataset CSVs) and serve static files.",
    version="1.0.0"
)

# Directory to save uploaded files
UPLOAD_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "static_uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount the static directory so files can be accessed via URL
# e.g., http://127.0.0.1:8000/static/my_dataset.csv
app.mount("/static", StaticFiles(directory=UPLOAD_DIR), name="static")

@app.post("/upload-dataset")
async def upload_dataset(file: UploadFile = File(..., description="The CSV dataset file to upload")):
    """
    Endpoint demonstrating File Uploads.
    Validates that the file uploaded is a CSV before saving it to the static upload folder.
    """
    # 1. Validate file extension
    if not file.filename.endswith(".csv"):
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Invalid file type. Only CSV files (.csv) are allowed."
        )
        
    # 2. Define target save path
    save_path = os.path.join(UPLOAD_DIR, file.filename)
    
    # 3. Read and write file contents asynchronously
    try:
        content = await file.read()
        with open(save_path, "wb") as f:
            f.write(content)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Failed to save file: {str(e)}"
        )
        
    # 4. Return file information and public access URL
    return {
        "filename": file.filename,
        "content_type": file.content_type,
        "size_bytes": len(content),
        "access_url": f"/static/{file.filename}"
    }

if __name__ == "__main__":
    import uvicorn
    # Test file upload with curl:
    # curl -X POST -F "file=@diabetes.csv" http://127.0.0.1:8000/upload-dataset
    uvicorn.run(app, host="127.0.0.1", port=8000)
