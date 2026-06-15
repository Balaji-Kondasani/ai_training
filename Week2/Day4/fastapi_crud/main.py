from fastapi import FastAPI, HTTPException, status, Response
from pydantic import BaseModel, Field
from typing import Dict, List, Optional

app = FastAPI(
    title="Day 7: FastAPI CRUD API",
    description="A complete RESTful API implementing GET, POST, PUT, and DELETE methods.",
    version="1.0.0"
)

# --- Schemas ---
class TaskItem(BaseModel):
    title: str = Field(..., min_length=1, description="Title of the task")
    description: Optional[str] = Field(None, description="Detailed description")
    completed: bool = Field(default=False, description="Completion status")

class TaskUpdate(BaseModel):
    title: Optional[str] = Field(None, min_length=1)
    description: Optional[str] = None
    completed: Optional[bool] = None

class TaskResponse(BaseModel):
    id: int
    title: str
    description: Optional[str] = None
    completed: bool

# --- Mock Database ---
DB: Dict[int, dict] = {
    1: {"id": 1, "title": "Setup virtual environment", "description": "Initialize using uv venv", "completed": True},
    2: {"id": 2, "title": "Train Logistic Regression", "description": "Complete Day 2 ML models", "completed": False}
}
id_counter = 3

# --- CRUD Operations ---

# CREATE (POST)
@app.post("/tasks", response_model=TaskResponse, status_code=status.HTTP_201_CREATED)
def create_task(task: TaskItem):
    global id_counter
    new_task = {
        "id": id_counter,
        "title": task.title,
        "description": task.description,
        "completed": task.completed
    }
    DB[id_counter] = new_task
    id_counter += 1
    return new_task

# READ All (GET)
@app.get("/tasks", response_model=List[TaskResponse])
def get_all_tasks():
    return list(DB.values())

# READ One (GET)
@app.get("/tasks/{task_id}", response_model=TaskResponse)
def get_task(task_id: int):
    if task_id not in DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )
    return DB[task_id]

# UPDATE (PUT)
@app.put("/tasks/{task_id}", response_model=TaskResponse)
def update_task(task_id: int, task_update: TaskUpdate):
    if task_id not in DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )
    
    current_task = DB[task_id]
    
    # Apply updates if provided
    update_data = task_update.model_dump(exclude_unset=True)
    for key, value in update_data.items():
        current_task[key] = value
        
    DB[task_id] = current_task
    return current_task

# DELETE (DELETE)
@app.delete("/tasks/{task_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_task(task_id: int):
    if task_id not in DB:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Task with ID {task_id} not found"
        )
    del DB[task_id]
    return Response(status_code=status.HTTP_204_NO_CONTENT)

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
