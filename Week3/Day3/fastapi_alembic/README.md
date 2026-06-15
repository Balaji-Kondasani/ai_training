# Day 18: Database Migrations with Alembic

This folder teaches you how to manage your database schemas using **Alembic**, a database migration tool for SQLAlchemy. 

Unlike Day 17, which created database tables using `Base.metadata.create_all(bind=engine)` on application startup, this day uses Alembic to manage database changes sequentially. This is the industry-standard way of handling database modifications in production, as it allows you to upgrade, downgrade, and track schema changes in git.

> [!NOTE]
> **Initial Migration Pre-Applied:** The initial migration has already been generated and run, creating the database (`prediction_audit.db`) and table structure. You can run the server right away!

---

## Running commands inside this directory

To run migrations or interact with Alembic, open a terminal, **change directory into this folder first**, and activate your virtual environment:

```bash
cd machine_learning/day_18_fastapi_alembic
..\..\.venv\Scripts\activate
```

### 1. Generating a New Migration (e.g. after modifying models.py)
If you make any changes to your SQLAlchemy models in `models.py` (like adding a column), generate a new migration file:
```bash
..\..\.venv\Scripts\alembic revision --autogenerate -m "Add new column description"
```
This will create a new Python script inside `alembic/versions/` detailing the database upgrades/downgrades.

### 2. Applying Migrations to the Database
To apply any pending migrations and update the SQLite database table structure:
```bash
..\..\.venv\Scripts\alembic upgrade head
```

### 3. Downgrading the Database
If you need to roll back the database schema by one migration step:
```bash
..\..\.venv\Scripts\alembic downgrade -1
```

---

## Run the FastAPI Server

Start the API server from the root directory or inside this folder:
```bash
# From the project root
.venv\Scripts\uvicorn machine_learning.day_18_fastapi_alembic.main:app --reload
```

Test endpoints via Swagger UI at **`http://127.0.0.1:8000/docs`**.
