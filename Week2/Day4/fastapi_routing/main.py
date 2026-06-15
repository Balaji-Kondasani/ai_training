from fastapi import FastAPI, Query, Path
from typing import Optional

# Initialize the FastAPI application
app = FastAPI(
    title="Day 5: FastAPI Routing & Parameters",
    description="Covers FastAPI routing, Path parameters, and Query parameters.",
    version="1.0.0"
)

# Mock database of products
PRODUCTS = {
    1: {"name": "Laptop", "price": 999.99, "category": "electronics"},
    2: {"name": "Smartphone", "price": 499.99, "category": "electronics"},
    3: {"name": "Coffee Maker", "price": 79.99, "category": "appliances"},
    4: {"name": "Notebook", "price": 4.99, "category": "stationery"},
}

@app.get("/")
def read_root():
    """
    Root Endpoint - Returns a basic greeting.
    """
    return {
        "message": "Welcome to Day 5 of FastAPI and ML!",
        "docs_url": "/docs",
        "redoc_url": "/redoc"
    }

# Path Parameter Endpoint
@app.get("/products/{product_id}")
def get_product(
    product_id: int = Path(..., description="The ID of the product to retrieve", gt=0)
):
    """
    Endpoint demonstrating Path Parameters.
    FastAPI automatically casts product_id to an integer and validates it is greater than 0.
    """
    product = PRODUCTS.get(product_id)
    if not product:
        return {"error": f"Product with ID {product_id} not found"}, 404
    return {"product_id": product_id, "data": product}

# Query Parameter Endpoint
@app.get("/products")
def list_products(
    category: Optional[str] = Query(None, description="Filter products by category"),
    min_price: float = Query(0.0, description="Minimum price of products", ge=0.0),
    max_price: Optional[float] = Query(None, description="Maximum price of products")
):
    """
    Endpoint demonstrating Query Parameters.
    Example URL: /products?category=electronics&min_price=100.0
    """
    filtered_products = {}
    for pid, details in PRODUCTS.items():
        # Filter by category if provided
        if category and details["category"].lower() != category.lower():
            continue
        # Filter by minimum price
        if details["price"] < min_price:
            continue
        # Filter by maximum price if provided
        if max_price is not None and details["price"] > max_price:
            continue
        filtered_products[pid] = details
        
    return {
        "filters": {
            "category": category,
            "min_price": min_price,
            "max_price": max_price
        },
        "results": filtered_products
    }

# Running instructions:
# Run: uvicorn main:app --reload
# Or run using `uv run uvicorn main:app`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="127.0.0.1", port=8000)
