import os
from contextlib import asynccontextmanager
from typing import Literal

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from app.ml.recommender import RecipeRecommender, load_recommender


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the recommender at startup and store it in app state."""
    app.state.rec: RecipeRecommender = load_recommender()
    yield


app = FastAPI(title="Picky Eater", lifespan=lifespan)
app.mount("/static", StaticFiles(directory="app/static"), name="static")

os.makedirs("artifacts/recipe_images", exist_ok=True)
app.mount("/images", StaticFiles(directory="artifacts/recipe_images"), name="images")


@app.get("/")
def root():
    """Serve the frontend HTML."""
    return FileResponse("app/static/index.html")


@app.get("/user/{user_id}")
def get_user_metadata(user_id: str):
    """Return all ratings (training + session) and profile info for a user.

    Args:
        user_id: User ID from the URL path.

    Returns:
        Dict with user_id, n_ratings, ratings list, and profile.
    """
    rec: RecipeRecommender = app.state.rec
    ratings = rec.get_user_ratings(user_id)
    profile = rec.get_user_profile(user_id)
    return {"user_id": user_id, "n_ratings": len(ratings), "ratings": ratings, "profile": profile}


class RateRequest(BaseModel):
    """Request body for the rate endpoint."""

    recipe_id: str
    rating: int = Field(..., ge=1, le=5)


@app.post("/user/{user_id}/rate")
def rate_recipe(user_id: str, body: RateRequest):
    """Record a recipe rating for a user.

    Args:
        user_id: User ID from the URL path.
        body: Request body with recipe_id and rating (1–5).

    Returns:
        Dict confirming the recorded rating.

    Raises:
        HTTPException: 404 if the recipe does not exist.
    """
    rec: RecipeRecommender = app.state.rec
    if rec.get_recipe(body.recipe_id) is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    rec.rate(user_id, body.recipe_id, body.rating)
    return {"user_id": user_id, "recipe_id": body.recipe_id, "rating": body.rating}


@app.get("/recommendations/{user_id}")
def get_recommendations(
    user_id: str,
    top_k: int = Query(default=10, ge=1, le=100),
    fridge: list[str] = Query(default=[]),
    preferences: str = Query(default=None),
    seeds: list[str] = Query(default=[]),
):
    """Return personalized recipe recommendations for a user.

    Args:
        user_id: User ID from the URL path.
        top_k: Number of results to return.
        fridge: Ingredient list to filter by pantry overlap.
        preferences: Comma-separated diet labels.
        seeds: Recipe IDs to use as retrieval seeds.

    Returns:
        Dict with user_id, top_k, and a recommendations list.
    """
    rec: RecipeRecommender = app.state.rec
    results = rec.serve(
        user_id_str=user_id,
        top_n=top_k,
        fridge=fridge or None,
        preferences=preferences,
        seeds=seeds or None,
    )
    return {"user_id": user_id, "top_k": top_k, "recommendations": results}


@app.get("/recipe/{recipe_id}")
def get_recipe_metadata(recipe_id: str):
    """Return full metadata for a single recipe.

    Args:
        recipe_id: Recipe ID from the URL path.

    Returns:
        Recipe metadata dict.

    Raises:
        HTTPException: 404 if the recipe does not exist.
    """
    rec: RecipeRecommender = app.state.rec
    recipe = rec.get_recipe(recipe_id)
    if recipe is None:
        raise HTTPException(status_code=404, detail="Recipe not found")
    return recipe


@app.get("/search")
def search_recipes(
    q: str = Query(..., min_length=1),
    limit: int = Query(default=20, ge=1, le=100),
    mode: Literal["text", "semantic"] = Query(default="text"),
):
    """Search recipes by keyword or semantic similarity.

    Args:
        q: Search query string.
        limit: Maximum number of results to return.
        mode: "text" for keyword search, "semantic" for embedding-based search.

    Returns:
        Dict with query, mode, n_results, and a results list.
    """
    rec: RecipeRecommender = app.state.rec
    results = rec.search(q, limit=limit, mode=mode)
    return {"query": q, "mode": mode, "n_results": len(results), "results": results}


class RerankRequest(BaseModel):
    """Request body for the rerank endpoint."""

    recipe_ids: list[str]
    fridge: list[str] = []
    preferences: str | None = None


@app.post("/rerank/{user_id}")
def rerank_recipes(user_id: str, body: RerankRequest):
    """Re-rank a list of recipe IDs by predicted user preference.

    Args:
        user_id: User ID from the URL path.
        body: Request body with recipe_ids, fridge, and preferences.

    Returns:
        Dict with user_id, n_results, and a results list.
    """
    rec: RecipeRecommender = app.state.rec
    results = rec.rerank(
        recipe_ids=body.recipe_ids,
        user_id_str=user_id,
        fridge=body.fridge or None,
        preferences=body.preferences,
    )
    return {"user_id": user_id, "n_results": len(results), "results": results}


@app.get("/diets")
def get_available_diets():
    """Return all diet labels present in the dataset.

    Returns:
        Dict with a sorted list of diet label strings.
    """
    rec: RecipeRecommender = app.state.rec
    diets = sorted({d for ds in rec.recipe_diet_index.values() for d in ds})
    return {"diets": diets}


@app.get("/random-user")
def get_random_user(response: Response):
    """Return a random user ID from the training dataset.

    Args:
        response: FastAPI response object used to disable caching.

    Returns:
        Dict with a random user_id string.
    """
    response.headers["Cache-Control"] = "no-store, no-cache, must-revalidate, max-age=0"
    rec: RecipeRecommender = app.state.rec
    return {"user_id": rec.random_user()}