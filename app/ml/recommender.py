import ast
import pickle
from pathlib import Path

import lightgbm as lgb
import numpy as np
import pandas as pd
import torch

from app.ml.models import RecipeTwoTower

ARTIFACTS = Path(__file__).parent.parent.parent / "artifacts"

NUTRI_COLS = [
    "calories", "total_fat_pdv", "sugar_pdv", "sodium_pdv",
    "protein_pdv", "saturated_fat_pdv", "carbohydrates_pdv", "minutes",
]

RECIPE_METADATA_COLS = [
    "id", "minutes", "n_ingredients", "calories",
    "total_fat_pdv", "sugar_pdv", "sodium_pdv", "protein_pdv",
    "saturated_fat_pdv", "carbohydrates_pdv",
    "avg_rating", "n_ratings", "bayesian_avg",
]


def _safe_frozenset(val) -> frozenset:
    """Convert a value to a frozenset of strings.

    Handles lists, sets, ndarrays, and stringified lists safely.

    Args:
        val: The value to convert.

    Returns:
        frozenset of stripped strings, or an empty frozenset on failure.
    """
    if isinstance(val, (list, set, frozenset, np.ndarray)):
        return frozenset(str(v).strip() for v in val if v)
    try:
        r = ast.literal_eval(str(val).strip())
        if isinstance(r, (list, set, frozenset, np.ndarray)):
            return frozenset(str(x).strip() for x in r if x)
    except Exception:
        pass
    return frozenset()


class RecipeRecommender:
    """Three-stage recipe recommender: retrieval → LightGBM ranking → MMR.

    Holds all loaded models, encoders, and dataframes in memory.
    """

    def __init__(
        self,
        pytorch_model: RecipeTwoTower,
        lgbm_ranker: lgb.Booster,
        lgbm_features: list[str],
        user_encoder,
        recipe_encoder,
        text_dict: dict,
        sbert_norm: np.ndarray,
        recipe_universe: torch.Tensor,
        user_history_text: dict,
        user_history_profiles: pd.DataFrame,
        recipe_diet_index: dict,
        recipes_df: pd.DataFrame,
        profiles_df: pd.DataFrame,
        interactions_df: pd.DataFrame,
        device: torch.device,
        sbert_model=None,
    ):
        self.device = device
        self.model = pytorch_model
        self.ranker = lgbm_ranker
        self.lgbm_features = lgbm_features
        self.user_encoder = user_encoder
        self.recipe_encoder = recipe_encoder
        self.text_dict = text_dict
        self.sbert_norm = sbert_norm
        self.recipe_universe = recipe_universe
        self.user_history_text = dict(user_history_text)
        self.user_history_profiles = user_history_profiles.copy()
        self.recipe_diet_index = recipe_diet_index
        self.recipes_df = recipes_df
        self.profiles_df = profiles_df
        self.interactions_df = interactions_df
        self.sbert_model = sbert_model

        self.recipe_metadata = recipes_df[RECIPE_METADATA_COLS].copy()
        self.recipe_metadata["id"] = self.recipe_metadata["id"].astype(str)

        self._recipe_ing: dict[str, frozenset] = {
            str(row["id"]): _safe_frozenset(row["ingredients_clean"])
            for _, row in recipes_df.iterrows()
        }
        self._pantry_lookup: dict[str, frozenset] = {
            str(row["user_id"]): _safe_frozenset(row.get("available_ingredients", "[]"))
            for _, row in profiles_df.iterrows()
        }

        self._zero_hist = np.zeros(sbert_norm.shape[1], dtype=np.float32)
        self._session_ratings: dict[str, list[tuple[int, int]]] = {}

    def _get_user_idx(self, user_id_str: str) -> int | None:
        """Look up the integer index for a user ID string.

        Args:
            user_id_str: User ID as a string.

        Returns:
            Integer index, or None if the user is not in the encoder.
        """
        try:
            t = type(self.user_encoder.classes_[0])
            return int(self.user_encoder.transform([t(user_id_str)])[0])
        except Exception:
            return None

    def _get_recipe_idx(self, recipe_id_str: str) -> int | None:
        """Look up the integer index for a recipe ID string.

        Args:
            recipe_id_str: Recipe ID as a string.

        Returns:
            Integer index, or None if the recipe is not in the encoder.
        """
        try:
            t = type(self.recipe_encoder.classes_[0])
            return int(self.recipe_encoder.transform([t(recipe_id_str)])[0])
        except Exception:
            return None

    def _get_rated_ids(self, user_id_str: str) -> set[str]:
        """Return all recipe IDs this user has rated (training data + current session).

        Args:
            user_id_str: User ID as a string.

        Returns:
            Set of rated recipe ID strings.
        """
        rated: set[str] = set()
        u_idx = self._get_user_idx(user_id_str)
        if u_idx is not None:
            try:
                t = type(self.user_encoder.classes_[0])
                rows = self.interactions_df[self.interactions_df["user_id"] == t(user_id_str)]
                rated.update(rows["recipe_id"].astype(str))
            except Exception:
                pass
        for r_idx, _ in self._session_ratings.get(user_id_str, []):
            try:
                rated.add(str(self.recipe_encoder.inverse_transform([r_idx])[0]))
            except Exception:
                pass
        return rated

    def _resolve_pantry(self, user_id_str: str, fridge: list[str] | None) -> frozenset:
        """Resolve the ingredient pantry for a user.

        Uses fridge if provided, otherwise falls back to available_ingredients
        from the user's profile in profiles.parquet.

        Args:
            user_id_str: User ID as a string.
            fridge: Explicit ingredient list, or None to use the profile pantry.

        Returns:
            frozenset of ingredient strings.
        """
        if fridge is not None:
            return frozenset(fridge)
        return self._pantry_lookup.get(user_id_str, frozenset())

    def _resolve_diets(self, user_id_str: str, preferences: str | None) -> list[str] | None:
        """Resolve diet filters for a user.

        Uses preferences if provided, otherwise falls back to diet_restriction
        from the user's profile in profiles.parquet.

        Args:
            user_id_str: User ID as a string.
            preferences: Comma-separated diet labels, or None to use the profile.

        Returns:
            List of diet label strings, or None if no restrictions apply.
        """
        if preferences is not None:
            valid = [p.strip().lower() for p in preferences.split(",") if p.strip().lower() not in ("", "none")]
            if valid:
                return valid
        row = self.profiles_df[self.profiles_df["user_id"] == user_id_str]
        if not row.empty:
            diet = str(row.iloc[0].get("diet_restriction", "")).strip().lower()
            return [diet] if diet and diet != "none" else None
        return None

    def _retrieve(self, u_idx: int | None, hist_np: np.ndarray, K: int = 100) -> pd.DataFrame:
        """Run Two-Tower retrieval and return the top-K candidate recipes.

        Args:
            u_idx: User index, or None to use a zero-vector placeholder.
            hist_np: SBERT history vector as a numpy array.
            K: Number of candidates to retrieve.

        Returns:
            DataFrame with columns recipe_idx, pytorch_score, pytorch_rank.
        """
        u_t = torch.tensor([u_idx if u_idx is not None else 0], dtype=torch.long).to(self.device)
        hist_t = torch.tensor(hist_np, dtype=torch.float32).unsqueeze(0).to(self.device)
        self.model.eval()
        with torch.no_grad():
            u_vec = self.model.get_user_vec(u_t, hist_t)
            scores = torch.matmul(u_vec, self.recipe_universe.T).squeeze()
        top = torch.topk(scores, k=K)
        return pd.DataFrame({
            "recipe_idx": top.indices.cpu().numpy(),
            "pytorch_score": top.values.cpu().numpy(),
            "pytorch_rank": np.arange(1, K + 1),
        })

    def _popularity_retrieval(self, K: int = 100) -> pd.DataFrame:
        """Return top-K recipes by popularity score.

        Uses bayesian_avg if available, otherwise computes it on the fly
        from avg_rating and n_ratings using the global mean as a prior.

        Args:
            K: Number of candidates to return.

        Returns:
            DataFrame with columns recipe_idx, pytorch_score, pytorch_rank.
        """
        meta = self.recipe_metadata.copy()
        if "bayesian_avg" in meta.columns:
            meta["score"] = meta["bayesian_avg"]
        else:
            m = meta["n_ratings"].mean()
            C = meta["avg_rating"].mean()
            meta["score"] = (meta["n_ratings"] * meta["avg_rating"] + m * C) / (meta["n_ratings"] + m)
        meta = meta.nlargest(K, "score")
        idxs, scores = [], []
        for _, row in meta.iterrows():
            idx = self._get_recipe_idx(str(row["id"]))
            if idx is not None:
                idxs.append(idx)
                scores.append(float(row["score"]))
        return pd.DataFrame({
            "recipe_idx": np.array(idxs, dtype=np.int64),
            "pytorch_score": np.array(scores, dtype=np.float32),
            "pytorch_rank": np.arange(1, len(idxs) + 1),
        })

    def _session_hist_vec(self, user_id_str: str) -> np.ndarray:
        """Build a rating-weighted SBERT history vector from session ratings.

        Args:
            user_id_str: User ID as a string.

        Returns:
            Weighted mean SBERT vector, or a zero vector if no session ratings exist.
        """
        items = self._session_ratings.get(user_id_str, [])
        if not items:
            return self._zero_hist
        r_idxs = [r for r, _ in items]
        ratings = np.array([rt for _, rt in items], dtype=np.float32)
        vecs = np.array([self.text_dict.get(r, self._zero_hist) for r in r_idxs], dtype=np.float32)
        w = ratings / ratings.sum()
        return (vecs * w[:, None]).sum(axis=0)

    def _add_nutri_diffs(self, df: pd.DataFrame, u_idx: int | None, nutri_override: dict | None) -> pd.DataFrame:
        """Add per-nutrient difference columns relative to the user's average intake.

        Args:
            df: Candidate recipes DataFrame.
            u_idx: User index, or None for cold-start users.
            nutri_override: Pre-computed nutrition averages dict, or None to use stored profile.

        Returns:
            DataFrame with a *_diff column added for each nutrient in NUTRI_COLS.
        """
        if nutri_override is not None:
            avgs = nutri_override
        elif u_idx is not None:
            row = self.user_history_profiles[self.user_history_profiles["user_idx"] == u_idx]
            avgs = row.iloc[0].to_dict() if not row.empty else {}
        else:
            avgs = {}

        for col in NUTRI_COLS:
            avg = avgs.get(f"user_avg_{col}", float(df[col].mean()) if col in df.columns else 0.0)
            df[f"{col}_diff"] = abs(df[col] - avg) if col in df.columns else 0.0
        return df

    def _add_pantry(self, df: pd.DataFrame, pantry: frozenset) -> pd.DataFrame:
        """Add a pantry_overlap column with the fraction of ingredients the user has.

        Args:
            df: Candidate recipes DataFrame with an original_recipe_id column.
            pantry: frozenset of the user's available ingredients.

        Returns:
            DataFrame with pantry_overlap column added (0.0–1.0).
        """
        pantry_lower = {p.lower() for p in pantry}
        def fuzzy_overlap(r_id):
            ings = self._recipe_ing.get(str(r_id), frozenset())
            if not ings:
                return 0.0
            matched = sum(1 for ing in ings if any(p in ing.lower() or ing.lower() in p for p in pantry_lower))
            return matched / len(ings)
        df["pantry_overlap"] = df["original_recipe_id"].astype(str).map(fuzzy_overlap)
        return df

    def stage2_rank(self, candidates_df: pd.DataFrame, u_idx: int | None,
                    pantry: frozenset, nutri_override: dict | None = None) -> pd.DataFrame:
        """Score candidates with the LightGBM ranker.

        Args:
            candidates_df: DataFrame from stage 1 retrieval.
            u_idx: User index, or None for cold-start users.
            pantry: frozenset of the user's available ingredients.
            nutri_override: Pre-computed nutrition averages dict, or None to use stored profile.

        Returns:
            DataFrame sorted by lgbm_score descending.
        """
        df = candidates_df.copy()
        df["original_recipe_id"] = self.recipe_encoder.inverse_transform(df["recipe_idx"]).astype(str)
        df = df.merge(self.recipe_metadata, left_on="original_recipe_id", right_on="id", how="left").drop(columns=["id"], errors="ignore")
        if u_idx is not None:
            df["user_idx"] = u_idx
        df = self._add_nutri_diffs(df, u_idx, nutri_override)
        df = self._add_pantry(df, pantry)
        df = df.fillna(0)
        df["lgbm_score"] = self.ranker.predict(df[self.lgbm_features])
        return df.sort_values("lgbm_score", ascending=False).reset_index(drop=True)

    def stage3_mmr(self, ranked_df: pd.DataFrame, diets: list[str] | None,
                   top_n: int, lam: float = 0.7) -> pd.DataFrame:
        """Filter by diet then re-rank using Maximal Marginal Relevance.

        Args:
            ranked_df: DataFrame from stage 2 ranking.
            diets: Required diet labels to hard-filter by, or None to skip filtering.
            top_n: Number of results to return.
            lam: Trade-off weight — 1.0 is pure relevance, 0.0 is pure diversity.

        Returns:
            DataFrame of top_n diverse results.
        """
        df = ranked_df.copy()
        if diets:
            for d in diets:
                valid = {rid for rid, ds in self.recipe_diet_index.items() if d in ds}
                df = df[df["original_recipe_id"].isin(valid)]
        if df.empty:
            return df.head(top_n)

        idxs = df["recipe_idx"].values
        scores = df["lgbm_score"].values
        s_min, s_max = scores.min(), scores.max()
        norm_s = (scores - s_min) / (s_max - s_min + 1e-9)

        vecs = self.sbert_norm[idxs]
        selected, remaining = [], list(range(len(idxs)))

        while len(selected) < top_n and remaining:
            if not selected:
                best = max(remaining, key=lambda i: norm_s[i])
            else:
                sel_v = vecs[selected]
                best = max(
                    remaining,
                    key=lambda i: (lam * norm_s[i] - (1.0 - lam) * float((vecs[i] @ sel_v.T).max()))
                )
            selected.append(best)
            remaining.remove(best)

        return df.iloc[selected].reset_index(drop=True)

    def _format_results(self, df: pd.DataFrame, pantry: frozenset, strategy: str) -> list[dict]:
        """Convert a ranked DataFrame into the API response format.

        Args:
            df: Final ranked recipes DataFrame.
            pantry: frozenset of the user's available ingredients.
            strategy: Retrieval strategy label (e.g. "personalized", "popular").

        Returns:
            List of recipe dicts, each with metadata and a why block.
        """
        pantry_lower = {p.lower() for p in pantry}
        results = []
        for _, row in df.iterrows():
            ings = _safe_frozenset(row.get("ingredients_clean"))
            matched = sorted(ing for ing in ings if any(p in ing.lower() or ing.lower() in p for p in pantry_lower))
            fridge_pct = round(row.get("pantry_overlap", 0) * 100)
            results.append({
                "recipe_id": str(row["original_recipe_id"]),
                "name": row["name"],
                "calories": round(float(row["calories"]), 1) if pd.notna(row.get("calories")) else None,
                "minutes": int(row["minutes"]) if pd.notna(row.get("minutes")) else None,
                "avg_rating": round(float(row["avg_rating"]), 2) if pd.notna(row.get("avg_rating")) else None,
                "n_ratings": int(row["n_ratings"]) if pd.notna(row.get("n_ratings")) else 0,
                "description": row.get("description") if pd.notna(row.get("description")) else "",
                "final_score": round(float(row.get("lgbm_score", 0)), 4) if pd.notna(row.get("lgbm_score")) else 0.0,
                "why": {
                    "strategy": strategy,
                    "retrieval_rank": int(row.get("pytorch_rank", 0)),
                    "fridge_pct": fridge_pct,
                    "fridge_matched": matched[:6],
                },
            })
        return results

    def serve(self, user_id_str: str, top_n: int = 10,
              fridge: list[str] | None = None, preferences: str | None = None,
              seeds: list[str] | None = None, lam: float = 0.7) -> list[dict]:
        """Return top-N personalized recipe recommendations for a user.

        Picks a retrieval strategy based on user state:
        known user → Two-Tower, seeds → centroid retrieval,
        session-only → SBERT history, cold-start → popularity.

        Args:
            user_id_str: User ID as a string.
            top_n: Number of results to return.
            fridge: Explicit ingredient list, or None to use the profile pantry.
            preferences: Comma-separated diet labels, or None to use the profile.
            seeds: Recipe IDs to use as retrieval seeds.
            lam: MMR trade-off — 1.0 is pure relevance, 0.0 is pure diversity.

        Returns:
            List of recipe dicts.
        """
        u_idx = self._get_user_idx(user_id_str)
        rated_ids = self._get_rated_ids(user_id_str)
        pantry = self._resolve_pantry(user_id_str, fridge)
        diets = self._resolve_diets(user_id_str, preferences)

        if u_idx is not None:
            strategy = "personalized"
            hist = self.user_history_text.get(u_idx, self._zero_hist)
            s1 = self._retrieve(u_idx, hist)

        elif seeds:
            strategy = "seeds"
            seed_idxs = []
            for r in seeds:
                idx = self._get_recipe_idx(r)
                if idx is not None:
                    seed_idxs.append(idx)
            if seed_idxs:
                centroid = self.recipe_universe[seed_idxs].mean(dim=0)
                scores = torch.matmul(centroid.unsqueeze(0), self.recipe_universe.T).squeeze()
                top = torch.topk(scores, k=100)
                s1 = pd.DataFrame({
                    "recipe_idx": top.indices.cpu().numpy(),
                    "pytorch_score": top.values.cpu().numpy(),
                    "pytorch_rank": np.arange(1, 101),
                })
            else:
                s1 = self._popularity_retrieval()

        elif self._session_ratings.get(user_id_str):
            strategy = "session_content"
            hist = self._session_hist_vec(user_id_str)
            s1 = self._retrieve(None, hist)

        else:
            strategy = "popular"
            s1 = self._popularity_retrieval()

        if rated_ids:
            rated_idxs = {self._get_recipe_idx(r) for r in rated_ids} - {None}
            s1 = s1[~s1["recipe_idx"].isin(rated_idxs)].reset_index(drop=True)
            s1["pytorch_rank"] = np.arange(1, len(s1) + 1)

        nutri_override = None
        if u_idx is None and self._session_ratings.get(user_id_str):
            items = self._session_ratings[user_id_str]
            try:
                r_ids = [str(self.recipe_encoder.inverse_transform([r])[0]) for r, _ in items]
                rows = self.recipes_df[self.recipes_df["id"].isin(r_ids)]
                if not rows.empty:
                    nutri_override = {f"user_avg_{c}": float(rows[c].mean()) for c in NUTRI_COLS if c in rows.columns}
            except Exception:
                pass

        s2 = self.stage2_rank(s1, u_idx, pantry, nutri_override)
        s2 = s2.merge(
            self.recipes_df[["id", "name", "ingredients_clean", "tags_clean", "description"]],
            left_on="original_recipe_id", right_on="id", how="left", suffixes=("", "_r"),
        )
        final = self.stage3_mmr(s2, diets, top_n, lam)
        return self._format_results(final, pantry, strategy)

    def rerank(self, recipe_ids: list[str], user_id_str: str,
               fridge: list[str] | None = None, preferences: str | None = None) -> list[dict]:
        """Re-rank an explicit list of recipe IDs for a user.

        Skips stage 1 retrieval and runs stages 2 and 3 only.

        Args:
            recipe_ids: Ordered list of recipe IDs to re-rank.
            user_id_str: User ID as a string.
            fridge: Explicit ingredient list, or None to use the profile pantry.
            preferences: Comma-separated diet labels, or None to use the profile.

        Returns:
            Re-ranked list of recipe dicts.
        """
        u_idx = self._get_user_idx(user_id_str)
        rated_ids = self._get_rated_ids(user_id_str)
        pantry = self._resolve_pantry(user_id_str, fridge)
        diets = self._resolve_diets(user_id_str, preferences)

        rows = []
        for pos, rid in enumerate(recipe_ids):
            if rid in rated_ids:
                continue
            r_idx = self._get_recipe_idx(rid)
            if r_idx is None:
                continue
            rows.append({"recipe_idx": r_idx, "pytorch_score": 1.0 / (pos + 1), "pytorch_rank": pos + 1})
        if not rows:
            return []

        s2 = self.stage2_rank(pd.DataFrame(rows), u_idx, pantry)
        s2 = s2.merge(
            self.recipes_df[["id", "name", "ingredients_clean", "tags_clean", "description"]],
            left_on="original_recipe_id", right_on="id", how="left", suffixes=("", "_r"),
        )
        final = self.stage3_mmr(s2, diets, len(rows))
        return self._format_results(final, pantry, "search_reranked")

    def rate(self, user_id_str: str, recipe_id_str: str, rating: int) -> None:
        """Record a rating and update the user's in-memory history vectors.

        Args:
            user_id_str: User ID as a string.
            recipe_id_str: Recipe ID as a string.
            rating: Integer rating from 1 to 5.
        """
        r_idx = self._get_recipe_idx(recipe_id_str)
        if r_idx is None:
            return
        u_idx = self._get_user_idx(user_id_str)

        if u_idx is not None:
            new_v = self.text_dict.get(r_idx, self._zero_hist)
            old_v = self.user_history_text.get(u_idx, self._zero_hist.copy())
            alpha = (rating / 5.0) * 0.15
            self.user_history_text[u_idx] = old_v + alpha * (new_v - old_v)

            recipe_row = self.recipes_df[self.recipes_df["id"] == recipe_id_str]
            if not recipe_row.empty:
                mask = self.user_history_profiles["user_idx"] == u_idx
                if mask.any():
                    for col in NUTRI_COLS:
                        if col in recipe_row.columns:
                            old_avg = float(self.user_history_profiles.loc[mask, f"user_avg_{col}"].values[0])
                            new_val = float(recipe_row[col].values[0])
                            self.user_history_profiles.loc[mask, f"user_avg_{col}"] = old_avg * 0.9 + new_val * 0.1

        sess = self._session_ratings.setdefault(user_id_str, [])
        for i, (ri, _) in enumerate(sess):
            if ri == r_idx:
                sess[i] = (r_idx, rating)
                return
        sess.append((r_idx, rating))

    def search(self, query: str, limit: int = 20, mode: str = "text") -> list[dict]:
        """Search recipes by query string.

        Args:
            query: Search query.
            limit: Maximum number of results to return.
            mode: "text" for keyword search, "semantic" for embedding-based search.

        Returns:
            List of matching recipe dicts.
        """
        if mode == "semantic" and self.sbert_model is not None:
            return self._semantic_search(query, limit)
        return self._text_search(query, limit)

    def _text_search(self, query: str, limit: int) -> list[dict]:
        """Search recipes by matching against name, tags, and ingredients.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching recipe dicts.
        """
        q = query.lower().strip()
        df = self.recipes_df
        mask = (
            df["name"].str.lower().str.contains(q, na=False, regex=False)
            | df["tags_clean"].apply(lambda x: any(q in t for t in (x if isinstance(x, (list, set)) else [])))
            | df["ingredients_clean"].apply(lambda x: any(q in t for t in (x if isinstance(x, (list, set)) else [])))
        )
        return [r for r in (self.get_recipe(str(row["id"])) for _, row in df[mask].head(limit).iterrows()) if r]

    def _semantic_search(self, query: str, limit: int) -> list[dict]:
        """Search recipes using SBERT embeddings and cosine similarity.

        Args:
            query: Search query string.
            limit: Maximum number of results to return.

        Returns:
            List of matching recipe dicts, each with a similarity score.
        """
        q_vec = self.sbert_model.encode([query], normalize_embeddings=True, convert_to_numpy=True)[0]
        sims = self.sbert_norm @ q_vec
        top_idxs = np.argsort(-sims)[:limit]
        results = []
        for idx in top_idxs:
            try:
                r_id = str(self.recipe_encoder.inverse_transform([idx])[0])
                recipe = self.get_recipe(r_id)
                if recipe:
                    recipe["similarity"] = round(float(sims[idx]), 4)
                    results.append(recipe)
            except Exception:
                continue
        return results

    def get_recipe(self, recipe_id_str: str) -> dict | None:
        """Return full metadata for a single recipe.

        Args:
            recipe_id_str: Recipe ID as a string.

        Returns:
            Recipe metadata dict, or None if not found.
        """
        row = self.recipes_df[self.recipes_df["id"] == recipe_id_str]
        if row.empty:
            return None
        r = row.iloc[0]
        return {
            "recipe_id": str(r["id"]),
            "name": r["name"],
            "minutes": int(r["minutes"]),
            "n_steps": int(r["n_steps"]),
            "n_ingredients": int(r["n_ingredients"]),
            "calories": round(float(r["calories"]), 1),
            "total_fat_pdv": round(float(r["total_fat_pdv"]), 1),
            "sugar_pdv": round(float(r["sugar_pdv"]), 1),
            "sodium_pdv": round(float(r["sodium_pdv"]), 1),
            "protein_pdv": round(float(r["protein_pdv"]), 1),
            "saturated_fat_pdv": round(float(r["saturated_fat_pdv"]), 1),
            "carbohydrates_pdv": round(float(r["carbohydrates_pdv"]), 1),
            "avg_rating": round(float(r["avg_rating"]), 2),
            "n_ratings": int(r["n_ratings"]),
            "ingredients": list(_safe_frozenset(r["ingredients_clean"])),
            "tags": list(_safe_frozenset(r["tags_clean"])),
            "suitable_diets": r.get("suitable_diets") or "",
            "description": r.get("description") or "",
            "steps": r.get("steps") or "",
        }

    def get_user_profile(self, user_id_str: str) -> dict:
        """Return a user's diet restriction and pantry from their profile.

        Args:
            user_id_str: User ID as a string.

        Returns:
            Dict with keys diet (str or None) and fridge (list of strings).
        """
        row = self.profiles_df[self.profiles_df["user_id"] == user_id_str]
        if row.empty:
            return {"diet": None, "fridge": []}
        r = row.iloc[0]
        diet = str(r.get("diet_restriction") or "").strip().lower()
        return {
            "diet": diet if diet and diet != "none" else None,
            "fridge": list(self._pantry_lookup.get(user_id_str, [])),
        }

    def get_user_ratings(self, user_id_str: str) -> list[dict]:
        """Return all ratings for a user — session ratings first, then training history.

        Args:
            user_id_str: User ID as a string.

        Returns:
            List of rating dicts sorted by rating descending within each group.
        """
        u_idx = self._get_user_idx(user_id_str)
        training_rows, session_rows = [], []

        if u_idx is not None:
            try:
                user_rows = self.interactions_df[
                    self.interactions_df["user_id"].astype(str) == str(user_id_str)
                ].copy()
                user_rows["recipe_id"] = user_rows["recipe_id"].astype(str)
                merged = user_rows.merge(
                    self.recipes_df[["id", "name", "calories", "minutes"]],
                    left_on="recipe_id", right_on="id", how="left",
                ).sort_values(["rating", "date"], ascending=[False, False])
                training_rows = [{
                    "recipe_id": str(r["recipe_id"]),
                    "name": r.get("name") or f"Recipe #{r['recipe_id']}",
                    "rating": int(r["rating"]),
                    "date": str(r["date"])[:10],
                    "calories": round(float(r["calories"]), 1) if pd.notna(r.get("calories")) else None,
                    "minutes": int(r["minutes"]) if pd.notna(r.get("minutes")) else None,
                } for _, r in merged.iterrows()]
            except Exception as e:
                print(f"get_user_ratings training read failed for {user_id_str}: {e}")

        for r_idx, rating in self._session_ratings.get(user_id_str, []):
            try:
                r_id = str(self.recipe_encoder.inverse_transform([r_idx])[0])
                recipe = self.get_recipe(r_id)
                session_rows.append({
                    "recipe_id": r_id,
                    "name": recipe["name"] if recipe else r_id,
                    "rating": rating,
                    "date": "session",
                    "calories": recipe["calories"] if recipe else None,
                    "minutes": recipe["minutes"] if recipe else None,
                })
            except Exception as e:
                print(f"get_user_ratings session read failed for {user_id_str}: {e}")

        session_rows.sort(key=lambda x: x["rating"], reverse=True)
        return session_rows + training_rows

    def random_user(self) -> str:
        """Return a random user ID from the training dataset.

        Returns:
            User ID as a string.
        """
        return str(np.random.choice(self.user_encoder.classes_))


def load_recommender() -> RecipeRecommender:
    """Load all artifacts from disk and return a ready RecipeRecommender.

    Returns:
        Fully initialized RecipeRecommender instance.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Loading recommender from {ARTIFACTS} on {device}...")

    with open(ARTIFACTS / "encoders.pkl", "rb") as f:
        user_encoder, recipe_encoder = pickle.load(f)

    ckpt = torch.load(ARTIFACTS / "best_model.pt", map_location=device)
    model = RecipeTwoTower(
        num_users=ckpt["num_users"],
        num_recipes=ckpt["num_recipes"],
        text_dim=ckpt["text_dim"],
        embed_dim=ckpt["embed_dim"],
    ).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    print("  Two-Tower model loaded.")

    ranker = lgb.Booster(model_file=str(ARTIFACTS / "ranker.txt"))
    print("  LightGBM ranker loaded.")

    with open(ARTIFACTS / "lgbm_features.pkl", "rb") as f:
        lgbm_features = pickle.load(f)

    with open(ARTIFACTS / "text_dict.pkl", "rb") as f:
        text_dict = pickle.load(f)

    with open(ARTIFACTS / "recipe_diet_index.pkl", "rb") as f:
        recipe_diet_index = pickle.load(f)

    with open(ARTIFACTS / "user_history_text.pkl", "rb") as f:
        user_history_text = pickle.load(f)

    sbert_norm = np.load(str(ARTIFACTS / "sbert_norm.npy"))
    recipe_universe = torch.load(str(ARTIFACTS / "recipe_universe.pt"), map_location=device)

    recipes_df = pd.read_parquet(ARTIFACTS / "recipes.parquet")
    recipes_df["id"] = recipes_df["id"].astype(str)

    profiles_df = pd.read_parquet(ARTIFACTS / "profiles.parquet")
    profiles_df["user_id"] = profiles_df["user_id"].astype(str)

    user_history_profiles = pd.read_parquet(ARTIFACTS / "user_history_profiles.parquet")
    interactions_df = pd.read_parquet(ARTIFACTS / "interactions.parquet")

    sbert_model = None
    try:
        from sentence_transformers import SentenceTransformer
        sbert_model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        print("  SBERT loaded for semantic search.")
    except Exception as e:
        print(f"  SBERT unavailable, semantic search will fall back to text: {e}")

    print("Recommender ready.")
    return RecipeRecommender(
        pytorch_model=model,
        lgbm_ranker=ranker,
        lgbm_features=lgbm_features,
        user_encoder=user_encoder,
        recipe_encoder=recipe_encoder,
        text_dict=text_dict,
        sbert_norm=sbert_norm,
        recipe_universe=recipe_universe,
        user_history_text=user_history_text,
        user_history_profiles=user_history_profiles,
        recipe_diet_index=recipe_diet_index,
        recipes_df=recipes_df,
        profiles_df=profiles_df,
        interactions_df=interactions_df,
        device=device,
        sbert_model=sbert_model,
    )