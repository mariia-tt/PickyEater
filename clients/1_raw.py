# Databricks notebook source
import ast
import pandas as pd

# COMMAND ----------

PATH_DF1 = "/Workspace/Picky_Eater/data/RAW_interactions.csv"

clients = pd.read_csv(PATH_DF1, dtype=str)

# quick check
print("clients shape:", clients.shape)
display(clients.head(5))

# COMMAND ----------

print("clients full duplicates:", clients.duplicated().sum())

# COMMAND ----------

print("Nulls in 'user_id':", clients['user_id'].isna().sum())
print("Nulls in 'recipe_id':", clients['recipe_id'].isna().sum())
print("Nulls in 'rating':", clients['rating'].isna().sum())
print("Nulls in 'date':", clients['date'].isna().sum())


# COMMAND ----------

subset_duplicates = clients.duplicated(subset=['user_id', 'recipe_id']).sum()
print(f"Number of duplicates in pairs [user_id, recipe_id]: {subset_duplicates}")

#if subset_duplicates > 0:
#    duplicates_df = clients[clients.duplicated(subset=['user_id', 'recipe_id'], keep=False)]
#    display(duplicates_df.sort_values(by=['user_id', 'recipe_id']).head(10))

# COMMAND ----------

# MAGIC %md
# MAGIC # Rating

# COMMAND ----------

print(clients['rating'].value_counts(dropna=False).sort_index())

unique_users = clients['user_id'].nunique()
unique_recipes = clients['recipe_id'].nunique()
total_clients = clients.shape[0]

print(f"\n--- 3. Базова статистика ---")
print(f"Number of logs: {total_clients}")
print(f"Unique users : {unique_users}")
print(f"Unique recipes: {unique_recipes}")

density = (total_clients / (unique_users * unique_recipes)) * 100
print(f"Density: {density:.4f}%")