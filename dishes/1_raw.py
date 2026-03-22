# Databricks notebook source
import ast
import pandas as pd


# COMMAND ----------

PATH_DF1 = "/Workspace/Picky_Eater/data/RAW_recipes.csv"

dishes = pd.read_csv(PATH_DF1, dtype=str)

# quick check
print("dishes shape:", dishes.shape)
display(dishes.head(5))

# COMMAND ----------

print("dishes full duplicates:", dishes.duplicated().sum())

# COMMAND ----------

print("Duplicates in 'id':", dishes['id'].duplicated().sum())
print("Nulls in 'id':", dishes['id'].isna().sum())

# COMMAND ----------

parsed_nutrition = dishes['nutrition'].apply(ast.literal_eval)
nutrition_lengths = parsed_nutrition.apply(len).value_counts()
print(nutrition_lengths)