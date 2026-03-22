# Databricks notebook source
# MAGIC %run ./1_raw

# COMMAND ----------

# MAGIC %run ../dishes/3_gold

# COMMAND ----------

# MAGIC %md
# MAGIC Зміна типів даних + Заповнення порожніх відгуків

# COMMAND ----------

clients_silver = clients.copy()

clients_silver['date'] = pd.to_datetime(clients_silver['date'], errors='coerce')
clients_silver['rating'] = pd.to_numeric(clients_silver['rating'], errors='coerce').astype('Int64')

null_reviews_before = clients_silver['review'].isna().sum()
clients_silver['review'] = clients_silver['review'].fillna("")
null_reviews_after = clients_silver['review'].isna().sum()

# COMMAND ----------

# MAGIC %md
# MAGIC Видалення відгуків до неіснуючих рецептів

# COMMAND ----------

initial_rows = clients_silver.shape[0]

valid_recipe_ids = set(dishes_gold['id'])

clients_silver = clients_silver[clients_silver['recipe_id'].isin(valid_recipe_ids)]

final_rows = clients_silver.shape[0]

print(f"Number of removed reviews: {initial_rows - final_rows}")

# COMMAND ----------

