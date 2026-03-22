# Databricks notebook source
# MAGIC %run ./1_raw

# COMMAND ----------

print(f"Number of rows in dishes (Bronze): {dishes.shape[0]}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Name

# COMMAND ----------

null_names = dishes['name'].isna().sum()
print(f"No 'name': {null_names}")

# COMMAND ----------

missing_name_row = dishes[dishes['name'].isna()]
display(missing_name_row)

# COMMAND ----------

# MAGIC %md
# MAGIC # Time

# COMMAND ----------

dishes['minutes'] = pd.to_numeric(dishes['minutes'], errors='coerce')
invalid_minutes = (dishes['minutes'] <= 0).sum()
print(f"time <= 0: {invalid_minutes}")

# COMMAND ----------

print("\nPercentiles of 'minutes':")
print(dishes['minutes'].describe(percentiles=[0.5, 0.90, 0.95, 0.99]))

# COMMAND ----------

p99 = dishes['minutes'].quantile(0.99)
outliers_count = (dishes['minutes'] > p99).sum()

print(f"99 percentile: {p99}")
print(f"number of dishes with minutes > 99 percentile: {outliers_count}")

# COMMAND ----------

# MAGIC %md
# MAGIC # Steps & Ingradients & Data

# COMMAND ----------

n_steps_num = pd.to_numeric(dishes['n_steps'], errors='coerce')
n_ingredients_num = pd.to_numeric(dishes['n_ingredients'], errors='coerce')

invalid_steps = ((n_steps_num <= 0) | n_steps_num.isna()).sum()
invalid_ingredients = ((n_ingredients_num <= 0) | n_ingredients_num.isna()).sum()

submitted_date = pd.to_datetime(dishes['submitted'], errors='coerce')
invalid_dates = submitted_date.isna().sum()

print(f"Number of dishes with invalid steps is : {invalid_steps}")
print(f"Number of dishes with invalid ingradients is: {invalid_ingredients}")
print(f"Number of dishes with invalid submishen date: {invalid_dates}")

# COMMAND ----------

anomaly_row = dishes[(n_steps_num <= 0) | (n_steps_num.isna())]
display(anomaly_row)

# COMMAND ----------

# MAGIC %md
# MAGIC # Final Pipeline

# COMMAND ----------



dishes_silver = dishes.copy()

dishes_silver['name'] = dishes_silver['name'].fillna("No Name")

dishes_silver = dishes_silver[dishes_silver['id'].notna()]

dishes_silver['minutes'] = pd.to_numeric(dishes_silver['minutes'], errors='coerce')
p99_minutes = dishes_silver['minutes'].quantile(0.99)

dishes_silver = dishes_silver[
    (dishes_silver['minutes'] > 0) & 
    (dishes_silver['minutes'] <= p99_minutes) # I'm not sure
]
print(f"Number of rows : {dishes_silver.shape[0]} after deleting outliers in 'minutes'")

dishes_silver['n_steps'] = pd.to_numeric(dishes_silver['n_steps'], errors='coerce')

empty_steps = dishes_silver['steps'].isna() | (dishes_silver['steps'] == '[]') | (dishes_silver['steps'] == '')
invalid_n_steps = dishes_silver['n_steps'].isna() | (dishes_silver['n_steps'] <= 0)

dishes_silver = dishes_silver[~(invalid_n_steps & empty_steps)]
print(f"Number of rows: {dishes_silver.shape[0]} after deleting outliers in 'n_steps' ")

dishes_silver['n_ingredients'] = pd.to_numeric(dishes_silver['n_ingredients'], errors='coerce')

empty_ingredients = dishes_silver['ingredients'].isna() | (dishes_silver['ingredients'] == '[]') | (dishes_silver['ingredients'] == '')
invalid_n_ingredients = dishes_silver['n_ingredients'].isna() | (dishes_silver['n_ingredients'] <= 0)

dishes_silver = dishes_silver[~(invalid_n_ingredients & empty_ingredients)]
print(f"Number of rows: {dishes_silver.shape[0]} after deleting outliers in 'n_ingredients'")

dishes_silver['submitted'] = pd.to_datetime(dishes_silver['submitted'], errors='coerce')

dishes_silver = dishes_silver[dishes_silver['submitted'].notna()]
print(f"Number of rows: {dishes_silver.shape[0]} after deleting invalid date in 'submitted date'")