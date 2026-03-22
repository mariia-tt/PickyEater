# Databricks notebook source
# MAGIC %run ./2_clean

# COMMAND ----------

clients_gold = clients_silver.copy()

# COMMAND ----------

from collections import defaultdict
import numpy as np

# COMMAND ----------

# MAGIC %md
# MAGIC #Дієти

# COMMAND ----------

diet_rules = {
    'lactose_intolerant': {'milk', 'cheese', 'butter', 'cream', 'yogurt', 'whey', 'parmesan', 'mozzarella'},
    'vegetarian': {'beef', 'pork', 'chicken', 'fish', 'bacon', 'meat', 'ham', 'sausage', 'turkey', 'shrimp'},
    'vegan': {'beef', 'pork', 'chicken', 'fish', 'bacon', 'meat', 'ham', 'sausage', 'turkey', 'shrimp', 
              'milk', 'cheese', 'butter', 'cream', 'yogurt', 'eggs', 'egg', 'honey'},
    'nut_allergy': {'peanut', 'almond', 'walnut', 'pecan', 'macadamia', 'nuts', 'cashew', 'pine nut'},
    'gluten_free': {'flour', 'wheat', 'bread', 'pasta', 'macaroni', 'spaghetti', 'noodle'},
    'no_tomatoes': {'tomato', 'tomatoes', 'ketchup', 'tomato paste', 'tomato sauce'},
    'no_onions_garlic': {'onion', 'onions', 'garlic', 'garlic clove', 'garlic powder'}
}

diet_probs = {
    'lactose_intolerant': 0.15,
    'vegetarian': 0.05,
    'vegan': 0.02,
    'nut_allergy': 0.015,
    'gluten_free': 0.05,
    'no_tomatoes': 0.01,
    'no_onions_garlic': 0.01,
    'none': 0.695 
}

print("Правила та ймовірності завантажено!")

# COMMAND ----------

def parse_ingredients(ing_str):
    try:
        return set(ast.literal_eval(ing_str))
    except:
        return set()

dishes_gold['ing_set'] = dishes_gold['ingredients'].apply(parse_ingredients)

def get_violations(ing_set):
    violations = []
    for diet, forbidden_foods in diet_rules.items():
        if ing_set.intersection(forbidden_foods):
            violations.append(diet)
    return violations

dishes_gold['diet_violations'] = dishes_gold['ing_set'].apply(get_violations)

recipe_violations_dict = dict(zip(dishes_gold['id'], dishes_gold['diet_violations']))
print("Страви успішно проаналізовано на наявність алергенів/м'яса тощо!")

# COMMAND ----------

clients_gold['recipe_violations'] = clients_gold['recipe_id'].map(recipe_violations_dict)

violations_df = clients_gold.dropna(subset=['recipe_violations']).copy()

user_impossible_diets = (
    violations_df.explode('recipe_violations')
    .groupby('user_id')['recipe_violations']
    .apply(set)
    .to_dict()
)

clients_gold = clients_gold.drop(columns=['recipe_violations'])

print(f"Проаналізовано історію для {len(user_impossible_diets)} користувачів з усіх наявних відгуків.")

# COMMAND ----------

unique_users = clients_gold['user_id'].unique()
user_diet_mapping = {}

all_diets = list(diet_probs.keys())
all_probs = np.array(list(diet_probs.values()))

for u_id in unique_users:
    impossible_diets = user_impossible_diets.get(u_id, set())
    
    valid_mask = np.array([diet not in impossible_diets for diet in all_diets])
    
    if not valid_mask.any():
        user_diet_mapping[u_id] = 'none'
        continue
        
    valid_probs = all_probs * valid_mask
    valid_probs = valid_probs / valid_probs.sum()
    
    chosen_diet = np.random.choice(all_diets, p=valid_probs)
    user_diet_mapping[u_id] = chosen_diet

clients_gold['diet_restriction'] = clients_gold['user_id'].map(user_diet_mapping)

print("Розподіл згенерованих дієт серед клієнтів:")
print(clients_gold['diet_restriction'].value_counts(normalize=True) * 100)

# COMMAND ----------

reviews_per_user = clients_gold.groupby('diet_restriction')['user_id'].count() / clients_gold.groupby('diet_restriction')['user_id'].nunique()

print("Середня кількість відгуків на одного користувача:")
print(reviews_per_user.sort_values(ascending=False))

# COMMAND ----------

# MAGIC %md
# MAGIC # ПРодукти

# COMMAND ----------

import random

base_staples = ['salt', 'black pepper', 'olive oil', 'water', 'sugar', 'flour', 'butter', 'onion', 'garlic', 'eggs', 'milk']

diet_replacements = {
    'lactose_intolerant': {'milk': 'almond milk', 'cheese': 'vegan cheese', 'butter': 'olive oil', 'cream': 'coconut cream', 'yogurt': 'coconut yogurt', 'parmesan': 'nutritional yeast'},
    'vegan': {'milk': 'soy milk', 'cheese': 'vegan cheese', 'butter': 'olive oil', 'eggs': 'tofu', 'honey': 'maple syrup', 'chicken': 'tofu', 'beef': 'tempeh', 'pork': 'jackfruit', 'bacon': 'smoked tempeh', 'meat': 'soy meat', 'fish': 'seaweed'},
    'vegetarian': {'chicken': 'tofu', 'beef': 'tempeh', 'pork': 'jackfruit', 'bacon': 'smoked tempeh', 'meat': 'soy meat', 'fish': 'seaweed'},
    'nut_allergy': {'peanut': 'sunflower seeds', 'almond': 'pumpkin seeds', 'walnut': 'hemp seeds', 'nuts': 'mixed seeds', 'pecan': 'oats'},
    'gluten_free': {'flour': 'almond flour', 'wheat': 'buckwheat', 'bread': 'gluten-free bread', 'pasta': 'gluten-free pasta', 'spaghetti': 'rice noodles'},
    'no_tomatoes': {'tomato': 'red bell pepper', 'tomatoes': 'red bell peppers', 'ketchup': 'beet ketchup', 'tomato paste': 'pureed red peppers', 'tomato sauce': 'nomato sauce'},
    'no_onions_garlic': {'onion': 'fennel', 'onions': 'celery', 'garlic': 'asafoetida', 'garlic powder': 'asafoetida', 'garlic clove': 'asafoetida'}
}

all_ingredients_list = dishes_gold['ing_set'].explode().dropna()
popular_ingredients = all_ingredients_list.value_counts().head(200).index.tolist()

popular_ingredients = [ing for ing in popular_ingredients if ing not in base_staples]

# COMMAND ----------

mapped_reviews = clients_gold.merge(
    dishes_gold[['id', 'ing_set']], 
    left_on='recipe_id', 
    right_on='id', 
    how='inner'
)

user_ingredients_pool = (
    mapped_reviews.groupby('user_id')['ing_set']
    .apply(lambda sets: set().union(*sets))
    .to_dict()
)

print(f"Collected ingredient pools for {len(user_ingredients_pool)} users from all reviews.")

# COMMAND ----------

def generate_pantry(user_id, diet):
    pantry = set(base_staples)
    
    if user_id in user_ingredients_pool:
        user_pool = list(user_ingredients_pool[user_id])
        k = min(15, len(user_pool))
        pantry.update(random.sample(user_pool, k))
        
    pantry.update(random.sample(popular_ingredients, 5))
    
    final_pantry = set()
    
    if diet != 'none' and diet in diet_replacements:
        replacements = diet_replacements[diet]
        forbidden_words = diet_rules[diet]
        
        for item in pantry:
            if item in replacements:
                final_pantry.add(replacements[item]) 
            elif any(f_word in item for f_word in forbidden_words):
                pass 
            else:
                final_pantry.add(item)
    else:
        final_pantry = pantry
        
    return list(final_pantry)

clients_gold['available_ingredients'] = clients_gold.apply(
    lambda row: generate_pantry(row['user_id'], row['diet_restriction']), 
    axis=1
)

display(clients_gold[['user_id', 'diet_restriction', 'available_ingredients']].sample(10))

# COMMAND ----------

client_profiles = clients_gold[['user_id', 'diet_restriction', 'available_ingredients']].copy()
client_profiles = client_profiles.drop_duplicates(subset=['user_id'])

cleaned_interactions = clients_gold[['user_id', 'recipe_id', 'date', 'rating', 'review']].copy()

SAVE_DIR = "/Workspace/Picky_Eater/new_data"

client_profiles.to_csv(f"{SAVE_DIR}/client_profiles.csv", index=False)
cleaned_interactions.to_csv(f"{SAVE_DIR}/cleaned_interactions.csv", index=False)