# Databricks notebook source
# MAGIC %run ./2_clean

# COMMAND ----------

print(f"Number of rows: {dishes_silver.shape[0]}")
print(f"Number of columns: {dishes_silver.shape[1]}")

dishes_gold = dishes_silver.copy()

# COMMAND ----------

dishes_gold['nutrition_list'] = dishes_gold['nutrition'].apply(
    lambda x: ast.literal_eval(x) if pd.notna(x) else []
)

nutrition_cols = [
    'calories', 'total_fat_pdv', 'sugar_pdv', 'sodium_pdv', 
    'protein_pdv', 'saturated_fat_pdv', 'carbohydrates_pdv'
]

for i, col in enumerate(nutrition_cols):
    dishes_gold[col] = dishes_gold['nutrition_list'].apply(
        lambda x: x[i] if len(x) == 7 else None
    )

dishes_gold = dishes_gold.drop(columns=['nutrition', 'nutrition_list'])

# Перевіряємо результат
print("\nNutrition:")
display(dishes_gold[['name'] + nutrition_cols].head(3))

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

def parse_ingredients(ing_str):
    try:
        return set(ast.literal_eval(ing_str))
    except:
        return set()

dishes_gold['ing_set'] = dishes_gold['ingredients'].apply(parse_ingredients)

def get_suitable_diets(ing_set):
    suitable = []
    for diet, forbidden_foods in diet_rules.items():
        if not ing_set.intersection(forbidden_foods):
            suitable.append(diet)
    return suitable

dishes_gold['suitable_diets'] = dishes_gold['ing_set'].apply(get_suitable_diets)

final_dishes = dishes_gold.drop(columns=['ing_set'])

display(final_dishes[['name', 'ingredients', 'suitable_diets']].head(5))

# COMMAND ----------

SAVE_DIR = "/Workspace/Picky_Eater/new_data"

SAVE_PATH_DISHES = f"{SAVE_DIR}/cleaned_dishes.csv"
final_dishes.to_csv(SAVE_PATH_DISHES, index=False)

# COMMAND ----------

SAVE_DIR = "/Workspace/Picky_Eater/new_data"

SAVE_PATH_DISHES = f"{SAVE_DIR}/cleaned_dishes.csv"
final_dishes.to_csv(SAVE_PATH_DISHES, index=False)