import requests
import os
from dotenv import load_dotenv
import streamlit as st

load_dotenv()  
def get_recipe_and_nutrition(class_name):
    api_key ="3c8cb738939e43b6b10b5d673743d9ff"
    search_url = f"https://api.spoonacular.com/recipes/complexSearch?query={class_name}&addRecipeInformation=true&number=1&apiKey={api_key}"

    try:
        response = requests.get(search_url)
        response.raise_for_status()
        data = response.json()

        if not data.get('results'):
            return "No recipe found."

        recipe = data['results'][0]
        title = recipe.get('title', f"Recipe for {class_name.capitalize()}")
        summary = recipe.get('summary', 'No summary').replace('<b>', '').replace('</b>', '').replace('&nbsp;', ' ')
        source_url = recipe.get('sourceUrl', '')
        image_url = recipe.get('image', '')
        servings = recipe.get('servings', 'Not available')
        ready_in = recipe.get('readyInMinutes', 'Not available')
        cuisines = ', '.join(recipe.get('cuisines', [])) or 'Not specified'
        dish_types = ', '.join(recipe.get('dishTypes', [])) or 'Not specified'

        result = f"### {title}\n"
        if image_url:
            result += f'<img src="{image_url}" width="400">\n\n'

        result += f"**Summary:** {summary}\n\n"
        result += f"**Cuisines:** {cuisines}  \n"
        result += f"**Dish Types:** {dish_types}  \n"
        result += f"**Servings:** {servings}  \n"
        result += f"**Ready in:** {ready_in} minutes\n\n"

        if source_url:
            result += f"👉 [View Full Recipe]({source_url})\n"

        return result

    except Exception as e:
        return f"Error fetching recipe: {e}"
