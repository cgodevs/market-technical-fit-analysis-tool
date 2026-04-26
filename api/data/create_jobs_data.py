import json
import pandas as pd
import psycopg2
from os import getenv

# =============================================================================== #
# Filter Relevant Data
# =============================================================================== #

COLUMNS_SELECTION = ["id", "date_posted", "date_created", "title", "description_text", "seniority", "url", "countries_derived", "locations_derived", "organization", "organization_logo", "linkedin_org_url"]

with open("data/linkedin_api.json") as f:
    data = json.load(f)       
    df = pd.DataFrame(data)
    selected_columns_df = df[COLUMNS_SELECTION] 

# Give more weight to job postings similar to each other and avoid cluttering database with its duplicates.
selected_columns_df["weight"] = selected_columns_df.groupby(["organization", "title", "description_text"])["id"].transform("count")    
filtered_df = selected_columns_df.drop_duplicates(subset=["organization", "title", "description_text"])

# =============================================================================== #
# Transform Data
# =============================================================================== #

renamed_columns = {
    "description_text": "description",
    "countries_derived": "country",
    "locations_derived": "location"
}
country = lambda x: x["country"][0] if x["country"] else None
location = lambda x: x["location"][0] if x["location"] else None

transformed_df = filtered_df.rename(columns=renamed_columns)

transformed_df["country"] = transformed_df.apply(country, axis=1)
transformed_df["location"] = transformed_df.apply(location, axis=1)

transformed_df["date_posted"] = transformed_df["date_posted"].apply(lambda x: x.split("T")[0] if x else None)
transformed_df["date_created"] = transformed_df["date_created"].apply(lambda x: x.split("T")[0] if x else None)

renamed_seniority = {
    "Pleno-sênior": "Mid",
    "Não aplicável": None,
    "Assistente": "Junior",
    "Júnior": "Junior",
    "Cadre": None,
    "Non pertinent": None,
    "Mid-Senior level": "Mid",
    "Directeur": "Director",
    "Confirmé": None,
    "Associate": "Associate",
    "Estagiário": "Intern",
    "Estágio": "Intern"
}
transformed_df.replace({"seniority": renamed_seniority}, inplace=True)

transformed_df["c_source"] = "LinkedIn API"
transformed_df["f_ai_min_seniority"] = transformed_df["seniority"]
transformed_df["ai_experience_time_months"] = None
transformed_df["ai_industries"] = None  # List of industries extracted from job description using AI

del transformed_df["seniority"]
transformed_df

# =============================================================================== #
# Save data to local database
# =============================================================================== #

database_user = getenv("DB_USER")
database_password = getenv("DB_PASSWORD")
HOST = "localhost"
DATABASE = "market_fit"
TABLE_NAME = "job_postings"

with psycopg2.connect(
    host=HOST,
    database=DATABASE,
    user=database_user,
    password=database_password
) as conn:

    if conn:
        print("Connection to the database was successful!")
        conn.autocommit = True

    cur = conn.cursor()
    for index, row in transformed_df.iterrows():
        insert_query = f"""
            INSERT INTO {TABLE_NAME} (
                id, date_posted, date_created, title, description, url, country, location,
                organization, organization_logo, linkedin_org_url, weight, c_source,
                f_ai_min_seniority, ai_experience_time_months, ai_industries
            ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
            --ON CONFLICT (id) DO NOTHING
        """
        values = (
            row["id"], row["date_posted"], row["date_created"], row["title"], row["description"], row["url"], row["country"], row["location"],
            row["organization"], row["organization_logo"], row["linkedin_org_url"], row["weight"], row["c_source"],
            row["f_ai_min_seniority"], row["ai_experience_time_months"], row["ai_industries"]
        )
        cur.execute(insert_query, values)
