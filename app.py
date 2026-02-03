import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import json
import re
from typing import List, Optional
from pydantic import BaseModel, Field

# --- LIBRARIES FROM YOUR PROMPT ---
try:
    from langchain_core.messages import HumanMessage
    from langchain_core.prompts import ChatPromptTemplate
    from langchain_google_genai import ChatGoogleGenerativeAI
    import pymupdf4llm 
except ImportError:
    st.error("Please install requirements: pip install streamlit pandas plotly langchain-google-genai pydantic pymupdf4llm")

# ==========================================
# 1. DATA & MODELS (Based on your Input)
# ==========================================

# --- Your Pydantic Models ---
class HardSkill(BaseModel):
    description: str = Field(description="Description of the hard skill.")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the hard skill application.")

class Position(BaseModel):
    name: str = Field(description="Name of the goal position or experienced position")
    field: str = Field(description="Field of the position")
    time_experience: Optional[float] = Field(description="Optional time experience in months identified for the held position.")

class ProfessionalProfile(BaseModel):
    positions: List[Position] = Field(description="Maximum of 3 names for matching positions identified in the profile description")
    main_position: List[Position] = Field(description="Main position goal identified for profile description")
    main_position_name_variations: list = Field(description="Main position name variations. Example: Senior Accountant, Accountant III, Accountant Specialist, Experienced Accountant")
    hard_skills: List[HardSkill]
    soft_skills: List[str] = Field(description="Description of the soft skill")

# --- Job Database (From your uploaded file) ---
JOBS_DB = [
  {
    "id": 889977, "title": "Data Engineer", "company_name": "NBC Universal", "location": "San Francisco, CA",
    "description": "We are seeking a Senior Data Engineer... skills: Snowflake, LiveRamp, Databricks, Python, MLOps, SQL, Snowpark, PySpark, Airflow, dbt, Great Expectations, LangChain, Snowflake Cortex."
  },
  {
    "id": 901122, "title": "Principal Data Engineer (FinTech)", "company_name": "Apex Finance Systems", "location": "New York, NY",
    "description": "Real-time Kafka-based architecture. Flink, Kafka, Snowflake, Python, Java, Scala, AWS, Kinesis, MSK, Lambda."
  },
  {
    "id": 901123, "title": "Junior Data Engineer", "company_name": "RetailFlow Inc.", "location": "Austin, TX",
    "description": "Maintain dbt projects and BigQuery. SQL, dbt, Airflow, Looker, Python, Git, GCP."
  },
  {
    "id": 901124, "title": "Senior Data Engineer - MLOps", "company_name": "NeuralPath AI", "location": "Remote",
    "description": "LLM training sets. Vector database, Pinecone, Weaviate, PySpark, Databricks, Spark, Kubernetes, MLflow."
  },
  {
    "id": 901125, "title": "Data Engineer (Contract)", "company_name": "Global Logistics Corp", "location": "Chicago, IL",
    "description": "Migration to Azure Data Factory and Synapse. MapReduce, Spark, Scala, Azure, Hadoop, Hive."
  },
  {
    "id": 901126, "title": "Lead Data Infrastructure Engineer", "company_name": "CloudScale Systems", "location": "Seattle, WA",
    "description": "Manage Terraform scripts. Trino, Presto, CI/CD, Go, Python, Terraform, CloudFormation, AWS."
  },
  {
    "id": 901127, "title": "Healthcare Data Engineer", "company_name": "BioHealth Data", "location": "Boston, MA",
    "description": "HIPAA-compliant data lakes. FHIR, HL7, AWS Glue, Athena, Python, Encryption."
  },
  {
    "id": 901128, "title": "Data Engineer - Analytics Engineering", "company_name": "Mountain Metrics", "location": "Denver, CO",
    "description": "dbt Cloud, Snowflake, Fivetran, Tableau, Unit Testing."
  },
  {
    "id": 901129, "title": "ETL Developer / Data Engineer", "company_name": "AdReach Agency", "location": "Miami, FL",
    "description": "Consolidating data. SQL, Python, API, Airbyte, Meltano."
  },
  {
    "id": 901130, "title": "Staff Data Engineer (Data Mesh focus)", "company_name": "Enterprise Scale Co", "location": "Remote",
    "description": "Decentralize architecture. Data Mesh, Data Fabric, Architecture."
  },
  {
    "id": 901131, "title": "Data Engineer - Game Analytics", "company_name": "Starlight Gaming", "location": "Los Angeles, CA",
    "description": "Track millions of events. Java, Kotlin, Scala, Google Cloud Dataflow, Apache Beam, BigQuery, Looker."
  }
]

# ==========================================
# 2. LOGIC & EXTRACTION
# ==========================================

# Helper: Simple Keyword Extractor for the JOBS DB (to simulate "Market Skills")
# In a production app, we would run the LLM on every job description to extract structured skills.
# For this prototype, we scan the description for known Data keywords.
KNOWN_SKILLS = [
    "python", "sql", "java", "scala", "go", "kotlin",
    "aws", "azure", "gcp", "snowflake", "databricks", "bigquery", "redshift",
    "spark", "pyspark", "hadoop", "hive", "kafka", "flink", "airflow", "dbt", 
    "docker", "kubernetes", "terraform", "ci/cd", "git",
    "looker", "tableau", "power bi",
    "machine learning", "mlops", "langchain", "llm", "rag", "vector database"
]

def enrich_jobs_with_skills(jobs):
    """Enriches the raw job descriptions with a list of identified skills."""
    enriched = []
    for job in jobs:
        desc = job['description'].lower()
        found_skills = [skill for skill in KNOWN_SKILLS if skill in desc]
        # Also add skills explicitly listed in the description text provided in the JSON if any
        job['extracted_skills'] = list(set(found_skills)) 
        enriched.append(job)
    return enriched

# --- Your LangChain Logic ---
def extract_professional_structured_data(text: str, api_key: str) -> dict:
    """Uses Google GenAI to extract the CV profile."""
    try:
        # Initialize Google GenAI
        llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash", # Using a fast model for the prototype
            temperature=0,
            google_api_key=api_key,
            max_retries=2
        )

        prompt = ChatPromptTemplate.from_messages([
            ("system", "Your role is to extract relevant data for building a professional role description object out of either a curriculum vitae text or a description for a job position out of input text"),
            ("human", "{input}")
        ])

        structured_llm = llm.with_structured_output(schema=ProfessionalProfile)
        chain = prompt | structured_llm
        response = chain.invoke({"input": text})
        return response.model_dump()
        
    except Exception as e:
        st.error(f"Error during extraction: {str(e)}")
        return None

def mock_extract_data():
    """Fallback mock data if no API key is provided."""
    return {
        "positions": [{"name": "Data Engineer", "field": "Data", "time_experience": 24.0}],
        "main_position": [{"name": "Data Engineer", "field": "Technology", "time_experience": 36.0}],
        "main_position_name_variations": ["Big Data Engineer"],
        "hard_skills": [
            {"description": "python", "time_experience": 48.0},
            {"description": "sql", "time_experience": 48.0},
            {"description": "aws", "time_experience": 24.0},
            {"description": "spark", "time_experience": 12.0},
            {"description": "airflow", "time_experience": 12.0},
        ],
        "soft_skills": ["Communication", "Problem Solving"]
    }

# ==========================================
# 3. STREAMLIT UI
# ==========================================

st.set_page_config(page_title="Data Engineer Market Fit", layout="wide")

# Initialize Session State
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'profile' not in st.session_state:
    st.session_state.profile = None

# --- SIDEBAR ---
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Required for live analysis using Gemini.")
    use_mock = st.checkbox("Use Mock Data (No API Key)", value=False)
    
    st.markdown("---")
    if st.button("Reset App"):
        st.session_state.page = 'upload'
        st.session_state.profile = None
        st.rerun()

# --- PAGE 1: UPLOAD ---
if st.session_state.page == 'upload':
    st.title("🚀 Data Engineer CV Analysis")
    st.markdown("### Upload your resume to check your Market Technical Fit")
    
    uploaded_file = st.file_uploader("Upload PDF Resume", type="pdf")
    
    if uploaded_file:
        if st.button("Analyze CV"):
            with st.spinner("Extracting Skills & analyzing Market Fit..."):
                # 1. Parse PDF
                if use_mock:
                    cv_text = "Mock Text"
                    profile_data = mock_extract_data()
                else:
                    if not api_key:
                        st.error("Please enter a Google API Key or select 'Use Mock Data'.")
                        st.stop()
                    
                    try:
                        # Attempt to read PDF
                        pdf_bytes = uploaded_file.read()
                        cv_text = pymupdf4llm.to_markdown(uploaded_file) # Assuming file path, but for stream we might need pymupdf directly
                        # NOTE: pymupdf4llm usually takes a file path. For Streamlit upload, we might need a workaround or save to temp.
                        # For prototype stability, let's treat it as text extraction:
                        import fitz # PyMuPDF
                        doc = fitz.open(stream=pdf_bytes, filetype="pdf")
                        cv_text = ""
                        for page in doc:
                            cv_text += page.get_text()
                            
                        profile_data = extract_professional_structured_data(cv_text, api_key)
                        
                    except Exception as e:
                        st.error(f"Error reading PDF: {e}")
                        st.stop()

                if profile_data:
                    st.session_state.profile = profile_data
                    st.session_state.page = 'dashboard'
                    st.rerun()

# --- PAGE 2: DASHBOARD ---
elif st.session_state.page == 'dashboard':
    profile = st.session_state.profile
    
    # 1. PREPARE DATA
    # Normalize Applicant Skills
    applicant_skills = set([s['description'].lower() for s in profile['hard_skills']])
    
    # Prepare Market Data
    market_data = enrich_jobs_with_skills(JOBS_DB)
    df_jobs = pd.DataFrame(market_data)
    
    # --- CALCULATIONS ---
    
    # A. Match Analysis per Job
    job_matches = []
    for job in market_data:
        job_skills = set(job['extracted_skills'])
        match = applicant_skills.intersection(job_skills)
        missing = job_skills.difference(applicant_skills)
        
        job_matches.append({
            "Job Title": job['title'],
            "Company": job['company_name'],
            "Match Count": len(match),
            "Total Job Skills": len(job_skills),
            "Match %": round(len(match) / len(job_skills) * 100, 1) if len(job_skills) > 0 else 0,
            "Missing Skills": ", ".join(list(missing))
        })
    df_matches = pd.DataFrame(job_matches).sort_values("Match Count", ascending=False)
    
    # B. Popular Applicant Skills
    # Count how many jobs require the skills the applicant HAS
    app_skill_counts = {}
    for skill in applicant_skills:
        count = sum(1 for job in market_data if skill in job['extracted_skills'])
        if count > 0:
            app_skill_counts[skill] = count
    df_app_pop = pd.DataFrame(list(app_skill_counts.items()), columns=['Skill', 'Number of Jobs Demands']).sort_values('Number of Jobs Demands', ascending=True)

    # C. Missing Skills Gap Analysis
    # Count how many jobs require skills the applicant DOES NOT HAVE
    missing_counter = {}
    total_jobs = len(market_data)
    for job in market_data:
        for skill in job['extracted_skills']:
            if skill not in applicant_skills:
                missing_counter[skill] = missing_counter.get(skill, 0) + 1
    
    gap_data = []
    for skill, count in missing_counter.items():
        if count > 1: # Only show significant gaps
            gap_data.append({
                "Skill": skill.title(), 
                "Frequency": count, 
                "% of Market": round((count/total_jobs)*100, 1)
            })
    df_gaps = pd.DataFrame(gap_data).sort_values("% of Market", ascending=False).head(10)

    # --- DASHBOARD UI ---
    
    st.button("← Upload New CV", on_click=lambda: st.session_state.update(page='upload'))
    
    st.title(f"📊 Market Technical Fit: {profile['main_position'][0]['name']}")
    
    # Top Metrics
    c1, c2, c3 = st.columns(3)
    c1.metric("Applicant Hard Skills", len(applicant_skills))
    c2.metric("Open Positions Analyzed", len(JOBS_DB))
    c3.metric("Avg. Match Score", f"{df_matches['Match %'].mean():.1f}%")

    st.divider()

    # ROW 1: CHARTS
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("✅ Your Most Valued Skills")
        st.caption("Which of your skills are most in demand across open positions?")
        if not df_app_pop.empty:
            fig1 = px.bar(df_app_pop, x='Number of Jobs Demands', y='Skill', orientation='h', text='Number of Jobs Demands', color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("No overlaps found with current market data.")

    with col2:
        st.subheader("⚠️ Critical Skill Gaps")
        st.caption("Skills you are missing that appear most frequently in job listings.")
        if not df_gaps.empty:
            fig2 = px.bar(df_gaps, x='% of Market', y='Skill', orientation='h', text='% of Market', color_discrete_sequence=['#FF5252'])
            fig2.update_traces(texttemplate='%{text}%')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("You are a perfect match for the current market! No major gaps found.")

    st.divider()

    # ROW 2: DETAILED MATCHES
    st.subheader("🏆 Overall Best Matching Jobs")
    st.caption("Ranked by total count of matching hard skills.")
    
    # Formatting the table
    st.dataframe(
        df_matches[['Job Title', 'Company', 'Match Count', 'Match %', 'Missing Skills']],
        use_container_width=True,
        hide_index=True,
        column_config={
            "Match %": st.column_config.ProgressColumn(
                "Match %",
                format="%.1f%%",
                min_value=0,
                max_value=100,
            ),
        }
    )