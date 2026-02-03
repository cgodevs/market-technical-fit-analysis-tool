import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

from utils import *
import pymupdf4llm 


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
    if st.button("Restart"):
        st.session_state.page = 'upload'
        st.session_state.profile = None
        st.rerun()

# --- PAGE 1: UPLOAD ---
if st.session_state.page == 'upload':
    st.title("Market Technical Fit Analysis for your Resume", text_alignment="center")
    st.space("medium")
    uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf", width="stretch")
    
    if uploaded_file:
        if st.button("Analyze CV"):
            with st.spinner("Extracting Skills & analyzing Market Fit..."):
                # 1. Parse PDF
                if use_mock:
                    cv_text = "Placeholder "  # pymupdf4llm.to_markdown("./resume_example.pdf")
                    profile_data = STRUCTURED_CV  # extract_professional_structured_data(cv_text, api_key)    
                else:
                    if not api_key:
                        st.error("Please enter a Google API Key or select 'Use Mock Data'.")
                        st.stop()
                    
                    try:
                        cv_text = pymupdf4llm.to_markdown(uploaded_file)  # Assuming pymupdf4llm accepts file paths
                        profile_data = STRUCTURED_CV  # extract_professional_structured_data(cv_text, api_key)
                        
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
    
    st.title(f"📊 Market Technical Fit: {profile['main_position']['name']}")
    
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