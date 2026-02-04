import os
import pandas as pd
import plotly.express as px
from utils import *
import pymupdf4llm
import streamlit as st
from database_mock import JOBS_DB, STRUCTURED_CV, STRUCTURED_JOBS


# ========= Decide initial sidebar state BEFORE any UI =========
page = st.session_state.get('page', 'upload')
st.set_page_config(
    page_title="Technical Fit Analysis",
    layout="wide",
    initial_sidebar_state="collapsed" if page == 'dashboard' else "expanded"
)

# ========= Initialize Session State =========
if 'page' not in st.session_state:
    st.session_state.page = 'upload'
if 'profile' not in st.session_state:
    st.session_state.profile = None
if 'cv_image' not in st.session_state:
    st.session_state.cv_image = None
if 'usage' not in st.session_state:
    st.session_state.usage = None

if st.session_state.page == 'dashboard' and st.session_state.usage == 'mock':
    resume_path = os.path.join(os.path.dirname(__file__), "resume_example.pdf")
    with open(resume_path, "rb") as local_resume:
        st.session_state.cv_image = pdf_first_page_as_image(local_resume)

# ========= SIDEBAR =========
with st.sidebar:
    st.header("⚙️ Configuration")
    api_key = st.text_input("Google API Key", type="password", help="Required for live analysis using Gemini.")
    st.space("small")
    st.markdown("---")

    # Show first page image if available
    if st.session_state.cv_image:
        st.image(st.session_state.cv_image, caption="Uploaded CV", use_container_width=True)

# ========= PAGE 1: UPLOAD =========
if st.session_state.page == 'upload':
    st.title("Market Technical Fit Analysis for your Resume", text_alignment="center")
    st.space("medium")
    col1, col2, col3 = st.columns([2, 1, 2])
    with col2:
        if st.button("Use Mock Data", type="secondary", icon="🧪", use_container_width=True):
            st.session_state.usage = 'mock'
            st.session_state.page = 'dashboard'
            st.rerun()

    uploaded_file = st.file_uploader("Upload your PDF Resume", type="pdf", width="stretch")
    if uploaded_file:
        st.session_state.cv_image = pdf_first_page_as_image(uploaded_file)

    if uploaded_file and st.button("Analyze CV"):
        with st.spinner("Extracting Skills & analyzing Market Fit..."):
            if not api_key:
                st.error("Please enter a Google API Key or select 'Use Mock Data'.")
                st.stop()
            try:
                # cv_text = pymupdf4llm.to_markdown(uploaded_file)  
                profile_data = STRUCTURED_CV  # TODO: extract from cv_text & api_key
            except Exception as e:
                st.error(f"Error reading PDF: {e}")
                st.stop()
            if profile_data:
                st.session_state.profile = profile_data
                st.session_state.usage = 'live'
                st.session_state.page = 'dashboard'
                st.rerun()

# ========= PAGE 2: DASHBOARD =========
elif st.session_state.page == 'dashboard':
    usage = st.session_state.usage

    if usage == 'mock':
        profile = STRUCTURED_CV  # Using mock data
        market_data = STRUCTURED_JOBS
    else:
        profile = st.session_state.profile
        market_data = STRUCTURED_JOBS  # call API # TODO

    applicant_skills = set([s['description'].lower() for s in profile['hard_skills']])

    # A. Match Analysis per Job
    job_matches = []
    for job in market_data:
        job_hard_skills = set([skill['description'].lower() for skill in job['hard_skills']])
        match = applicant_skills.intersection(job_hard_skills)
        missing = job_hard_skills.difference(applicant_skills)

        job_matches.append({
            "Job Title": job['main_position']['name'],
            "Company": job['company_name'],
            "Match Count": len(match),
            "Total Job Skills": len(job_hard_skills),
            "Match %": round(len(match) / len(job_hard_skills) * 100, 1) if len(job_hard_skills) > 0 else 0,
            "Missing Skills": ", ".join(list(missing))
        })
    df_matches = pd.DataFrame(job_matches).sort_values("Match Count", ascending=False)

    # B. Popular Applicant Skills
    app_skill_counts = {}
    for skill in applicant_skills:
        count = sum(1 for job in market_data if skill in {s['description'].lower() for s in job['hard_skills']})
        if count > 0:
            app_skill_counts[skill] = count
    df_app_pop = pd.DataFrame(list(app_skill_counts.items()),
                              columns=['Skill', 'Number of Jobs Demands']).sort_values('Number of Jobs Demands', ascending=True)

    # C. Missing Skills Gap Analysis
    missing_counter = {}
    total_jobs = len(market_data)
    for job in market_data:
        for skill in {s['description'].lower() for s in job['hard_skills']}:
            if skill not in applicant_skills:
                missing_counter[skill] = missing_counter.get(skill, 0) + 1

    gap_data = []
    for skill, count in missing_counter.items():
        if count > 1:
            gap_data.append({
                "Skill": skill.title(),
                "Frequency": count,
                "% of Market": round((count / total_jobs) * 100, 1)
            })
    df_gaps = pd.DataFrame(gap_data).sort_values("% of Market", ascending=False).head(10)

    # ========= DASHBOARD UI =========
    st.button("← Upload New CV", on_click=lambda: st.session_state.update(page='upload'))

    st.title("Technical Fit Market Analysis for:", text_alignment="center")
    st.subheader(f":blue[{profile['main_position']['name']}]", text_alignment="center")
    st.space("medium")

    # Top Metrics
    c1, c2, c3 = st.columns(3)
    with c1.container(height="content"):
        with st.popover(f"Hard Skills Identified: ({len(applicant_skills)})", use_container_width=True):
            skills_md = "  \n• ".join(["All Skills:  "] + sorted(applicant_skills))
            st.markdown(skills_md)
    with c2.container(height="content", border=True):
        st.metric("Open Positions Analyzed", len(JOBS_DB))
    with c3.container(height="content", border=True):
        st.metric("Avg. Match Score", f"{df_matches['Match %'].mean():.1f}%")

    st.divider()

    # ROW 1: Charts
    col1, col2 = st.columns(2)
    with col1:
        st.subheader("✅ Your Most Valued Skills")
        st.caption("Which of your skills are most in demand across open positions?")
        if not df_app_pop.empty:
            fig1 = px.bar(df_app_pop, x='Number of Jobs Demands', y='Skill',
                          orientation='h', text='Number of Jobs Demands',
                          color_discrete_sequence=['#4CAF50'])
            st.plotly_chart(fig1, use_container_width=True)
        else:
            st.warning("No overlaps found with current market data.")
    with col2:
        st.subheader("⚠️ Skill Gaps")
        st.caption("Skills you are missing that appear most frequently in job listings.")
        if not df_gaps.empty:
            fig2 = px.bar(df_gaps, x='% of Market', y='Skill',
                          orientation='h', text='% of Market',
                          color_discrete_sequence=['#FF5252'])
            fig2.update_traces(texttemplate='%{text}%')
            st.plotly_chart(fig2, use_container_width=True)
        else:
            st.success("You are a perfect match for the current market! No major gaps found.")

    st.divider()

    # ROW 2: Detailed matches
    st.subheader("🏆 Overall Best Matching Jobs")
    st.caption("Ranked by total count of matching hard skills.")

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