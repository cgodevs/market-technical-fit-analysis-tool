-- DROP DATABASE IF EXISTS market_fit;

CREATE DATABASE market_fit
    WITH
    OWNER = postgres
    ENCODING = 'UTF8'
    LC_COLLATE = 'en_US.UTF-8'
    LC_CTYPE = 'en_US.UTF-8'
    LOCALE_PROVIDER = 'libc'
    TABLESPACE = pg_default
    CONNECTION LIMIT = -1
    IS_TEMPLATE = False;

COMMENT ON DATABASE market_fit
    IS 'Database dedicated to data served from the Market Fit Analysis tool';

CREATE TABLE job_postings (
    id                      VARCHAR(255) PRIMARY KEY,
    date_posted             DATE,
    date_created            DATE,
    title                   VARCHAR(255)        NOT NULL,
    description             TEXT                NOT NULL,
    url                     TEXT,
    country                 VARCHAR(100),
    location                VARCHAR(255),
    organization            VARCHAR(255),
    organization_logo       TEXT,
    linkedin_org_url        TEXT,
    weight                  SMALLINT            DEFAULT 1,
    c_source                VARCHAR(100),
    f_ai_min_seniority      VARCHAR(50),
    ai_experience_time_months   SMALLINT,
    ai_industries               VARCHAR(255)[]
);

-- Indexes for common query patterns
CREATE INDEX idx_job_postings_date_posted       ON job_postings (date_posted DESC);
CREATE INDEX idx_job_postings_date_created      ON job_postings (date_created DESC);
CREATE INDEX idx_job_postings_country           ON job_postings (country);


CREATE TABLE soft_skills (
    id          SERIAL PRIMARY KEY,
    job_id      VARCHAR(255) NOT NULL REFERENCES job_postings(id) ON DELETE CASCADE,
    skill_description TEXT NOT NULL,
    weight      FLOAT,
    embedding   VECTOR(3072)  -- gemini-embedding-001 output dimension
);

CREATE TABLE hard_skills (
    id              SERIAL PRIMARY KEY,
    job_id          VARCHAR(255) NOT NULL REFERENCES job_postings(id) ON DELETE CASCADE,
    skill_description TEXT NOT NULL,
    time_experience FLOAT,
    weight          FLOAT,
    embedding       VECTOR(3072)
);

CREATE TABLE work_industries (
    id              SERIAL PRIMARY KEY,
    title           VARCHAR(255) NOT NULL,
    embedding       VECTOR(3072)
);

CREATE TABLE resumes (
    id                      VARCHAR(255) PRIMARY KEY,
	user_id					VARCHAR(255),
	upload_id				VARCHAR(255),
	upload_date				TIMESTAMP,
	description				TEXT,
	industries				VARCHAR(255)[],
	position				VARCHAR(255),
	time_experience_months	SMALLINT,	
	position_embedding		VECTOR(3072)
);

CREATE TABLE candidate_hard_skills (
    id          SERIAL PRIMARY KEY,
    resume_id   VARCHAR(255) NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
	description TEXT,
    weight      FLOAT,
	time_experience_months SMALLINT,
    embedding   VECTOR(3072)  
);

CREATE TABLE candidate_soft_skills (
    id          SERIAL PRIMARY KEY,
    resume_id   VARCHAR(255) NOT NULL REFERENCES resumes(id) ON DELETE CASCADE,
	description TEXT,
    weight      FLOAT,
    embedding   VECTOR(3072)  
);
