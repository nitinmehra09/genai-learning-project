import os
from pathlib import Path
from dotenv import load_dotenv
from groq import Groq

from pypdf import PdfReader

def extract_pdf_text(file_path):
    reader = PdfReader(file_path)

    text = ""

    for page in reader.pages:
        text += page.extract_text() or ""
    if not text.strip():
        raise ValueError(f"No text could be extracted from {file_path}")

    return text

jd = extract_pdf_text("job_description.pdf")
resume = extract_pdf_text("resume2.pdf")

load_dotenv()
my_api_key = os.getenv("GROQ_API_KEY")

if not my_api_key:
    raise ValueError("Api key is not found")

client = Groq(api_key=my_api_key)
model = "openai/gpt-oss-120b"

def extract_info_from_jd():
    prompt = f"""
    Extract the key information from this job description:

    * Job Title & Company
    * Required skills/technologies
    * Preferred skills
    * Responsibilities
    * Experience & education requirements
    * Location & work mode
    * Salary, if mentioned
    * ATS keywords
    * Interview topics
    * Most important skills to qualify for the role

    Keep it concise and structured. Do not invent information not present in the JD.

    Job Description:
    {jd}
    """
    sys_prompt = """
    You are an expert Job Description Analyzer and Technical Recruiter.
    Analyze job descriptions accurately and extract only relevant, useful information. Be concise, structured, and evidence-based. Clearly separate facts explicitly mentioned in the JD from reasonable inferences. Never invent requirements, technologies, or responsibilities.
    Focus on:
    * Required vs preferred skills
    * Technical stack
    * Responsibilities
    * Qualifications and experience
    * ATS keywords
    * Interview topics
    * Role and seniority
    * Important skills candidates should prioritize
    Use simple language, avoid unnecessary explanations, and highlight the most important information first.
    """
    message = {
        "role": "user",
        "content": prompt
    }
    message_sys = {
        "role": "system",
        "content": sys_prompt
    }
    messages = [message_sys,message]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer
def extract_info_from_resume():
    prompt = f"""
    Extract the key information from this resume:
    * Name & contact details
    * Education
    * Technical skills
    * Programming languages
    * Frameworks & tools
    * Projects
    * Work/internship experience
    * Certifications
    * Achievements
    * Relevant domain expertise
    * Current experience level
    * Suitable job roles
    * ATS keywords
    Keep it concise and structured. Only use information present in the resume; do not invent anything.
    Resume:
    {resume}
    """
    sys_prompt = """
    You are an expert Resume Analyzer and Technical Recruiter.
    Analyze resumes accurately and extract useful career information. Be concise, structured, and evidence-based. Never invent skills, experience, projects, achievements, or qualifications that are not present in the resume.
    Focus on:
    * Education
    * Technical skills
    * Programming languages
    * Frameworks and tools
    * Projects
    * Work/internship experience
    * Certifications
    * Achievements
    * Relevant domain expertise
    * ATS keywords
    * Career level and target roles
    * Strengths and skill gaps
    Clearly distinguish between information explicitly stated in the resume and reasonable inferences. Prioritize information that is relevant for job matching and recruitment.
    """
    message = {
        "role": "user",
        "content": prompt
    }
    message_sys = {
        "role": "system",
        "content": sys_prompt
    }
    messages = [message_sys,message]

    response = client.chat.completions.create(
        model=model,
        messages=messages
    )

    answer = response.choices[0].message.content
    return answer


def compare_both_jd_or_resume(jd_info, resume_info):

    sys_prompt = """
    You are a strict Job Description vs Resume Matching Engine.
    Your task is to compare the provided JD information and Resume information.
    STRICT RULES:
    1. Use ONLY information explicitly present in the provided JD and Resume.
    2. NEVER guess, assume, infer, or fill missing information.
    3. Do NOT give credit for related or transferable skills.
    4. An exact skill match requires the skill to be explicitly present.
    5. Do NOT infer a skill from a project unless the skill itself is explicitly mentioned.
    6. If information is missing, mark it as "Not Mentioned".
    7. Every score must be supported by explicit evidence.
    8. Do not use external knowledge.
    9. Do not change the requirements of the JD.
    10. Do not use subjective language.
    SCORING:
    Required Skills Match = 40%
    Experience Match = 20%
    Education/Qualification Match = 15%
    Responsibilities Match = 10%
    Preferred Skills Match = 10%
    ATS Keyword Match = 5%
    Calculate each category from 0-100.
    Final ATS Score:
    (
    Required Skills Match × 0.40
    + Experience Match × 0.20
    + Qualification Match × 0.15
    + Responsibilities Match × 0.10
    + Preferred Skills Match × 0.10
    + Keyword Match × 0.05
    )
    Round the final score to the nearest whole number.
    IMPORTANT:
    A missing CRITICAL REQUIRED requirement automatically means:
    "Do Not Proceed"
    Recruiter Action rules:
    If ATS Score >= 80 AND there are NO missing critical required requirements:
    "Proceed to Human Review"
    Otherwise:
    "Do Not Proceed"
    Do not make the final hiring decision.
    The result is only an evidence-based screening recommendation.
    Return exactly this structure:
    ATS SCORE: X/100
    REQUIRED SKILLS MATCH: X/100
    EXPERIENCE MATCH: X/100
    QUALIFICATION MATCH: X/100
    RESPONSIBILITIES MATCH: X/100
    PREFERRED SKILLS MATCH: X/100
    KEYWORD MATCH: X/100
    MATCHED REQUIREMENTS:
    - requirement → exact resume evidence
    MISSING REQUIRED REQUIREMENTS:
    - requirement → "Not Mentioned"
    CRITICAL MISSING REQUIREMENTS:
    - requirement → "Not Mentioned"
    RESUME EVIDENCE:
    - evidence
    FINAL RESULT:
    Strong Match / Moderate Match / Weak Match
    RECRUITER ACTION:
    Proceed to Human Review / Do Not Proceed
    """

    prompt = f"""
    JOB DESCRIPTION INFORMATION:
    {jd_info}

    RESUME INFORMATION:
    {resume_info}

    Compare them strictly according to the system instructions.
    """

    messages = [
        {
            "role": "system",
            "content": sys_prompt
        },
        {
            "role": "user",
            "content": prompt
        }
    ]

    stream = client.chat.completions.create(
        model=model,
        messages=messages,
        stream = True
    )

    for chunk in stream:
        content = chunk.choices[0].delta.content
        if(content):
            print(content , end="" , flush=True)

jd_info = extract_info_from_jd()

resume_info = extract_info_from_resume()

compare_both_jd_or_resume(jd_info,resume_info)