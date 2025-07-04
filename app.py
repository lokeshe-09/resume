import streamlit as st
import os
import pandas as pd
import re
import google.generativeai as genai
import PyPDF2
from io import BytesIO
import json
from datetime import datetime

# Get API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY environment variable not found! Please set your API key.")
    st.info("Set your environment variable: `export GEMINI_API_KEY=your_api_key_here`")
    st.stop()

# Configure Gemini
genai.configure(api_key=GEMINI_API_KEY)

@st.cache_resource
def get_gemini_model():
    """Initialize Gemini model"""
    try:
        return genai.GenerativeModel('gemini-1.5-flash')
    except Exception as e:
        st.error(f"Error initializing Gemini model: {str(e)}")
        return None

def extract_text_from_pdf(pdf_file):
    """Extract text from uploaded PDF file"""
    try:
        pdf_reader = PyPDF2.PdfReader(BytesIO(pdf_file.read()))
        text = ""
        for page in pdf_reader.pages:
            text += page.extract_text() + "\n"
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {str(e)}")
        return ""

def generate_response(model, prompt):
    """Generate response from Gemini API"""
    try:
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_skills_from_job_description(model, job_description):
    """Extract skills from job description using Gemini"""
    prompt = f"""
    Extract all technical skills, soft skills, and qualifications from the following job description.
    Return only a comma-separated list of skills without any additional text or formatting.
    
    Job Description:
    {job_description}
    """
    
    response = generate_response(model, prompt)
    skills = [skill.strip() for skill in response.split(',') if skill.strip()]
    return skills

def analyze_resume(model, resume_text, job_description, required_skills):
    """Analyze resume against job description and extract information"""
    prompt = f"""
    Analyze the following resume against the job description and required skills.
    
    Job Description: {job_description}
    
    Required Skills: {', '.join(required_skills)}
    
    Resume Text: {resume_text}
    
    Please provide the analysis in the following JSON format:
    {{
        "candidate_name": "Name of the candidate",
        "experience_years": "Number of years of experience (just the number)",
        "skills_match": {{
            "{required_skills[0] if required_skills else 'skill1'}": true,
            "{required_skills[1] if len(required_skills) > 1 else 'skill2'}": false
        }},
        "overall_match_percentage": "Percentage match (just the number without % symbol)",
        "summary": "Brief summary of the candidate's profile"
    }}
    
    For skills_match, include all required skills and mark each as true or false based on whether the candidate has that skill.
    Return only the JSON without any additional text or formatting.
    """
    
    response = generate_response(model, prompt)
    try:
        # Clean the response to extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            parsed_json = json.loads(json_str)
            
            # Ensure all required skills are in skills_match
            if 'skills_match' not in parsed_json:
                parsed_json['skills_match'] = {}
            
            for skill in required_skills:
                if skill not in parsed_json['skills_match']:
                    parsed_json['skills_match'][skill] = False
            
            return parsed_json
    except Exception as e:
        st.warning(f"JSON parsing failed for a resume: {str(e)}")
    
    # Fallback if JSON parsing fails
    return {
        "candidate_name": "Unknown",
        "experience_years": "0",
        "skills_match": {skill: False for skill in required_skills},
        "overall_match_percentage": "0",
        "summary": "Analysis failed - could not parse resume properly"
    }

def apply_custom_styles():
    """Apply custom CSS for professional look"""
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    
    /* Main container styling */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
        max-width: 1200px;
    }
    
    /* Header styling */
    .main-header {
        text-align: center;
        padding: 2rem 0;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 10px;
        margin-bottom: 2rem;
    }
    
    /* Section styling */
    .section-header {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 1rem 0;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #007bff;
        border-radius: 10px;
        padding: 2rem;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 10px;
        border: none;
        padding: 0.75rem 2rem;
        font-weight: 600;
        font-size: 1.1rem;
        transition: all 0.3s;
        width: 100%;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    
    /* Text area styling */
    .stTextArea > div > div > textarea {
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        font-family: 'Arial', sans-serif;
    }
    
    .stTextArea > div > div > textarea:focus {
        border-color: #007bff;
        box-shadow: 0 0 0 0.2rem rgba(0,123,255,.25);
    }
    
    /* DataFrame styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }
    
    /* Success/Error message styling */
    .stSuccess, .stError, .stWarning, .stInfo {
        border-radius: 10px;
        padding: 1rem;
        margin: 1rem 0;
    }
    
    /* Progress bar styling */
    .stProgress > div > div > div > div {
        background-color: #007bff;
        border-radius: 10px;
    }
    
    /* Expander styling */
    .streamlit-expanderHeader {
        background-color: #f8f9fa;
        border-radius: 10px;
        border: 1px solid #e0e0e0;
    }
    
    /* Metric styling */
    .metric-container {
        background-color: #f8f9fa;
        padding: 1rem;
        border-radius: 10px;
        text-align: center;
        border: 1px solid #e0e0e0;
    }
    </style>
    """, unsafe_allow_html=True)

def display_metrics(results):
    """Display key metrics about the analysis"""
    if not results:
        return
    
    total_candidates = len(results)
    avg_match = sum(float(str(r.get('overall_match_percentage', 0)).replace('%', '')) for r in results) / total_candidates
    top_match = max(float(str(r.get('overall_match_percentage', 0)).replace('%', '')) for r in results)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #007bff; margin: 0;">📊 Total Candidates</h3>
            <h2 style="margin: 0.5rem 0;">{total_candidates}</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #28a745; margin: 0;">📈 Average Match</h3>
            <h2 style="margin: 0.5rem 0;">{avg_match:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown(f"""
        <div class="metric-container">
            <h3 style="color: #ffc107; margin: 0;">🏆 Top Match</h3>
            <h2 style="margin: 0.5rem 0;">{top_match:.1f}%</h2>
        </div>
        """, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="Resume Analyzer - AI Powered",
        page_icon="📄",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styles
    apply_custom_styles()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📄 AI-Powered Resume Analyzer</h1>
        <p>Analyze multiple resumes against job descriptions with advanced AI technology</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Initialize Gemini model
    model = get_gemini_model()
    if not model:
        st.error("Failed to initialize Gemini model. Please check your API key and try again.")
        st.stop()
    
    # Job description section
    st.markdown('<div class="section-header"><h2>🎯 Step 1: Enter Job Description</h2></div>', unsafe_allow_html=True)
    
    job_description = st.text_area(
        "Paste the complete job description here:",
        height=250,
        placeholder="""Example:
We are looking for a Senior Software Engineer with:
- 5+ years of Python development experience
- Experience with React and JavaScript
- Knowledge of AWS cloud services
- Strong problem-solving skills
- Bachelor's degree in Computer Science
- Experience with Docker and Kubernetes
- Excellent communication skills""",
        help="Include all requirements, skills, qualifications, and experience needed for the role"
    )
    
    # File upload section
    st.markdown('<div class="section-header"><h2>📁 Step 2: Upload Resume Files</h2></div>', unsafe_allow_html=True)
    
    uploaded_files = st.file_uploader(
        "Upload PDF resumes (multiple files supported)",
        type=['pdf'],
        accept_multiple_files=True,
        help="Select multiple PDF files to analyze them all at once"
    )
    
    # Display uploaded files info
    if uploaded_files:
        st.success(f"✅ {len(uploaded_files)} resume(s) uploaded successfully!")
        with st.expander("📋 View uploaded files"):
            for i, file in enumerate(uploaded_files, 1):
                st.write(f"{i}. {file.name} ({file.size} bytes)")
    
    # Analysis section
    st.markdown('<div class="section-header"><h2>🔍 Step 3: Start Analysis</h2></div>', unsafe_allow_html=True)
    
    if st.button("🚀 Analyze Resumes", type="primary"):
        if not job_description.strip():
            st.error("❌ Please enter a job description first!")
        elif not uploaded_files:
            st.error("❌ Please upload at least one resume!")
        else:
            # Create analysis container
            analysis_container = st.container()
            
            with analysis_container:
                with st.spinner("🔄 Starting analysis... This may take a few minutes."):
                    try:
                        # Step 1: Extract skills from job description
                        st.info("🎯 Extracting required skills from job description...")
                        required_skills = extract_skills_from_job_description(model, job_description)
                        
                        if required_skills:
                            st.success(f"✅ Found {len(required_skills)} required skills")
                            with st.expander("📋 View extracted skills"):
                                skills_text = ", ".join(required_skills)
                                st.write(skills_text)
                        else:
                            st.warning("⚠️ Could not extract skills from job description. Using default analysis.")
                            required_skills = ["Communication", "Problem Solving", "Technical Skills", "Experience"]
                        
                        # Step 2: Analyze each resume
                        st.info("📄 Analyzing resumes...")
                        results = []
                        progress_bar = st.progress(0)
                        status_text = st.empty()
                        
                        for i, uploaded_file in enumerate(uploaded_files):
                            status_text.text(f"Analyzing resume {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                            
                            # Reset file pointer
                            uploaded_file.seek(0)
                            
                            # Extract text from PDF
                            resume_text = extract_text_from_pdf(uploaded_file)
                            
                            if resume_text.strip():
                                # Analyze resume
                                analysis = analyze_resume(model, resume_text, job_description, required_skills)
                                analysis['file_name'] = uploaded_file.name
                                results.append(analysis)
                            else:
                                st.warning(f"⚠️ Could not extract text from {uploaded_file.name}")
                            
                            progress_bar.progress((i + 1) / len(uploaded_files))
                        
                        status_text.empty()
                        
                        if results:
                            # Sort by overall match percentage
                            results.sort(key=lambda x: float(str(x.get('overall_match_percentage', 0)).replace('%', '')), reverse=True)
                            
                            st.success("🎉 Analysis completed successfully!")
                            
                            # Display metrics
                            st.markdown("---")
                            st.markdown("## 📊 Analysis Overview")
                            display_metrics(results)
                            
                            # Results section
                            st.markdown("---")
                            st.markdown("## 📋 Detailed Results")
                            
                            # Summary table
                            st.subheader("📈 Candidate Summary")
                            summary_data = []
                            for result in results:
                                row = {
                                    'Rank': len(summary_data) + 1,
                                    'Candidate Name': result.get('candidate_name', 'Unknown'),
                                    'File Name': result.get('file_name', ''),
                                    'Experience (Years)': result.get('experience_years', '0'),
                                    'Overall Match (%)': f"{result.get('overall_match_percentage', '0')}%",
                                    'Summary': result.get('summary', '')
                                }
                                summary_data.append(row)
                            
                            summary_df = pd.DataFrame(summary_data)
                            st.dataframe(summary_df, use_container_width=True, hide_index=True)
                            
                            # Skills matrix
                            st.subheader("🎯 Skills Matching Matrix")
                            skills_data = []
                            for result in results:
                                row = {
                                    'Candidate': result.get('candidate_name', 'Unknown'),
                                    'Experience': f"{result.get('experience_years', '0')} years",
                                    'Match %': f"{result.get('overall_match_percentage', '0')}%"
                                }
                                
                                # Add skills columns
                                skills_match = result.get('skills_match', {})
                                for skill in required_skills:
                                    row[skill] = "✅ Yes" if skills_match.get(skill, False) else "❌ No"
                                
                                skills_data.append(row)
                            
                            skills_df = pd.DataFrame(skills_data)
                            st.dataframe(skills_df, use_container_width=True, hide_index=True)
                            
                            # Top candidates detailed view
                            st.subheader("🏆 Top Matching Candidates")
                            
                            for i, result in enumerate(results[:5]):  # Top 5 candidates
                                match_percentage = float(str(result.get('overall_match_percentage', 0)).replace('%', ''))
                                
                                # Color coding based on match percentage
                                if match_percentage >= 80:
                                    color = "🟢"
                                elif match_percentage >= 60:
                                    color = "🟡"
                                else:
                                    color = "🔴"
                                
                                with st.expander(f"{color} #{i+1} - {result.get('candidate_name', 'Unknown')} ({result.get('overall_match_percentage', '0')}% match)"):
                                    col1, col2 = st.columns(2)
                                    
                                    with col1:
                                        st.markdown("**📄 Basic Information**")
                                        st.write(f"**File:** {result.get('file_name', '')}")
                                        st.write(f"**Experience:** {result.get('experience_years', '0')} years")
                                        st.write(f"**Match Percentage:** {result.get('overall_match_percentage', '0')}%")
                                    
                                    with col2:
                                        st.markdown("**🎯 Skills Analysis**")
                                        skills_match = result.get('skills_match', {})
                                        matched_skills = [skill for skill, match in skills_match.items() if match]
                                        missing_skills = [skill for skill, match in skills_match.items() if not match]
                                        
                                        if matched_skills:
                                            st.success(f"✅ **Matching Skills:** {', '.join(matched_skills)}")
                                        if missing_skills:
                                            st.error(f"❌ **Missing Skills:** {', '.join(missing_skills)}")
                                    
                                    st.markdown("**📝 Summary**")
                                    st.info(result.get('summary', 'No summary available'))
                            
                            # Download section
                            st.markdown("---")
                            st.subheader("📥 Download Analysis Report")
                            
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                # Create Excel file
                                output = BytesIO()
                                with pd.ExcelWriter(output, engine='openpyxl') as writer:
                                    summary_df.to_excel(writer, sheet_name='Summary', index=False)
                                    skills_df.to_excel(writer, sheet_name='Skills Matrix', index=False)
                                    
                                    # Add job description sheet
                                    job_desc_df = pd.DataFrame({
                                        'Job Description': [job_description],
                                        'Required Skills': [', '.join(required_skills)],
                                        'Analysis Date': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
                                    })
                                    job_desc_df.to_excel(writer, sheet_name='Job Description', index=False)
                                
                                st.download_button(
                                    label="📊 Download Excel Report",
                                    data=output.getvalue(),
                                    file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                                    use_container_width=True
                                )
                            
                            with col2:
                                # Create CSV file
                                csv_output = BytesIO()
                                summary_df.to_csv(csv_output, index=False)
                                
                                st.download_button(
                                    label="📄 Download CSV Report",
                                    data=csv_output.getvalue(),
                                    file_name=f"resume_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                                    mime="text/csv",
                                    use_container_width=True
                                )
                        
                        else:
                            st.error("❌ No resumes could be analyzed. Please check your PDF files and try again.")
                    
                    except Exception as e:
                        st.error(f"❌ An error occurred during analysis: {str(e)}")
                        st.info("💡 Please try again or check your API key and internet connection.")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 2rem;">
        <p>🤖 Powered by Google Gemini AI | Built with Streamlit</p>
        <p>📧 For support or feedback, please contact your administrator</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
