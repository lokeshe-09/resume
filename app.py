import streamlit as st
import base64
import os
import pandas as pd
import re
from google import genai
from google.genai import types
import PyPDF2
from io import BytesIO
import json
from datetime import datetime

# Set your API key here
GEMINI_API_KEY = "AIzaSyCup89v2NbecBFc1oVYLFBoPhfpvCzrVGk"  # Replace with your actual API key

# Initialize Gemini client
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

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

def generate_response(client, prompt, model="gemini-2.0-flash"):
    """Generate response from Gemini API"""
    try:
        contents = [
            types.Content(
                role="user",
                parts=[
                    types.Part.from_text(text=prompt),
                ],
            ),
        ]
        
        generate_content_config = types.GenerateContentConfig(
            response_mime_type="text/plain",
        )
        
        response_text = ""
        for chunk in client.models.generate_content_stream(
            model=model,
            contents=contents,
            config=generate_content_config,
        ):
            response_text += chunk.text
        
        return response_text
    except Exception as e:
        return f"Error generating response: {str(e)}"

def extract_skills_from_job_description(client, job_description):
    """Extract skills from job description using Gemini"""
    prompt = f"""
    Extract all technical skills, soft skills, and qualifications from the following job description.
    Return only a comma-separated list of skills without any additional text or formatting.
    
    Job Description:
    {job_description}
    """
    
    response = generate_response(client, prompt)
    skills = [skill.strip() for skill in response.split(',') if skill.strip()]
    return skills

def analyze_resume(client, resume_text, job_description, required_skills):
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
            "skill_name": true/false for each required skill
        }},
        "overall_match_percentage": "Percentage match (just the number)",
        "summary": "Brief summary of the candidate's profile"
    }}
    
    Return only the JSON without any additional text or formatting.
    """
    
    response = generate_response(client, prompt)
    try:
        # Clean the response to extract JSON
        json_start = response.find('{')
        json_end = response.rfind('}') + 1
        if json_start != -1 and json_end != -1:
            json_str = response[json_start:json_end]
            return json.loads(json_str)
    except:
        pass
    
    # Fallback if JSON parsing fails
    return {
        "candidate_name": "Unknown",
        "experience_years": "0",
        "skills_match": {skill: False for skill in required_skills},
        "overall_match_percentage": "0",
        "summary": "Analysis failed"
    }

def main():
    st.set_page_config(
        page_title="AI Assistant & Resume Analyzer",
        page_icon="🤖",
        layout="wide"
    )
    
    st.title("🤖 AI Assistant & Resume Analyzer")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chatbot", "📄 Resume Analyzer"])
    
    # Initialize Gemini client
    client = get_gemini_client()
    
    with tab1:
        st.header("💬 AI Chatbot")
        st.write("Chat with AI assistant - Your conversation history is maintained!")
        
        # Chat interface
        chat_container = st.container()
        
        # Display chat history
        with chat_container:
            for i, (user_msg, bot_msg, timestamp) in enumerate(st.session_state.chat_history):
                with st.chat_message("user"):
                    st.write(f"**You:** {user_msg}")
                    st.caption(timestamp)
                
                with st.chat_message("assistant"):
                    st.write(f"**AI:** {bot_msg}")
        
        # Chat input
        user_input = st.chat_input("Type your message here...")
        
        if user_input:
            # Add user message to history
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate AI response
            with st.spinner("AI is thinking..."):
                # Include chat history for context
                context = ""
                if st.session_state.chat_history:
                    context = "Previous conversation:\n"
                    for user_msg, bot_msg, _ in st.session_state.chat_history[-5:]:  # Last 5 exchanges
                        context += f"User: {user_msg}\nAI: {bot_msg}\n\n"
                
                full_prompt = f"{context}Current question: {user_input}"
                ai_response = generate_response(client, full_prompt)
            
            # Add to chat history
            st.session_state.chat_history.append((user_input, ai_response, timestamp))
            
            # Rerun to update the display
            st.rerun()
        
        # Clear chat button
        if st.button("🗑️ Clear Chat History"):
            st.session_state.chat_history = []
            st.rerun()
    
    with tab2:
        st.header("📄 Resume Analyzer")
        st.write("Analyze multiple resumes against a job description")
        
        # Job description input
        st.subheader("1. Enter Job Description")
        job_description = st.text_area(
            "Paste the job description here:",
            height=200,
            placeholder="Enter the complete job description including required skills, qualifications, and experience..."
        )
        
        # File upload
        st.subheader("2. Upload Resume Files")
        uploaded_files = st.file_uploader(
            "Upload PDF resumes (multiple files allowed)",
            type=['pdf'],
            accept_multiple_files=True
        )
        
        if st.button("🔍 Analyze Resumes", type="primary"):
            if not job_description.strip():
                st.error("Please enter a job description first!")
            elif not uploaded_files:
                st.error("Please upload at least one resume!")
            else:
                with st.spinner("Analyzing resumes... This may take a few minutes."):
                    # Extract skills from job description
                    st.info("Extracting skills from job description...")
                    required_skills = extract_skills_from_job_description(client, job_description)
                    
                    st.success(f"Found {len(required_skills)} required skills:")
                    st.write(", ".join(required_skills))
                    
                    # Analyze each resume
                    results = []
                    progress_bar = st.progress(0)
                    
                    for i, uploaded_file in enumerate(uploaded_files):
                        st.info(f"Analyzing resume {i+1}/{len(uploaded_files)}: {uploaded_file.name}")
                        
                        # Extract text from PDF
                        resume_text = extract_text_from_pdf(uploaded_file)
                        
                        if resume_text.strip():
                            # Analyze resume
                            analysis = analyze_resume(client, resume_text, job_description, required_skills)
                            analysis['file_name'] = uploaded_file.name
                            results.append(analysis)
                        else:
                            st.warning(f"Could not extract text from {uploaded_file.name}")
                        
                        progress_bar.progress((i + 1) / len(uploaded_files))
                    
                    if results:
                        # Sort by overall match percentage
                        results.sort(key=lambda x: float(x.get('overall_match_percentage', 0)), reverse=True)
                        
                        st.success("Analysis completed!")
                        
                        # Display results
                        st.subheader("📊 Analysis Results")
                        
                        # Create summary table
                        summary_data = []
                        for result in results:
                            row = {
                                'Rank': len(summary_data) + 1,
                                'Candidate Name': result.get('candidate_name', 'Unknown'),
                                'File Name': result.get('file_name', ''),
                                'Experience (Years)': result.get('experience_years', '0'),
                                'Overall Match (%)': result.get('overall_match_percentage', '0'),
                                'Summary': result.get('summary', '')
                            }
                            summary_data.append(row)
                        
                        summary_df = pd.DataFrame(summary_data)
                        st.dataframe(summary_df, use_container_width=True)
                        
                        # Skills matrix
                        st.subheader("🎯 Skills Matrix")
                        
                        skills_data = []
                        for result in results:
                            row = {
                                'Candidate': result.get('candidate_name', 'Unknown'),
                                'Experience': result.get('experience_years', '0'),
                                'Match %': result.get('overall_match_percentage', '0')
                            }
                            
                            # Add skills columns
                            skills_match = result.get('skills_match', {})
                            for skill in required_skills:
                                row[skill] = "✅" if skills_match.get(skill, False) else "❌"
                            
                            skills_data.append(row)
                        
                        skills_df = pd.DataFrame(skills_data)
                        st.dataframe(skills_df, use_container_width=True)
                        
                        # Top candidates
                        st.subheader("🏆 Top Matching Candidates")
                        
                        for i, result in enumerate(results[:3]):  # Top 3 candidates
                            with st.expander(f"#{i+1} - {result.get('candidate_name', 'Unknown')} ({result.get('overall_match_percentage', '0')}% match)"):
                                col1, col2 = st.columns(2)
                                
                                with col1:
                                    st.write(f"**File:** {result.get('file_name', '')}")
                                    st.write(f"**Experience:** {result.get('experience_years', '0')} years")
                                    st.write(f"**Match Percentage:** {result.get('overall_match_percentage', '0')}%")
                                
                                with col2:
                                    st.write("**Skills Match:**")
                                    skills_match = result.get('skills_match', {})
                                    matched_skills = [skill for skill, match in skills_match.items() if match]
                                    missing_skills = [skill for skill, match in skills_match.items() if not match]
                                    
                                    if matched_skills:
                                        st.success(f"✅ Has: {', '.join(matched_skills)}")
                                    if missing_skills:
                                        st.error(f"❌ Missing: {', '.join(missing_skills)}")
                                
                                st.write(f"**Summary:** {result.get('summary', '')}")
                        
                        # Download results
                        st.subheader("📥 Download Results")
                        
                        # Create Excel file
                        output = BytesIO()
                        with pd.ExcelWriter(output, engine='openpyxl') as writer:
                            summary_df.to_excel(writer, sheet_name='Summary', index=False)
                            skills_df.to_excel(writer, sheet_name='Skills Matrix', index=False)
                        
                        st.download_button(
                            label="📊 Download Analysis Report (Excel)",
                            data=output.getvalue(),
                            file_name=f"resume_analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )
                    
                    else:
                        st.error("No resumes could be analyzed. Please check your PDF files.")

if __name__ == "__main__":
    main()
