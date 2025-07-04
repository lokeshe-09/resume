import streamlit as st
import base64
import os
import pandas as pd
import re
import google.generativeai as genai
from google.generativeai.types import Content, Part
import PyPDF2
from io import BytesIO
import json
from datetime import datetime
from PIL import Image
import io

# Get API key from environment variables
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    st.error("⚠️ GEMINI_API_KEY environment variable not found! Please set your API key.")
    st.info("Set your environment variable: `export GEMINI_API_KEY=your_api_key_here`")
    st.stop()

# Initialize Gemini client
@st.cache_resource
def get_gemini_client():
    return genai.Client(api_key=GEMINI_API_KEY)

def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return img_str

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

def generate_response(client, prompt, image=None, model="gemini-2.0-flash"):
    """Generate response from Gemini API with optional image"""
    try:
        parts = [types.Part.from_text(text=prompt)]
        
        # Add image if provided
        if image is not None:
            img_base64 = image_to_base64(image)
            parts.append(
                types.Part.from_bytes(
                    data=base64.b64decode(img_base64),
                    mime_type="image/png"
                )
            )
        
        contents = [
            types.Content(
                role="user",
                parts=parts,
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

def apply_chat_styles():
    """Apply custom CSS for ChatGPT-like interface"""
    st.markdown("""
    <style>
    /* Hide default Streamlit elements */
    .stDeployButton {display:none;}
    footer {visibility: hidden;}
    .stDecoration {display:none;}
    
    /* Chat container styling */
    .chat-container {
        height: 60vh;
        overflow-y: auto;
        padding: 1rem;
        border: 1px solid #e0e0e0;
        border-radius: 10px;
        background-color: #fafafa;
        margin-bottom: 1rem;
    }
    
    /* Message styling */
    .user-message {
        background-color: #007bff;
        color: white;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-left: 20%;
        text-align: right;
    }
    
    .assistant-message {
        background-color: #f1f3f4;
        color: #333;
        padding: 0.75rem 1rem;
        border-radius: 18px;
        margin: 0.5rem 0;
        margin-right: 20%;
    }
    
    .message-timestamp {
        font-size: 0.7rem;
        opacity: 0.7;
        margin-top: 0.25rem;
    }
    
    /* Input area styling */
    .input-container {
        position: fixed;
        bottom: 0;
        left: 0;
        right: 0;
        background-color: white;
        padding: 1rem;
        border-top: 1px solid #e0e0e0;
        z-index: 999;
    }
    
    /* Adjust main content to account for fixed input */
    .main-content {
        padding-bottom: 120px;
    }
    
    /* Image preview styling */
    .image-preview {
        max-width: 200px;
        max-height: 200px;
        border-radius: 10px;
        margin: 0.5rem 0;
    }
    
    /* File uploader styling */
    .stFileUploader > div > div > div > div {
        background-color: #f8f9fa;
        border: 2px dashed #dee2e6;
        border-radius: 10px;
        padding: 1rem;
        text-align: center;
    }
    
    /* Button styling */
    .stButton > button {
        background-color: #007bff;
        color: white;
        border-radius: 20px;
        border: none;
        padding: 0.5rem 1.5rem;
        font-weight: 500;
    }
    
    .stButton > button:hover {
        background-color: #0056b3;
    }
    
    /* Tab styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 2rem;
    }
    
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        padding: 0 1.5rem;
        background-color: #f8f9fa;
        border-radius: 10px 10px 0 0;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #007bff;
        color: white;
    }
    </style>
    """, unsafe_allow_html=True)

def display_chat_message(message_type, content, timestamp, image=None):
    """Display a chat message with proper styling"""
    if message_type == "user":
        message_class = "user-message"
        prefix = "You"
    else:
        message_class = "assistant-message"
        prefix = "AI Assistant"
    
    message_html = f"""
    <div class="{message_class}">
        <strong>{prefix}:</strong><br>
        {content.replace('\n', '<br>')}
        {f'<br><img src="data:image/png;base64,{image}" class="image-preview">' if image else ''}
        <div class="message-timestamp">{timestamp}</div>
    </div>
    """
    st.markdown(message_html, unsafe_allow_html=True)

def main():
    st.set_page_config(
        page_title="AI Assistant & Resume Analyzer",
        page_icon="🤖",
        layout="wide",
        initial_sidebar_state="collapsed"
    )
    
    # Apply custom styles
    apply_chat_styles()
    
    st.title("🤖 AI Assistant & Resume Analyzer")
    
    # Initialize session state
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'uploaded_image' not in st.session_state:
        st.session_state.uploaded_image = None
    if 'image_preview' not in st.session_state:
        st.session_state.image_preview = None
    
    # Create tabs
    tab1, tab2 = st.tabs(["💬 Chatbot", "📄 Resume Analyzer"])
    
    # Initialize Gemini client
    client = get_gemini_client()
    
    with tab1:
        st.markdown('<div class="main-content">', unsafe_allow_html=True)
        
        st.header("💬 AI Chatbot")
        st.write("Chat with AI assistant - Upload images and ask questions!")
        
        # Chat history container
        chat_container = st.container()
        
        with chat_container:
            st.markdown('<div class="chat-container">', unsafe_allow_html=True)
            
            # Display chat history
            for i, chat_item in enumerate(st.session_state.chat_history):
                if len(chat_item) == 4:  # With image
                    user_msg, bot_msg, timestamp, image_b64 = chat_item
                    display_chat_message("user", user_msg, timestamp, image_b64)
                else:  # Without image
                    user_msg, bot_msg, timestamp = chat_item
                    display_chat_message("user", user_msg, timestamp)
                
                display_chat_message("assistant", bot_msg, timestamp)
            
            st.markdown('</div>', unsafe_allow_html=True)
        
        # Fixed input area
        st.markdown('<div class="input-container">', unsafe_allow_html=True)
        
        # Image upload section
        col1, col2 = st.columns([3, 1])
        
        with col1:
            uploaded_image = st.file_uploader(
                "Upload an image (optional)",
                type=['png', 'jpg', 'jpeg', 'gif', 'bmp'],
                key="chat_image_uploader",
                help="Upload an image to analyze along with your text prompt"
            )
        
        with col2:
            if uploaded_image:
                image = Image.open(uploaded_image)
                st.image(image, caption="Uploaded Image", width=100)
                st.session_state.uploaded_image = image
                st.session_state.image_preview = image_to_base64(image)
            elif st.session_state.uploaded_image:
                st.image(st.session_state.uploaded_image, caption="Current Image", width=100)
        
        # Clear image button
        if st.session_state.uploaded_image:
            if st.button("🗑️ Remove Image"):
                st.session_state.uploaded_image = None
                st.session_state.image_preview = None
                st.rerun()
        
        # Chat input
        user_input = st.text_input(
            "Type your message here...",
            key="chat_input",
            placeholder="Ask me anything or upload an image for analysis..."
        )
        
        # Send button
        col1, col2, col3 = st.columns([1, 1, 1])
        with col2:
            send_button = st.button("📤 Send Message", key="send_btn", use_container_width=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Process message
        if (send_button or user_input) and user_input.strip():
            timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            
            # Generate AI response
            with st.spinner("🤔 AI is thinking..."):
                # Include chat history for context
                context = ""
                if st.session_state.chat_history:
                    context = "Previous conversation context:\n"
                    for chat_item in st.session_state.chat_history[-3:]:  # Last 3 exchanges
                        if len(chat_item) >= 3:
                            user_msg, bot_msg = chat_item[0], chat_item[1]
                            context += f"User: {user_msg}\nAI: {bot_msg}\n\n"
                
                full_prompt = f"{context}Current message: {user_input}"
                
                # Generate response with or without image
                ai_response = generate_response(
                    client, 
                    full_prompt, 
                    image=st.session_state.uploaded_image
                )
            
            # Add to chat history
            if st.session_state.uploaded_image:
                st.session_state.chat_history.append((
                    user_input, 
                    ai_response, 
                    timestamp, 
                    st.session_state.image_preview
                ))
                # Clear the uploaded image after sending
                st.session_state.uploaded_image = None
                st.session_state.image_preview = None
            else:
                st.session_state.chat_history.append((user_input, ai_response, timestamp))
            
            # Clear input and rerun
            st.rerun()
        
        # Clear chat button (at the top)
        if st.button("🗑️ Clear Chat History", key="clear_chat"):
            st.session_state.chat_history = []
            st.session_state.uploaded_image = None
            st.session_state.image_preview = None
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
