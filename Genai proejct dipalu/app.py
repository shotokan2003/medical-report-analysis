import streamlit as st
import os
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pytesseract
import pdf2image
import PyPDF2
import io

# Load environment variables
load_dotenv()

# Initialize Groq client
client = Groq(
    api_key='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'
)

# Initialize session state for chat history
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []

def extract_text_from_pdf(pdf_file):
    pdf_reader = PyPDF2.PdfReader(pdf_file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def extract_text_from_image(image_file):
    image = Image.open(image_file)
    text = pytesseract.image_to_string(image)
    return text

def analyze_medical_report(text):
    prompt = f"""You are a medical report assistant. Please analyze the following medical report and provide:
    1. Key findings
    2. Potential concerns
    3. Recommended follow-up actions
    4. Simplified explanation for the patient

    Medical Report:
    {text}
    """
    
    response = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": prompt,
            }
        ],
        model="llama-3.3-70b-versatile",  # Using Llama model through Groq
        temperature=0.3,
        max_tokens=2048,
    )
    
    return response.choices[0].message.content

def format_chat_history(chat_history):
    formatted_history = ""
    for message in chat_history:
        formatted_history += f"{message['role'].title()}: {message['content']}\n"
    return formatted_history

def get_chatbot_response(user_input, context="", chat_history=[]):
    # Format the chat history
    formatted_history = format_chat_history(chat_history)
    
    prompt = f"""You are a helpful medical assistant. Use the following information to provide accurate and helpful responses.
    
    Medical Context (if any): {context}
    
    Previous Conversation:
    {formatted_history}
    
    Current User Question: {user_input}
    
    Please provide a response that:
    1. Is consistent with the previous conversation
    2. Takes into account any medical context provided
    3. Is accurate and helpful
    4. Uses simple, clear language
    """
    
    messages = [
        {
            "role": "system",
            "content": "You are a knowledgeable medical assistant. Provide accurate, helpful, and easy-to-understand responses."
        }
    ]
    
    # Add chat history to messages
    for msg in chat_history[-5:]:  # Include last 5 messages for context
        messages.append({
            "role": msg["role"],
            "content": msg["content"]
        })
    
    # Add current prompt
    messages.append({
        "role": "user",
        "content": prompt
    })
    
    response = client.chat.completions.create(
        messages=messages,
        model="llama-3.3-70b-versatile",
        temperature=0.3,
        max_tokens=1024,
    )
    
    return response.choices[0].message.content

def main():
    st.set_page_config(page_title="Medical Report Assistant", page_icon="🏥", layout="wide")
    
    # Initialize session states
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    if 'current_context' not in st.session_state:
        st.session_state.current_context = ""
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    page = st.sidebar.radio("Choose a feature:", ["Report Analysis", "Medical Chat", "Health Tips"])
    
    if page == "Report Analysis":
        st.title("Medical Report Assistant 🏥")
        st.write("Upload your medical report (PDF or Image) for analysis")
        
        uploaded_file = st.file_uploader("Choose a file", type=['pdf', 'png', 'jpg', 'jpeg'])
        
        if uploaded_file is not None:
            try:
                # Show loading spinner
                with st.spinner('Processing your report...'):
                    # Extract text based on file type
                    file_type = uploaded_file.type
                    if 'pdf' in file_type:
                        text = extract_text_from_pdf(uploaded_file)
                    else:
                        text = extract_text_from_image(uploaded_file)
                    
                    # Show extracted text in expander
                    with st.expander("View Extracted Text"):
                        st.text(text)
                    
                    # Analyze the text
                    analysis = analyze_medical_report(text)
                    
                    # Display results
                    st.subheader("Analysis Results")
                    st.markdown(analysis)
                    
                    # Save context for chat
                    st.session_state.current_context = text
                    
            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
    
    elif page == "Medical Chat":
        st.title("Medical Chat Assistant 💬")
        st.write("Ask any medical-related questions or upload documents for context!")
        
        # Sidebar controls for chat
        with st.sidebar:
            st.subheader("Chat Controls")
            if st.button("Clear Chat History"):
                st.session_state.chat_history = []
                st.session_state.current_context = ""
                st.rerun()
            
            # Document upload in sidebar
            st.subheader("Upload Documents")
            uploaded_file = st.file_uploader(
                "Upload medical documents for context",
                type=['pdf', 'png', 'jpg', 'jpeg'],
                key="chat_file_uploader"
            )
            
            if uploaded_file:
                try:
                    with st.spinner('Processing your document...'):
                        # Extract text based on file type
                        file_type = uploaded_file.type
                        if 'pdf' in file_type:
                            text = extract_text_from_pdf(uploaded_file)
                        else:
                            text = extract_text_from_image(uploaded_file)
                        
                        # Update context
                        st.session_state.current_context = text
                        
                        # Add system message about new document
                        doc_summary = get_chatbot_response(
                            "Summarize this medical document briefly",
                            text,
                            []
                        )
                        
                        # Add document upload notification to chat
                        st.session_state.chat_history.extend([
                            {
                                "role": "system",
                                "content": f"📄 New document uploaded and processed. Summary: {doc_summary}"
                            }
                        ])
                        
                        st.success("Document processed successfully!")
                        
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
        
        # Main chat area
        # Show current medical context if any
        if st.session_state.current_context:
            with st.expander("Current Medical Context"):
                st.text(st.session_state.current_context[:500] + "...")
                if st.button("Clear Current Context"):
                    st.session_state.current_context = ""
                    st.rerun()
        
        # Chat interface
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                # Add citation button for system messages about documents
                if message["role"] == "system" and "New document uploaded" in message["content"]:
                    if st.button("📎 View Document Context", key=f"doc_{hash(message['content'])}"):
                        st.text(st.session_state.current_context[:1000] + "...")
        
        # Chat input
        if user_input := st.chat_input("Type your question here or ask about uploaded documents..."):
            # Display user message
            with st.chat_message("user"):
                st.write(user_input)
            
            # Get and display assistant response
            with st.chat_message("assistant"):
                context = st.session_state.current_context
                response = get_chatbot_response(
                    user_input, 
                    context, 
                    st.session_state.chat_history
                )
                st.write(response)
            
            # Store the conversation
            st.session_state.chat_history.extend([
                {"role": "user", "content": user_input},
                {"role": "assistant", "content": response}
            ])
    
    else:  # Health Tips page
        st.title("Daily Health Tips 💡")
        
        # Display random health tips
        tips = get_chatbot_response("Give me 5 random health tips for today in a numbered list format.")
        st.markdown(tips)
        
        # BMI Calculator
        st.subheader("BMI Calculator")
        col1, col2 = st.columns(2)
        
        with col1:
            weight = st.number_input("Weight (kg)", min_value=0.0, max_value=500.0, value=70.0)
        with col2:
            height = st.number_input("Height (m)", min_value=0.0, max_value=3.0, value=1.70)
        
        if st.button("Calculate BMI"):
            bmi = weight / (height ** 2)
            st.write(f"Your BMI is: {bmi:.2f}")
            
            # Get BMI interpretation
            bmi_interpretation = get_chatbot_response(f"Give a brief interpretation of BMI {bmi:.2f}")
            st.write(bmi_interpretation)

if __name__ == "__main__":
    main() 