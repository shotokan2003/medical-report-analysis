from flask import Flask, render_template, request, jsonify, session, redirect, url_for
import os
from dotenv import load_dotenv
from groq import Groq
from PIL import Image
import pytesseract
import pdf2image
import PyPDF2
import io
import json
import time
from werkzeug.utils import secure_filename
import spacy
from sentence_transformers import SentenceTransformer
import numpy as np
import faiss
from typing import List, Dict, Tuple
from datetime import timedelta
from nlp_utils import clean_text, highlight_medical_entities
from flask_session import Session
import tempfile
import zlib

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.urandom(24)

# Configure session handling
app.config.update(
    SESSION_TYPE='filesystem',
    SESSION_FILE_DIR=os.path.join(tempfile.gettempdir(), 'flask_session'),
    SESSION_PERMANENT=True,
    PERMANENT_SESSION_LIFETIME=timedelta(hours=5),
    MAX_CONTENT_LENGTH=16 * 1024 * 1024,  # 16MB max upload
    UPLOAD_FOLDER='uploads'
)

# Create required directories
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
os.makedirs(app.config['SESSION_FILE_DIR'], exist_ok=True)

# Initialize session interface
Session(app)

# Initialize Groq client
client = Groq(
    api_key=os.getenv("GROQ_API_KEY", "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")
)

# Initialize models
nlp = spacy.load("en_core_sci_md")
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Add these near the top of app.py with other configurations
ALLOWED_EXTENSIONS = {'pdf', 'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# Conversation Memory class
class ConversationMemory:
    def __init__(self, dimension: int = 768, k: int = 5, max_history: int = 50):
        self.dimension = dimension
        self.k = k
        self.max_history = max_history
        self.index = faiss.IndexFlatL2(dimension)
        self.texts = []
        
    def add_interaction(self, text: str):
        embedding = encoder.encode([text])[0]
        self.index.add(np.array([embedding]).astype('float32'))
        self.texts.append(text)
        
        # Maintain maximum history size
        if len(self.texts) > self.max_history:
            self.texts = self.texts[-self.max_history:]
            # Rebuild index with truncated history
            embeddings = encoder.encode(self.texts)
            self.index = faiss.IndexFlatL2(self.dimension)
            self.index.add(embeddings.astype('float32'))
    
    def get_relevant_history(self, query: str, k: int = None) -> List[str]:
        if k is None:
            k = self.k
            
        if not self.texts:
            return []
        
        query_vector = encoder.encode([query])[0]
        k = min(self.k, len(self.texts))
        D, I = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        # Sort by relevance score
        results = [(D[0][i], self.texts[I[0][i]]) for i in range(len(I[0]))]
        results.sort(key=lambda x: x[0])  # Sort by distance (lower is better)
        
        return [text for _, text in results]

# Text extraction functions
def extract_text_from_file(file_path):
    """Extract text from PDF or image files"""
    file_extension = os.path.splitext(file_path)[1].lower()
    
    # Handle PDF files
    if file_extension == '.pdf':
        return extract_text_from_pdf(file_path)
    
    # Handle image files
    elif file_extension in ['.png', '.jpg', '.jpeg']:
        return extract_text_from_image(file_path)
    
    else:
        raise ValueError(f"Unsupported file type: {file_extension}")

def extract_text_from_pdf(pdf_path):
    """Extract text from PDF files using PyPDF2 and OCR if needed"""
    text = ""
    
    # Try to extract text directly first
    with open(pdf_path, 'rb') as file:
        reader = PyPDF2.PdfReader(file)
        for page_num in range(len(reader.pages)):
            page = reader.pages[page_num]
            page_text = page.extract_text()
            
            # If page has text, add it
            if page_text and page_text.strip():
                text += page_text + "\n\n"
    
    # If no text was extracted, try OCR
    if not text.strip():
        try:
            # Convert PDF to images
            images = pdf2image.convert_from_path(pdf_path)
            
            # Extract text from each image using OCR
            for img in images:
                text += pytesseract.image_to_string(img) + "\n\n"
        except Exception as e:
            print(f"OCR error: {str(e)}")
    
    return text.strip()

def extract_text_from_image(image_path):
    """Extract text from image files using OCR"""
    try:
        img = Image.open(image_path)
        text = pytesseract.image_to_string(img)
        return text.strip()
    except Exception as e:
        print(f"Image OCR error: {str(e)}")
        return ""

def highlight_entities(text, entities):
    """Create HTML with highlighted medical entities"""
    highlighted_text = text
    
    # Define entity colors
    entity_colors = {
        'DISEASE': '#ff9999',
        'CHEMICAL': '#99ff99',
        'PROCEDURE': '#9999ff',
        'ANATOMY': '#ffcc99',
        'PROBLEM': '#ff99cc',
        'TEST': '#99ffff',
        'TREATMENT': '#ffff99',
        'MEDICATION': '#cc99ff',
        'SYMPTOM': '#ffd699',
        'VITAL_SIGN': '#99ccff'
    }
    
    # Sort entities by length (longest first) to handle overlapping entities
    sorted_entities = sorted(entities, key=lambda x: len(x['text']), reverse=True)
    
    # Create HTML for each entity
    for entity in sorted_entities:
        entity_text = entity['text']
        entity_type = entity['label']
        color = entity_colors.get(entity_type, '#ddd')
        
        # Create the HTML mark with improved styling and tooltip
        html = f'<mark class="medical-entity" title="{entity_type}: {entity_text}" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; cursor: help; display: inline-block; margin: 0 1px;">{entity_text}</mark>'
        
        # Replace all occurrences of the entity text with the highlighted version
        highlighted_text = highlighted_text.replace(entity_text, html)
    
    # Preserve whitespace and line breaks
    highlighted_text = highlighted_text.replace('\n', '<br>')
    highlighted_text = f'<div style="white-space: pre-wrap;">{highlighted_text}</div>'
    
    return highlighted_text

def generate_report_analysis(text: str) -> str:
    """Generate analysis for the medical report"""
    try:
        prompt = """Analyze this medical report and provide:
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
                    "role": "system",
                    "content": "You are a medical report analyzer. Provide clear, concise analysis."
                },
                {
                    "role": "user",
                    "content": prompt.format(text=text)
                }
            ],
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content
    except Exception as e:
        return f"Error analyzing report: {str(e)}"

def split_text_into_chunks(text, chunk_size=200, overlap=50):
    """Split text into overlapping chunks for better RAG processing"""
    words = text.split()
    chunks = []
    
    for i in range(0, len(words), chunk_size - overlap):
        chunk = ' '.join(words[i:i + chunk_size])
        chunks.append(chunk)
        
        if i + chunk_size >= len(words):
            break
            
    return chunks

def compress_data(data):
    """Compress data for session storage"""
    if data:
        return zlib.compress(json.dumps(data).encode())
    return None

def decompress_data(compressed_data):
    """Decompress data from session storage"""
    if compressed_data:
        return json.loads(zlib.decompress(compressed_data).decode())
    return None

# Routes
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/report-analysis')
def report_analysis():
    return render_template('report_analysis.html')

@app.route('/medical-chat')
def medical_chat():
    if 'chat_history' not in session:
        session['chat_history'] = []
    
    # Check if there's a document in the session
    has_document = 'current_report' in session and session['current_report'].get('text')
    
    return render_template('medical_chat.html', 
                          chat_history=session['chat_history'],
                          has_document=has_document)

@app.route('/health-tips')
def health_tips():
    tips = get_chatbot_response("Give me 5 random health tips for today in a numbered list format.")
    return render_template('health_tips.html', tips=tips)

@app.route('/upload-report', methods=['POST'])
def upload_report():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
        
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
        
    if file and allowed_file(file.filename):
        try:
            # Save file temporarily
            filename = secure_filename(file.filename)
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            
            # Extract text
            text = extract_text_from_file(file_path)
            
            # Process text with NER
            highlighted_text, entities = highlight_medical_entities(text)
            
            # Generate analysis
            analysis = generate_report_analysis(text)
            
            # Compress and store in session
            session['current_report'] = compress_data({
                'text': text,
                'entities': entities,
                'analysis': analysis
            })
            
            # Clean up
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return jsonify({
                'highlighted_text': highlighted_text,
                'analysis': analysis,
                'entities': entities
            })
            
        except Exception as e:
            print(f"Error processing file: {str(e)}")  # Debug print
            return jsonify({'error': str(e)}), 500
            
    return jsonify({'error': 'Invalid file type'}), 400

@app.route('/chat', methods=['POST'])
def chat():
    data = request.json
    user_input = data.get('message', '')
    
    if not user_input:
        return jsonify({'error': 'No message provided'}), 400
    
    try:
        # Get current report context
        current_report = decompress_data(session.get('current_report'))
        context = current_report.get('text', '') if current_report else ''
        entities = current_report.get('entities', {}) if current_report else {}
        
        # Get chat history
        if 'chat_history' not in session:
            session['chat_history'] = []
        
        # Add user message to history
        session['chat_history'].append({
            'role': 'user',
            'content': user_input
        })
        
        # Get chatbot response with context
        response = get_chatbot_response(
            user_input=user_input,
            context=context,
            chat_history=session['chat_history']
        )
        
        # Add response to history
        session['chat_history'].append({
            'role': 'assistant',
            'content': response
        })
        
        # Limit history size and compress if needed
        if len(session['chat_history']) > 20:  # Reduced from 100 to prevent session overflow
            session['chat_history'] = session['chat_history'][-20:]
        
        session.modified = True
        
        return jsonify({
            'response': response,
            'entities': entities
        })
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get-history', methods=['GET'])
def get_history():
    # Return both chat history and current document context
    return jsonify({
        'history': session.get('chat_history', []),
        'has_document': 'current_report' in session,
        'document_preview': session.get('current_report', {}).get('text', '')[:200] + '...' if 'current_report' in session else ''
    })

@app.route('/get-model-context', methods=['GET'])
def get_model_context():
    # Get the current document context if available
    doc_context = session.get('current_report', {}).get('text', '')
    
    # Get the chat history
    chat_history = session.get('chat_history', [])
    
    # Format the context that would be sent to the model
    if chat_history:
        formatted_history = "\n".join([f"{msg['role'].upper()}: {msg['content']}" for msg in chat_history[-10:]])
    else:
        formatted_history = "No chat history available"
    
    return jsonify({
        'document_context': doc_context,
        'chat_history': formatted_history
    })

@app.route('/calculate-bmi', methods=['POST'])
def calculate_bmi():
    data = request.json
    weight = float(data.get('weight', 0))
    height = float(data.get('height', 0))
    
    if weight <= 0 or height <= 0:
        return jsonify({'error': 'Invalid weight or height'}), 400
    
    bmi = weight / (height ** 2)
    interpretation = get_chatbot_response(f"Give a brief interpretation of BMI {bmi:.2f}")
    
    return jsonify({
        'bmi': round(bmi, 2),
        'interpretation': interpretation
    })

@app.route('/clear-history', methods=['POST'])
def clear_history():
    # Clear both chat history and document context
    session['chat_history'] = []
    session.pop('current_report', None)
    session.modified = True
    return jsonify({'success': True})

@app.route('/check-document', methods=['GET'])
def check_document():
    has_document = 'current_report' in session and session['current_report'].get('text')
    document_preview = ""
    
    if has_document:
        document_preview = session['current_report']['text'][:200] + "..."
    
    return jsonify({
        'has_document': has_document,
        'document_preview': document_preview
    })

def get_chatbot_response(user_input: str, context: str = "", chat_history: List = None) -> str:
    try:
        if chat_history is None:
            chat_history = []
        
        # Create a focused prompt with context
        system_prompt = """You are a medical assistant AI with expertise in analyzing medical reports and answering health-related questions. 
        Your responses should be clear, accurate, and concise. When referring to the medical document, be specific about what you find in it."""
        
        # Format recent chat history
        recent_history = chat_history[-6:]  # Last 3 exchanges
        formatted_history = "\n".join([
            f"{msg['role'].upper()}: {msg['content']}"
            for msg in recent_history
        ])
        
        # Create messages array
        messages = [
            {"role": "system", "content": system_prompt}
        ]
        
        # Add document context if available
        if context:
            messages.append({
                "role": "system",
                "content": f"MEDICAL DOCUMENT CONTENT:\n{context[:1500]}"
            })
        
        # Add chat history
        if formatted_history:
            messages.append({
                "role": "system",
                "content": f"RECENT CONVERSATION:\n{formatted_history}"
            })
        
        # Add user query
        messages.append({"role": "user", "content": user_input})
        
        # Get response from Groq
        response = client.chat.completions.create(
            messages=messages,
            model="gemma2-9b-it",
            temperature=0.3,
            max_tokens=1024,
        )
        
        return response.choices[0].message.content
        
    except Exception as e:
        return f"I apologize, but I encountered an error: {str(e)}. Please try again."

if __name__ == '__main__':
    app.run(debug=True)