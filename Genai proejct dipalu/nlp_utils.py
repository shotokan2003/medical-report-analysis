import spacy
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from typing import List, Dict, Tuple
import json
import re

def load_nlp_model():
    """Load the NLP model with error handling and fallback options"""
    try:
        # Try to load the scientific model
        return spacy.load("en_core_sci_md")
    except OSError:
        print("Downloading scientific model...")
        import subprocess
        subprocess.run(["pip", "install", "https://s3-us-west-2.amazonaws.com/ai2-s2-scispacy/releases/v0.5.3/en_core_sci_md-0.5.3.tar.gz"])
        try:
            return spacy.load("en_core_sci_md")
        except OSError:
            print("Falling back to default English model...")
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            return spacy.load("en_core_web_sm")

# Load the NLP model
nlp = load_nlp_model()

# Add custom pipeline components if needed
if 'ner' not in nlp.pipe_names:
    ner = nlp.add_pipe('ner')

# Initialize sentence transformer
encoder = SentenceTransformer('pritamdeka/S-PubMedBert-MS-MARCO')

# Mock medical knowledge base (replace with real API integration)
MEDICAL_KNOWLEDGE = {
    "DISEASE": {
        "hypertension": "High blood pressure that can lead to serious health problems.",
        "diabetes": "A disease that occurs when blood glucose is too high.",
    },
    "DRUG": {
        "aspirin": "A common pain reliever and blood thinner.",
        "metformin": "A medication used to treat type 2 diabetes.",
    },
    "SYMPTOM": {
        "fever": "An elevated body temperature, often indicating infection.",
        "headache": "Pain in the head or upper neck area.",
    }
}

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
    
    def get_relevant_context(self, query: str) -> List[str]:
        if not self.texts:
            return []
        
        query_vector = encoder.encode([query])[0]
        k = min(self.k, len(self.texts))
        D, I = self.index.search(np.array([query_vector]).astype('float32'), k)
        
        # Sort by relevance score
        results = [(D[0][i], self.texts[I[0][i]]) for i in range(len(I[0]))]
        results.sort(key=lambda x: x[0])  # Sort by distance (lower is better)
        
        return [text for _, text in results]

def clean_text(text: str) -> str:
    """Clean text while preserving medical terms and structure"""
    text = re.sub(r'[^\w\s\n.,;:\-\/+]', ' ', text)
    text = '\n'.join(' '.join(line.split()) for line in text.split('\n'))
    return text.strip()

# Add custom components to improve medical entity recognition
def add_custom_pipes():
    """Add custom pipeline components to improve medical entity recognition"""
    # Add entity ruler if not present
    if "entity_ruler" not in nlp.pipe_names:
        ruler = nlp.add_pipe("entity_ruler", before="ner")
        
        # Define patterns for medical entities
        patterns = [
            {"label": "DISEASE", "pattern": [{"LOWER": {"IN": ["cough", "bronchiolitis", "pneumonia"]}}]},
            {"label": "SYMPTOM", "pattern": [{"LOWER": {"IN": ["chronic", "worsening", "severe"]}}]},
            {"label": "TEST", "pattern": [{"LOWER": {"IN": ["x-ray", "ppd", "cbc"]}}]},
            {"label": "PROCEDURE", "pattern": [{"LOWER": {"IN": ["evaluation", "examination", "follow-up"]}}]},
            {"label": "MEDICATION", "pattern": [{"LOWER": {"IN": ["antibiotics", "albuterol"]}}]},
        ]
        ruler.add_patterns(patterns)

    # Add phrase matcher for multi-word terms
    from spacy.matcher import PhraseMatcher
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    
    # Add multi-word patterns
    terms = {
        "TEST": ["chest x-ray", "chest x ray", "radiological study"],
        "PROCEDURE": ["well check", "follow up", "physical examination"],
        "SYMPTOM": ["chronic cough", "respiratory disease"],
    }
    
    for label, items in terms.items():
        patterns = [nlp.make_doc(text) for text in items]
        matcher.add(label, patterns)
    
    return matcher

# Initialize custom components
phrase_matcher = add_custom_pipes()

def highlight_medical_entities(text: str) -> Tuple[str, Dict]:
    """Process text to identify and highlight medical entities using NER model"""
    # Clean the text first
    text = clean_text(text)
    
    # Process with spaCy
    doc = nlp(text)
    
    # Entity categories and their colors
    entity_colors = {
        "DISEASE": "#ff9999",    # Light red
        "CHEMICAL": "#99ff99",   # Light green
        "PROCEDURE": "#9999ff",  # Light blue
        "ANATOMY": "#ffcc99",    # Light orange
        "TEST": "#99ffff",       # Light cyan
        "TREATMENT": "#ffff99",  # Light yellow
        "MEDICATION": "#cc99ff", # Light purple
        "SYMPTOM": "#ffd699",    # Light peach
    }
    
    # Store entities and their positions
    entities_found = {}
    entity_positions = []
    
    # Process NER entities
    for ent in doc.ents:
        ent_type = map_entity_type(ent.label_)
        if ent_type in entity_colors:
            add_entity(entity_positions, entities_found, ent.start_char, ent.end_char, ent.text, ent_type)
    
    # Process phrase matches
    matches = phrase_matcher(doc)
    for match_id, start, end in matches:
        span = doc[start:end]
        ent_type = nlp.vocab.strings[match_id]
        if ent_type in entity_colors:
            add_entity(entity_positions, entities_found, span.start_char, span.end_char, span.text, ent_type)
    
    # Remove overlapping entities
    entity_positions = remove_overlapping_entities(entity_positions)
    
    # Create highlighted HTML
    highlighted_html = create_highlighted_html(text, entity_positions, entity_colors)
    
    print(f"Found entities: {entities_found}")  # Debug output
    
    return highlighted_html, entities_found

def add_entity(positions: list, found: dict, start: int, end: int, text: str, ent_type: str):
    """Add entity to positions and found collections"""
    positions.append({
        'start': start,
        'end': end,
        'text': text,
        'type': ent_type
    })
    
    if ent_type not in found:
        found[ent_type] = []
    if text not in found[ent_type]:
        found[ent_type].append(text)

def map_entity_type(label: str) -> str:
    """Map model entity labels to our categories"""
    mapping = {
        "DISEASE": "DISEASE",
        "CHEMICAL": "MEDICATION",
        "DRUG": "MEDICATION",
        "DISORDER": "DISEASE",
        "FINDING": "SYMPTOM",
        "PROBLEM": "SYMPTOM",
        "TEST": "TEST",
        "TREATMENT": "TREATMENT",
        "BODY": "ANATOMY",
        "PROCEDURE": "PROCEDURE",
        "SIGN": "SYMPTOM",
        "SYMPTOM": "SYMPTOM"
    }
    return mapping.get(label, label)

def remove_overlapping_entities(entities: List[Dict]) -> List[Dict]:
    """Remove overlapping entities, keeping the most specific ones"""
    if not entities:
        return []
    
    # Sort by start position and length (longer entities first)
    entities.sort(key=lambda x: (x['start'], -len(x['text'])))
    
    result = []
    last_end = -1
    
    for entity in entities:
        if entity['start'] >= last_end:
            result.append(entity)
            last_end = entity['end']
    
    return result

def create_highlighted_html(text: str, entities: List[Dict], colors: Dict) -> str:
    """Create HTML with highlighted entities"""
    html_parts = []
    last_pos = 0
    
    for entity in sorted(entities, key=lambda x: x['start']):
        # Add text before entity
        if entity['start'] > last_pos:
            html_parts.append(text[last_pos:entity['start']])
        
        # Add highlighted entity
        color = colors[entity['type']]
        tooltip = f"{entity['type']}: {entity['text']}"
        highlighted_text = f'<mark class="medical-entity" style="background-color: {color}; padding: 2px 4px; border-radius: 3px; cursor: help;" title="{tooltip}">{text[entity["start"]:entity["end"]]}</mark>'
        html_parts.append(highlighted_text)
        
        last_pos = entity['end']
    
    # Add remaining text
    if last_pos < len(text):
        html_parts.append(text[last_pos:])
    
    # Join all parts and add line breaks
    highlighted_html = ''.join(html_parts)
    highlighted_html = highlighted_html.replace('\n', '<br>')
    
    return highlighted_html

def get_entity_info(entity: str, entity_type: str) -> str:
    """Get explanation for medical entity (mock implementation)"""
    entity = entity.lower()
    for category, items in MEDICAL_KNOWLEDGE.items():
        if entity in items:
            return items[entity]
    return f"Medical {entity_type.lower()}" 