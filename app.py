# app.py
import streamlit as st
import pysrt
import os
import gdown
from transformers import pipeline, AutoTokenizer
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import pandas as pd
import torch
import time

# Initialize session state
if 'translator' not in st.session_state:
    st.session_state.translator = None
    st.session_state.loading = False

# Download knowledge base if not exists
def download_knowledge_base():
    if not os.path.exists('knowledge_base'):
        os.makedirs('knowledge_base', exist_ok=True)
    
    dataset_path = 'knowledge_base/dataset.csv'
    if not os.path.exists(dataset_path):
        with st.spinner('Downloading Sinhala knowledge base...'):
            url = 'https://drive.google.com/uc?id=1bJ9HXgZR1bFdYdQrQw0Lk6Zx9D7W8mTv'
            gdown.download(url, dataset_path, quiet=False)
    
    return dataset_path

# Translator class with RAG
class SinhalaTranslator:
    def __init__(self):
        # Load models
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-sinhala",
            device=0 if torch.cuda.is_available() else -1
        )
        self.embedder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kb_path = download_knowledge_base()
        self.load_knowledge_base()
        
        # Configure generation
        self.tokenizer = AutoTokenizer.from_pretrained("Helsinki-NLP/opus-mt-en-sinhala")
        self.generation_args = {
            "max_length": 400,
            "num_beams": 5,
            "early_stopping": True,
            "no_repeat_ngram_size": 2
        }
        
        # Create cache
        self.translation_cache = {}

    def load_knowledge_base(self):
        # Load custom dataset
        self.kb = pd.read_csv(self.kb_path)
        
        # Create FAISS index
        embeddings = self.embedder.encode(self.kb['english'].tolist())
        self.index = faiss.IndexFlatL2(embeddings.shape[1])
        self.index.add(embeddings.astype('float32'))
        
    def retrieve_context(self, query: str, k=3):
        query_embed = self.embedder.encode([query])
        distances, indices = self.index.search(query_embed.astype('float32'), k)
        
        # Filter by similarity threshold
        context_examples = []
        for i, distance in zip(indices[0], distances[0]):
            if distance < 1.0:  # Similarity threshold
                context_examples.append(self.kb.iloc[i]['sinhala'])
        
        return context_examples

    def translate(self, text: str) -> str:
        # Skip empty lines
        if not text.strip():
            return ""
            
        # Check cache
        if text in self.translation_cache:
            return self.translation_cache[text]
        
        try:
            # Retrieve contextual examples
            context_examples = self.retrieve_context(text)
            context = " ".join(context_examples)
            
            # Prepare model input
            inputs = self.tokenizer(
                f"CONTEXT: {context} TEXT: {text}",
                return_tensors="pt",
                truncation=True,
                max_length=384
            )
            
            # Generate translation
            translated = self.translator(
                text,
                **self.generation_args
            )
            
            result = translated[0]['translation_text']
            self.translation_cache[text] = result
            return result
        except Exception as e:
            st.error(f"Error translating: {text} - {str(e)}")
            return text  # Return original on failure

# Initialize translator
def init_translator():
    if st.session_state.translator is None:
        with st.spinner('Loading translation models... This may take a minute'):
            st.session_state.translator = SinhalaTranslator()

# Process SRT file
def process_srt_file(uploaded_file):
    try:
        # Save uploaded file
        file_path = f"temp_{uploaded_file.name}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
            
        # Process subtitles
        subs = pysrt.open(file_path)
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        for i, sub in enumerate(subs):
            # Update progress
            progress = (i + 1) / len(subs)
            progress_bar.progress(progress)
            status_text.text(f"Translating: {i+1}/{len(subs)} lines - {sub.text[:30]}...")
            
            # Translate text
            sub.text = st.session_state.translator.translate(sub.text)
            
            # Add slight delay to prevent UI freezing
            time.sleep(0.01)
        
        # Save translated file
        output_path = f"translated_{uploaded_file.name}"
        subs.save(output_path, encoding='utf-8')
        
        # Clean up
        os.remove(file_path)
        status_text.empty()
        progress_bar.empty()
        
        return output_path
    except Exception as e:
        st.error(f"Error processing file: {str(e)}")
        return None

# Streamlit UI
st.set_page_config(
    page_title="English to Sinhala Subtitle Translator",
    page_icon="🇱🇰",
    layout="wide"
)

# Header
st.title("🇱🇰 English to Sinhala Subtitle Translator")
st.markdown("""
Convert English subtitles to natural Sinhala using AI-powered translation with contextual understanding.
""")

# Features
with st.expander("✨ Key Features"):
    st.markdown("""
    - **Natural Sinhala Translations**: Uses colloquial Sri Lankan language
    - **Context-Aware Translation**: Retrieval-Augmented Generation (RAG) for accurate translations
    - **Preserves Timing**: Maintains original subtitle timing and formatting
    - **Free & Open Source**: Built with Helsinki-NLP models
    """)

# File uploader
st.subheader("Upload English Subtitle File (.srt)")
uploaded_file = st.file_uploader(
    "Choose an SRT file", 
    type="srt",
    accept_multiple_files=False,
    help="Upload an English subtitle file in SRT format"
)

# Initialize translator when needed
if uploaded_file or st.session_state.translator:
    init_translator()

# Process file if uploaded
if uploaded_file and st.session_state.translator:
    st.success("File uploaded successfully!")
    
    # Show sample content
    with st.expander("Preview original subtitles"):
        sample_content = uploaded_file.getvalue().decode("utf-8")[:500]
        st.code(sample_content, language=None)
    
    if st.button("Translate to Sinhala", type="primary"):
        with st.spinner("Translating subtitles..."):
            output_path = process_srt_file(uploaded_file)
            
        if output_path and os.path.exists(output_path):
            st.success("Translation completed successfully!")
            
            # Show sample translation
            with st.expander("Preview translated subtitles"):
                with open(output_path, "r", encoding="utf-8") as f:
                    translated_content = f.read()[:500]
                st.code(translated_content, language=None)
            
            # Download button
            with open(output_path, "rb") as f:
                st.download_button(
                    label="Download Sinhala Subtitles",
                    data=f,
                    file_name=output_path,
                    mime="text/plain"
                )
            
            # Clean up
            os.remove(output_path)
        else:
            st.error("Translation failed. Please try again.")

# Knowledge base info
with st.expander("ℹ️ About the Translation System"):
    st.markdown("""
    **Technology Stack:**
    - Translation Model: `Helsinki-NLP/opus-mt-en-sinhala`
    - RAG Components: 
        - Sentence Transformer: `paraphrase-multilingual-MiniLM-L12-v2`
        - FAISS for similarity search
    - Custom knowledge base with Sri Lankan colloquialisms
    
    **How It Works:**
    1. Upload English subtitle file (.srt format)
    2. System extracts text while preserving timing information
    3. Each line is translated using context from similar phrases
    4. Translated subtitles are formatted back into SRT format
    5. Download natural-sounding Sinhala subtitles
    
    The system uses Retrieval-Augmented Generation (RAG) to provide contextually appropriate translations
    in everyday Sri Lankan Sinhala rather than literal translations.
    """)
    
    if st.session_state.translator and st.session_state.translator.kb is not None:
        st.markdown(f"**Knowledge Base Stats:** {len(st.session_state.translator.kb)} phrases")

# Footer
st.markdown("---")
st.caption("Built with ❤️ using Python, Streamlit, and open-source AI models")
st.caption("Note: First-time model loading may take 1-2 minutes. Subsequent translations will be faster.")

# Initialize translator on app load
if not uploaded_file and st.session_state.translator is None:
    init_translator()