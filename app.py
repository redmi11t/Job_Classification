import streamlit as st
import joblib
import torch
from transformers import DistilBertTokenizer, DistilBertModel
import re
import fitz  # PyMuPDF
import plotly.express as px
import pandas as pd

# --- CONFIGURATION ---
st.set_page_config(
    page_title="AI Resume Analyzer",
    page_icon="🤖",
    layout="centered"
)


# --- MODEL & TOKENIZER LOADING ---
@st.cache_resource
def load_model_and_tokenizer():
    """Load the saved classifier and the DistilBERT model/tokenizer."""
    try:
        model = joblib.load('resume_classifier_model.joblib')
        tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        bert_model = DistilBertModel.from_pretrained('distilbert-base-uncased')
        return model, tokenizer, bert_model
    except FileNotFoundError:
        st.error("Model file not found. Make sure 'resume_classifier_model.joblib' is in the same directory.")
        return None, None, None

classifier, tokenizer, bert_model = load_model_and_tokenizer()

# --- HELPER FUNCTIONS ---
def extract_text_from_pdf(file_bytes):
    """Extract text from an uploaded PDF file."""
    try:
        with fitz.open(stream=file_bytes, filetype="pdf") as doc:
            text = "".join(page.get_text() for page in doc)
        return text
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
        return None

def preprocess_text(text):
    """Simple text cleaning."""
    text = re.sub(r'\s+', ' ', text)  # Replace multiple spaces with a single one
    text = re.sub(r'[^a-zA-Z0-9\s,.]', '', text) # Remove non-alphanumeric chars except ,.
    return text.strip().lower()

def get_bert_embedding(text, tokenizer, bert_model):
    """Generate a BERT embedding for a given text."""
    if not text:
        return None
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = bert_model(**inputs)
    # Use the mean of the last hidden state as the sentence embedding
    embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
    return embedding.reshape(1, -1) # Reshape for single prediction

def extract_skills(text):
    """A simple function to extract potential skills using regex."""
    # List of some common skills/keywords (can be expanded)
    skills_list = [
        'python', 'java', 'c\+\+', 'sql', 'javascript', 'html', 'css', 'react', 'angular', 'vue',
        'node.js', 'django', 'flask', 'fastapi', 'aws', 'azure', 'gcp', 'docker', 'kubernetes',
        'terraform', 'ansible', 'machine learning', 'deep learning', 'tensorflow', 'pytorch',
        'scikit-learn', 'pandas', 'numpy', 'data analysis', 'data visualization', 'tableau',
        'power bi', 'agile', 'scrum', 'project management', 'git', 'jira', 'devops', 'ci/cd'
    ]
    found_skills = set()
    for skill in skills_list:
        if re.search(r'\b' + re.escape(skill) + r'\b', text, re.IGNORECASE):
            found_skills.add(skill.title())
    return list(found_skills)

# --- UI LAYOUT ---
st.title("📄 AI-Powered Resume Analyzer")
st.markdown("""
Welcome! This tool uses a sophisticated AI model to analyze resumes and suggest the most suitable job category. 
Upload a resume to see the magic happen.
""")

uploaded_file = st.file_uploader("Choose a Resume File", type=["pdf", "txt"])

if uploaded_file is not None:
    # Read and process the file
    if uploaded_file.type == "application/pdf":
        file_bytes = uploaded_file.getvalue()
        resume_text = extract_text_from_pdf(file_bytes)
    else: # txt file
        resume_text = uploaded_file.read().decode("utf-8")

    if resume_text and classifier:
        st.subheader("Resume Content:")
        st.text_area("", resume_text, height=250)
        
        # Analyze button
        if st.button("Analyze Resume", type="primary"):
            with st.spinner("🧠 AI is analyzing the resume... Please wait."):
                
                # 1. Preprocess and Embed
                cleaned_text = preprocess_text(resume_text)
                embedding = get_bert_embedding(cleaned_text, tokenizer, bert_model)

                if embedding is not None:
                    # 2. Predict Probabilities
                    probabilities = classifier.predict_proba(embedding)[0]
                    top_classes_indices = probabilities.argsort()[-3:][::-1]
                    top_classes_names = classifier.classes_[top_classes_indices]
                    top_classes_probs = probabilities[top_classes_indices]

                    # 3. Extract Skills
                    found_skills = extract_skills(cleaned_text)

                    # --- DISPLAY RESULTS ---
                    st.success("Analysis Complete!")
                    
                    col1, col2 = st.columns([2, 3])

                    with col1:
                        st.subheader("🔍 Top Skills Detected")
                        if found_skills:
                            for skill in found_skills:
                                st.markdown(f"- **{skill}**")
                        else:
                            st.write("No specific skills from our list were detected.")

                    with col2:
                        st.subheader("🏆 Top 3 Job Category Predictions")
                        
                        # Create a DataFrame for the Plotly chart
                        df_probs = pd.DataFrame({
                            'Category': top_classes_names,
                            'Confidence': top_classes_probs
                        })
                        
                        # Create the bar chart
                        fig = px.bar(
                            df_probs,
                            x='Confidence',
                            y='Category',
                            orientation='h',
                            text=df_probs['Confidence'].apply(lambda x: f'{x:.1%}'),
                            color_discrete_sequence=px.colors.sequential.Tealgrn
                        )
                        fig.update_layout(
                            xaxis_title="Confidence Score",
                            yaxis_title="Job Category",
                            showlegend=False,
                            yaxis={'categoryorder':'total ascending'}
                        )
                        st.plotly_chart(fig, use_container_width=True)

    else:
        st.error("Could not read the content of the uploaded file.")
else:
    st.info("Please upload a file to begin the analysis.")