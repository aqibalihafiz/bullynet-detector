import streamlit as st
import pickle
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(page_title="BullyNet Detector", page_icon="🛡️", layout="centered")

# Download NLTK data if not present
nltk.download('stopwords')
nltk.download('wordnet')

@st.cache_resource
def load_models():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_models()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    stop_words = set(stopwords.words('english'))
    lemmatizer = WordNetLemmatizer()
    tokens = [lemmatizer.lemmatize(w) for w in text.split() if w not in stop_words]
    return ' '.join(tokens)

# UI
st.title("🛡️ BullyNet - Cyberbullying Detector")
st.markdown("Built by **Mohd Aqib Ali** | AI/ML Engineer | Dubai, UAE")
st.markdown("---")

user_input = st.text_area("Enter comment / tweet to check:", "you are so stupid lol")

if st.button("Detect"):
    if user_input.strip() == "":
        st.warning("Please enter some text")
    else:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        prob = model.predict_proba(vec).max() * 100
        
        if pred == 1:
            st.error(f"🚨 Bullying Detected - {prob:.1f}% confidence")
        else:
            st.success(f"✅ Not Bullying - {prob:.1f}% confidence")
        
        st.metric("Model Accuracy", "92% | F1 0.89")
        st.caption("Dataset: 47k+ tweets | TF-IDF + Logistic Regression | Deployed on Streamlit")

st.markdown("---")
st.markdown("**Tech:** Python, NLTK, Scikit-learn, TF-IDF | **GitHub:** github.com/aqib-ali/bullynet-detector")
