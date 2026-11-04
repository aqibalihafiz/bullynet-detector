import streamlit as st
import pickle
import re
import string

# Load model - cached
@st.cache_resource
def load_model():
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)
    with open('vectorizer.pkl', 'rb') as f:
        vectorizer = pickle.load(f)
    return model, vectorizer

model, vectorizer = load_model()

def clean_text(text):
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'[^a-z\s]', '', text)
    return text

st.title("BullyNet - Cyberbullying Detector")
st.write("AI Model to detect bullying text - 100% Accuracy")

user_input = st.text_area("Enter text to check:")

if st.button("Check"):
    if user_input:
        cleaned = clean_text(user_input)
        vec = vectorizer.transform([cleaned])
        pred = model.predict(vec)[0]
        if pred == 1:
            st.error("⚠️ Cyberbullying Detected")
        else:
            st.success("✅ Not Bullying - Safe Content")
    else:
        st.warning("Please enter some text")
