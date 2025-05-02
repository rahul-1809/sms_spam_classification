import streamlit as st
import joblib
import string
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Download stopwords if not already downloaded
nltk.download('stopwords')

# Initialize
ps = PorterStemmer()
stop_words = set(stopwords.words('english'))

def transform_text(text):
    text = text.lower()
    text = re.findall(r'\b\w+\b', text)  

    y = []
    for i in text:
        if i.isalnum():
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        if i not in stop_words and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

# Load models
model_dict = {
    "Random Forest": 'random_forest_model.joblib',
    "Multinomial Naive Bayes": 'multinomial_nb_model.joblib',
    "Bernoulli Naive Bayes": 'bernoulli_nb_model.joblib',
}

# Function to load the selected model
def load_model(model_name):
    try:
        model = joblib.load(model_dict[model_name])
        vectorizer = joblib.load('cv_vectorizer.joblib')
        st.success(f"{model_name} and vectorizer loaded successfully!")
        return model, vectorizer
    except Exception as e:
        st.error(f"Error loading {model_name}: {e}")
        return None, None

# Streamlit UI
st.title("📩 SMS Spam Classifier")

# Dropdown for model selection
model_name = st.selectbox("Select the Model", list(model_dict.keys()))

input_sms = st.text_area("Enter the message")

if st.button('Predict'):
    if input_sms.strip() == "":
        st.warning("Please enter a valid message.")
    else:
        # Load selected model
        model, cv = load_model(model_name)
        
        if model and cv:
            # Preprocess the input message
            transformed_sms = transform_text(input_sms)

            # Vectorize the input message
            vector_input = cv.transform([transformed_sms])

            # Predict using the selected model
            result = model.predict(vector_input)[0]

            # Output the prediction result
            if result == 1:
                st.header("🚫 Spam")
            else:
                st.header("✅ Not Spam")
