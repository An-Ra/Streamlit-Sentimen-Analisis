import joblib
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from googletrans import Translator

# Load the saved model and vectorizer
clf = joblib.load("model_com.sav")
vectorizer = joblib.load("vectorizer_com.sav")

# Initialize VADER sentiment analyzer
sid = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    # Translate text to English using googletrans
    translator = Translator()
    translated_text = translator.translate(text, src='auto', dest='en').text

    # Vectorize the preprocessed text
    text_vectorized = vectorizer.transform([text])

    # Make prediction using the trained model
    prediction = clf.predict(text_vectorized)[0]

    # Calculate compound score using VADER
    compound_score = sid.polarity_scores(translated_text)['compound']

    return prediction, compound_score
