import streamlit as st
from prediction import predict_sentiment

st.title('Sentiment Analysis App')

# Input box for user to enter text
user_input = st.text_area("Masukan Teks:", "")

if st.button("Prediksi Sentiment"):
    prediction, compound_score = predict_sentiment(user_input)
    if compound_score >= 0.05:
        sentiment_text = "Positive"
    elif compound_score <= -0.05:
        sentiment_text = "Negative"
    else:
        sentiment_text = prediction
    
    st.write(f"Sentiment: {sentiment_text}")
    st.write(f"Compound Score: {compound_score}")
