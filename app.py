import streamlit as st
from prediction import predict_sentiment

st.title('Sentiment Analysis App')

# Input box for user to enter text
user_input = st.text_area("Masukan Text:", "")

if st.button("Prediksi Sentiment"):
    prediction, compound_score = predict_sentiment(user_input)
    st.write(f"Sentiment: {prediction}")
    st.write(f"Compound Score: {compound_score}")