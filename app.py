
import streamlit as st
from chatbot import get_response

st.title("NLP Chatbot")

user_input = st.text_input("Ask something")

if st.button("Send"):

    response = get_response(user_input)

    st.write("Bot:", response)
