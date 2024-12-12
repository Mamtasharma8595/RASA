

import streamlit as st
import requests
from flask import jsonify
from tabulate import tabulate
import pandas as pd
# Set the Flask URL
FLASK_URL = "http://127.0.0.1:5000/process_message"


# Streamlit app title
st.title("chat with rasa bot")
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_message" not in st.session_state:
    st.session_state.user_message = ""


def send_message(user_message):
    try:
        response = requests.post(FLASK_URL, json={"message": user_message})
        if response.status_code == 200:
            json_response = response.json()
            if json_response.get('status') == 'success':
                bot_response = []
                for item in json_response.get('response', []):
                    # if 'text' in item:
                        # if item['text'].startswith("|"):  # Markdown table
                        #     st.markdown(item['text'], unsafe_allow_html=True)
                        # else:
                        #     st.write(item['text'])
                    if 'text' in item:
                        bot_response.append(item['text'].replace("\n", "<br>"))
                return bot_response #"\n".join(bot_response)
            else:
                return f"Error: {json_response.get('message', 'Unknown error.')}"
        else:
            return f"Error connecting to Flask API. Status code: {response.status_code}"
    except requests.exceptions.RequestException as e:
        return f"An error occurred while connecting to the API: {e}"

# Function to handle user input
def handle_input():
    user_message = st.session_state.user_message.strip()
    if user_message:
        # Append the user's message to chat history
        st.session_state.chat_history.append(f"User: {user_message}")
        
        # Send the message to the Flask API and get the response
        bot_response = send_message(user_message)
        

        for message in bot_response:
            if isinstance(message, pd.DataFrame):
                # If the message is a table (DataFrame), display it using st.table
                st.table(message)
            else:
                # Otherwise, just display the message
                st.session_state.chat_history.append(f"Bot: {message}")
        # Append the bot's response to chat history
        # st.session_state.chat_history.append(f"Bot: {bot_response}")
        
        # Clear the input field for the next message
        st.session_state.user_message = ""

# Display chat history with color customization
st.subheader("Conversation")
for message in st.session_state.chat_history:
    if message.startswith("User:"):
        st.markdown(f"<p style='color:yellow;'>{message}</p>", unsafe_allow_html=True)
    elif message.startswith("Bot:"):
        # st.markdown(f"<p style='color:white;'>{message}</p>", unsafe_allow_html=True)
        # Handle formatted bot responses
        
        if "|" in message:  # Indicates tabular data
            st.markdown(f"<pre style='color:white;'>{message[5:]}</pre>", unsafe_allow_html=True)
        else:  # Plain text
            st.markdown(f"<p style='color:white;'>{message}</p>", unsafe_allow_html=True)

    # else:
    #     st.markdown(f"<p style='color:white;'>{message}</p>", unsafe_allow_html=True)

# Input and icon button layout
col1, col2 = st.columns([5, 1])
with col1:
    st.text_input(
        "Enter your message:",
        value=st.session_state.user_message,
        key="user_message",  # Correct key name
        on_change=handle_input,  # Trigger when Enter is pressed
        placeholder="Type your message here...",
    )
with col2:
    if st.button("➤", key="send_button"):
        handle_input()  # Trigger when the icon button is pressed


