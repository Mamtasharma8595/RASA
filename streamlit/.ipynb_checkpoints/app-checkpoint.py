
import streamlit as st
import requests
import pandas as pd

# Set the Flask URL
FLASK_URL = "http://127.0.0.1:5000/process_message"

# Streamlit app title
st.title("Chat with Rasa Bot")

# Initialize chat history and user message if they don't exist in session state
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "user_message" not in st.session_state:
    st.session_state.user_message = ""

def clean_data(data, headers):
    # Convert data to a pandas DataFrame
    df = pd.DataFrame(data, columns=headers)

    # Iterate through each column to check for and clean invalid entries
    for col in df.columns:
        # Check if the column contains lists or non-scalar values and clean them
        if df[col].apply(lambda x: isinstance(x, list)).any():  # If any value is a list
            df[col] = df[col].apply(lambda x: str(x) if isinstance(x, list) else x)

        # Replace NaN or None with a default value (optional)
        df[col] = df[col].fillna('')
    df.index = df.index + 1

    return df

def send_message(user_message):
    """Send the user message to the Flask API and process the response."""
    try:
        response = requests.post(FLASK_URL, json={"message": user_message})
        if response.status_code == 200:
            json_response = response.json()
            if json_response.get("status") == "success":
                bot_response = []
                sales_data = json_response.get("response", [])

                # Handle the 'custom' field for tables and text
                if sales_data:
                    sales_data = sales_data[0]
                    if "custom" in sales_data:
                        custom_data = sales_data["custom"]

                        # Displaying the text response
                        if "text" in custom_data:
                            bot_response.append({"text": custom_data["text"]})

                        # Displaying the tables
                        if "tables" in custom_data:
                            for table in custom_data["tables"]:
                                headers = table.get("headers", [])
                                data = table.get("data", [])
                                section = table.get("section", None)
                                if headers and data:
                                    if section: 
                                        bot_response.append({"type": "section", "content": section})
                                    df = clean_data(data, headers)
                                    bot_response.append({"type": "table", "content": df})
                                    
                return bot_response
            else:
                return [{"text": f"Error: {json_response.get('message', 'Unknown error.')}"}]
        else:
            return [{"text": f"Error connecting to Flask API. Status code: {response.status_code}"}]
    except requests.exceptions.RequestException as e:
        return [{"text": f"An error occurred while connecting to the API: {e}"}]


def handle_input():
    """Handle user input and update chat history."""
    user_message = st.session_state.user_message.strip()
    if user_message:
        st.session_state.chat_history.append(f"User: {user_message}")
        bot_responses = send_message(user_message)

        # Process the bot response and add it to the chat history
        for response in bot_responses:
            if "text" in response:
                st.session_state.chat_history.append(f"Bot: {response['text']}")
            elif response["type"] == "section":
                st.session_state.chat_history.append(f"{response['content']}")
            elif response["type"] == "table":
                st.session_state.chat_history.append({"table": response["content"]})

        
        st.session_state.user_message = ""


# Display conversation first
st.subheader("Conversation")

# Display chat history
for message in st.session_state.chat_history:
    if isinstance(message, dict) and "table" in message:
        st.dataframe(message["table"])  
    elif isinstance(message, str) and message.startswith("User:"):
        st.markdown(f"<p style='color:yellow;'>{message}</p>", unsafe_allow_html=True)
    elif isinstance(message, str) and message.startswith("Bot:"):
        st.markdown(f"<p style='color:white;'>{message}</p>", unsafe_allow_html=True)
    else:  # For section headers
        st.markdown(f"<p style='color:white;'>{message}</p>", unsafe_allow_html=True)

# Divider for layout clarity
st.markdown("---")

# Input box and button layout (Bottom Section)
st.subheader("Enter your message below")
col1, col2 = st.columns([5, 1])
with col1:
    st.text_input(
        "Type your message here:",
        value=st.session_state.user_message,
        key="user_message",
        on_change=handle_input,
        placeholder="Type your message here...",
    )
with col2:
    if st.button("➤", key="send_button"):
        handle_input()
