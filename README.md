#Running Rasa with Flask and Streamlit

This guide provides step-by-step instructions to run a chatbot application using Rasa, Flask, and Streamlit. The setup integrates Rasa for Natural Language Processing (NLP), Flask as the API backend, and Streamlit for the user interface.

##Prerequisites

Ensure the following tools are installed on your system:

Python (version 3.8 or higher)

pip (Python package manager)

Rasa (see Rasa installation guide)

Flask (install via pip)

Streamlit (install via pip)

Directory Structure

Create a directory with the following structure:

project-directory/
├── app.py                # Flask backend
├── chatbot_interface.py  # Streamlit frontend
├── rasa_project/         # Rasa project files
└── README.md            # This file

Steps to Run the Application

1. Set Up Rasa

Navigate to the rasa_project directory:

cd rasa_project

Initialize a Rasa project if not already done:

rasa init

Follow the prompts to create a basic Rasa chatbot.

Train the Rasa model:

rasa train

Run the Rasa server with the REST channel enabled:

rasa run --enable-api

The Rasa server will start on http://localhost:5005.

2. Run the Flask Backend

Navigate to the project root directory:

cd project-directory

Install Flask:

pip install flask

Run the Flask app:

python app.py

The Flask backend will start on http://localhost:5000.

3. Run the Streamlit Frontend

Install Streamlit:

pip install streamlit

Run the Streamlit app:

streamlit run chatbot_interface.py

Open the URL displayed in the terminal (e.g., http://localhost:8501) in your web browser.

4. Interact with the Chatbot

Use the Streamlit interface to send messages to the chatbot.

The Flask backend will forward user messages to the Rasa server.

Responses from Rasa will be displayed in the Streamlit interface.

Configuration Files

Flask (app.py)

This file handles API requests from Streamlit and forwards them to the Rasa server.

Streamlit (chatbot_interface.py)

This file provides a user-friendly interface for interacting with the chatbot.

Rasa Project (rasa_project)

Contains all Rasa-related files, such as:

domain.yml

data/nlu.yml

data/stories.yml

Troubleshooting

Rasa not responding: Ensure the Rasa server is running on http://localhost:5005.

Flask API errors: Check that Flask is running on http://localhost:5000.

Streamlit interface not loading: Verify the Streamlit app is running and accessible on http://localhost:8501.

Missing dependencies: Use pip install -r requirements.txt to install all dependencies (create a requirements.txt with necessary packages).

Notes

Customize the Rasa model and Streamlit interface as needed.

Ensure all servers are running simultaneously for the application to function correctly.

License

This project is licensed under the MIT License. See the LICENSE file for details.
