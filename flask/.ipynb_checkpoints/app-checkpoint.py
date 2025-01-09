
from flask import Flask, request, jsonify
import requests
import pandas as pd
import logging

logging.basicConfig(level=logging.DEBUG)
app = Flask(__name__)

# Rasa's REST webhook URL (adjust if needed)
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.route('/process_message', methods=['POST'])
def process_message():
    user_message = request.json.get('message', '')
    #logging.debug(f"Received message: {user_message}")
    if not user_message:
        return jsonify({"status": "error", "message": "No message provided"}), 400
        #logging.error("No message provided")
    # Send the user message to Rasa via its REST webhook
    try:
        # Send the message to Rasa's webhook
        response = requests.post(RASA_URL, json ={"message": user_message})
        
        if response.status_code == 200:
            # Return the response from Rasa to Streamlit
            rasa_response = response.json()
            logging.debug(f"Response from Rasa: {rasa_response}")
           
            return jsonify({"status": "success", "response": rasa_response})
        else:
            #logging.error(f"Failed to connect to Rasa: {response.status_code}")
            return jsonify({"status": "error", "message": "Failed to connect to Rasa"}), 500

    except requests.exceptions.RequestException as e:
        #logging.error(f"Request exception: {e}")
        return jsonify({"status": "error", "message": str(e)}), 500
         

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Flask running on localhost:5000



# from flask import Flask, jsonify, request
# import requests

# app = Flask(__name__)

# # Replace with your Rasa endpoint
# RASA_URL = "http://localhost:5005/webhooks/rest/webhook"  # Example Rasa URL

# @app.route('/get_sales_metrics', methods=['POST'])
# def get_sales_metrics():
#     # Get the user message from the incoming request
#     user_message = request.json.get("message", "")
    
#     # Send the message to Rasa and get the response
#     rasa_response = requests.post(RASA_URL, json={"message": user_message})
    
#     # Assuming Rasa's response returns JSON
#     return jsonify(rasa_response.json())

# if __name__ == '__main__':
#     app.run(debug=True, port=5000)

