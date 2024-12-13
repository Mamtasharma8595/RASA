
from flask import Flask, request, jsonify
import requests


app = Flask(__name__)

# Rasa's REST webhook URL (adjust if needed)
RASA_URL = "http://localhost:5005/webhooks/rest/webhook"

@app.route('/process_message', methods=['POST'])
def process_message():
    user_message = request.json.get('message', '')
    if not user_message:
        return jsonify({"status": "error", "message": "No message provided"}), 400
        
    # Send the user message to Rasa via its REST webhook
    try:
        # Send the message to Rasa's webhook
        response = requests.post(RASA_URL, json ={"message": user_message})

        if response.status_code == 200:
            # Return the response from Rasa to Streamlit
            rasa_response = response.json()
           
            return jsonify({"status": "success", "response": rasa_response})
        else:
            
            return jsonify({"status": "error", "message": "Failed to connect to Rasa"}), 500

    except requests.exceptions.RequestException as e:
        return jsonify({"status": "error", "message": str(e)}), 500
         

if __name__ == "__main__":
    app.run(debug=True, port=5000)  # Flask running on localhost:5000



