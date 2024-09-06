from flask import Flask, render_template, request
import requests
import json

# Initialize the Flask app
app = Flask(__name__)

# Define the Gemini API key
API_KEY = "SECRET-KEY"
url = f'https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash-latest:generateContent?key={API_KEY}'

# Function to generate text using the Gemini API
def generate_text(prompt):
    # Data to be sent to the API
    data = {
        "contents": [
            {
                "parts": [
                    {"text": prompt}
                ]
            }
        ]
    }

    # Request headers
    headers = {
        'Content-Type': 'application/json'
    }

    # Make the POST request to the API
    response = requests.post(url, headers=headers, data=json.dumps(data))

    # Check if the request was successful
    if response.status_code == 200:
        result = response.json()
        # Extract the generated text from the JSON response
        generated_text = result["candidates"][0]["content"]["parts"][0]["text"]
        clean_text = generated_text.replace('*', '').strip()
        return clean_text
    else:
        return f"Error: {response.status_code} - {response.text}"


# Main route for the form
@app.route('/')
def index():
    return '''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Text Generation App</title>
            <style>
                body {
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f4f9;
                }
                .container {
                    text-align: center;
                    background-color: #fff;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
                }
                textarea {
                    width: 100%;
                    height: 100px;
                    padding: 10px;
                    margin-bottom: 15px;
                    border-radius: 5px;
                    border: 1px solid #ccc;
                    resize: none;
                }
                input[type="submit"] {
                    padding: 10px 20px;
                    background-color: #28a745;
                    border: none;
                    color: white;
                    border-radius: 5px;
                    cursor: pointer;
                    font-size: 16px;
                }
                input[type="submit"]:hover {
                    background-color: #218838;
                }
                h1 {
                    color: #333;
                    margin-bottom: 20px;
                }
                label {
                    font-weight: bold;
                    display: block;
                    margin-bottom: 10px;
                    color: #555;
                }
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Text Generation Application</h1>
                <form method="POST" action="/generate">
                    <label for="prompt">Enter a prompt related to SDGs (Sustainable Development Goals):</label>
                    <textarea id="prompt" name="prompt"></textarea>
                    <input type="submit" value="Generate Text">
                </form>
            </div>
        </body>
        </html>
    '''


# Route to process the text generation
@app.route('/generate', methods=['POST'])
def generate():
    prompt = request.form['prompt']

    # Generate text using the defined function
    generated_text = generate_text(prompt)

    # Return the result in a simple HTML page
    return f'''
        <!DOCTYPE html>
        <html lang="en">
        <head>
            <meta charset="UTF-8">
            <meta name="viewport" content="width=device-width, initial-scale=1.0">
            <title>Generated Text</title>
            <style>
                body {{
                    font-family: Arial, sans-serif;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                    height: 100vh;
                    margin: 0;
                    background-color: #f4f4f9;
                }}
                .container {{
                    text-align: center;
                    background-color: #fff;
                    padding: 30px;
                    border-radius: 8px;
                    box-shadow: 0px 0px 15px rgba(0, 0, 0, 0.1);
                    width: 80%;
                    max-width: 600px;
                }}
                h1 {{
                    color: #333;
                    margin-bottom: 20px;
                }}
                p {{
                    color: #555;
                    font-size: 16px;
                }}
                a {{
                    display: inline-block;
                    margin-top: 20px;
                    padding: 10px 20px;
                    background-color: #007bff;
                    color: white;
                    text-decoration: none;
                    border-radius: 5px;
                }}
                a:hover {{
                    background-color: #0056b3;
                }}
            </style>
        </head>
        <body>
            <div class="container">
                <h1>Generated Text for the Prompt: "{prompt}"</h1>
                <p>{generated_text}</p>
                <a href="/">Go Back</a>
            </div>
        </body>
        </html>
    '''


# Run the Flask server
if __name__ == '__main__':
    app.run(debug=True)
