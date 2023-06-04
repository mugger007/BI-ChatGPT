from flask import Flask, request, render_template
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from pyngrok import ngrok
import threading

app = Flask(__name__)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
default_model = GPT2LMHeadModel.from_pretrained('gpt2')
fine_tuned_model = GPT2LMHeadModel.from_pretrained('gpt2')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    message = request.form['message']
    
    default_response = generate_response(default_model, message)
    fine_tuned_response = generate_response(fine_tuned_model, message)
    
    response = {
        'default': default_response,
        'fine_tuned': fine_tuned_response
    }
    
    return response

def generate_response(model, message):
    input_ids = tokenizer.encode(message, return_tensors='pt')
    output = model.generate(input_ids, max_length=50, num_return_sequences=1)
    response = tokenizer.decode(output[0], skip_special_tokens=True)
    
    return response

def run_flask_app():
    app.run()

def run_ngrok():
    public_url = ngrok.connect(5000)
    print("Public URL:", public_url)

flask_thread = threading.Thread(target=run_flask_app)
ngrok_thread = threading.Thread(target=run_ngrok)

flask_thread.start()
ngrok_thread.start()