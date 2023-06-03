from flask import Flask, render_template, request
from flask_ngrok import run_with_ngrok
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import torch

app = Flask(__name__)
run_with_ngrok(app)

default_model_name = 'gpt2'
default_model = GPT2LMHeadModel.from_pretrained(default_model_name)
default_tokenizer = GPT2Tokenizer.from_pretrained(default_model_name)

fine_tuned_model_name = 'gpt2'
fine_tuned_model = GPT2LMHeadModel.from_pretrained(fine_tuned_model_name)
fine_tuned_tokenizer = GPT2Tokenizer.from_pretrained(fine_tuned_model_name)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/chat', methods=['POST'])
def chat():
    user_input = request.form['user_input']

    input_ids = default_tokenizer.encode(user_input, return_tensors='pt')
    default_output = default_model.generate(input_ids, max_length=100, num_return_sequences=1)
    default_response = default_tokenizer.decode(default_output[0], skip_special_tokens=True)

    input_ids = fine_tuned_tokenizer.encode(user_input, return_tensors='pt')
    fine_tuned_output = fine_tuned_model.generate(input_ids, max_length=100, num_return_sequences=1)
    fine_tuned_response = fine_tuned_tokenizer.decode(fine_tuned_output[0], skip_special_tokens=True)

    return render_template('index.html', user_input=user_input, default_response=default_response,
                           fine_tuned_response=fine_tuned_response)

if __name__ == '__main__':
    app.run()
