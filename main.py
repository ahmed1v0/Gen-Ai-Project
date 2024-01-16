from flask import Flask, request, jsonify
from transformers import BertTokenizer, BertForQuestionAnswering
import torch

app = Flask(__name__)

model_name = "asafaya/bert-large-arabic"
tokenizer = BertTokenizer.from_pretrained(model_name)
model = BertForQuestionAnswering.from_pretrained(model_name)

def answer_question(question, document):
    inputs = tokenizer(question, document, return_tensors="pt", padding=True, truncation=True)
    outputs = model(**inputs)
    start_logits = outputs.start_logits
    end_logits = outputs.end_logits
    start_index = torch.argmax(start_logits)
    end_index = torch.argmax(end_logits) + 1
    answer = tokenizer.convert_tokens_to_string(tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_index:end_index]))
    return answer

@app.route('/', methods=['POST'])
def get_answer():
    data = request.get_json()
    question = data['question']
    document = data['document']
    answer = answer_question(question, document)
    return jsonify({'answer': answer})

if __name__ == '__main__':
    app.run(debug=True)
