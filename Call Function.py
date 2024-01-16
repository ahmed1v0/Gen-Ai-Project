from flask import Flask, render_template, request
from Predicted_Answer_function import Predicted_Answer

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/Answer', methods=['POST'])
def answer():
    question = request.form['question']
    # Process the question (you can add your logic here)
    # Read the answer from a text file
    answer = Predicted_Answer(question)
    
    return render_template('main.html', Answer=answer)

if __name__ == '__main__':
    app.run(debug=True) 
