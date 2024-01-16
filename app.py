from flask import Flask, render_template, request

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html')

@app.route('/Answer', methods=['POST'])
def answer():
    question = request.form['question']
    # Process the question (you can add your logic here)
    # Read the answer from a text file
    with open('content/data.txt', 'r', encoding='utf-8') as f:
        answer = f.read()
    
    return render_template('main.html', Answer=answer)

if __name__ == '__main__':
    app.run(debug=True)
