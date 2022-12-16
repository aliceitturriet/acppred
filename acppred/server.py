from flask import Flask, render_template, request
from acppred.models import Model

model = Model.load('data/models/model.pickle')
app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict ():
    sequence = request.form.get('sequence')
    prediction = model.predict(sequence)
    return render_template('index.html', sequence=sequence, prediction=prediction)

if __name__ == '__main__':
    app.run()