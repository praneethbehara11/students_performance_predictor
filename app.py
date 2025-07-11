import csv
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    if request.method == 'POST':
        hours = float(request.form['hours'])
        attendance = float(request.form['attendance'])
        internal = float(request.form['internal'])
        prediction = model.predict([[hours, attendance, internal]])
        if prediction[0] == 1:
            result = "✅ Pass 🎉"
        else:
            result = "❌ Fail 😟"
    return render_template('index.html', result=result)

if __name__ == '__main__':
    app.run(debug=True)
