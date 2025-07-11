from flask import Flask, render_template, request
import pickle
import csv
from datetime import datetime

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    hours = ''
    attendance = ''
    internal = ''

    if request.method == 'POST':
        hours = request.form['hours']
        attendance = request.form['attendance']
        internal = request.form['internal']

        prediction = model.predict([[float(hours), float(attendance), float(internal)]])
        if prediction[0] == 1:
            result = "✅ Pass 🎉"
        else:
            result = "❌ Fail 😟"

        # Save to CSV with date & time
        with open('predictions.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hours, attendance, internal, result, datetime.now().strftime("%Y-%m-%d %H:%M:%S")])

    return render_template('index.html', result=result, hours=hours, attendance=attendance, internal=internal)

@app.route('/history')
def history():
    history = []
    try:
        with open('predictions.csv', 'r') as f:
            reader = csv.reader(f)
            history = list(reader)
    except FileNotFoundError:
        pass
    return render_template('history.html', history=history)

if __name__ == '__main__':
    app.run(debug=True)
