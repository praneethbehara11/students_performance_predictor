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
        # Save to CSV
        with open('predictions.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hours, attendance, internal, result])
    return render_template('index.html', result=result)

@app.route('/history')
def history():
    data = []
    try:
        with open('predictions.csv', newline='') as f:
            reader = csv.reader(f)
            data = list(reader)
    except FileNotFoundError:
        pass  # no data yet
    return render_template('history.html', data=data)

if __name__ == '__main__':
    app.run(debug=True)
