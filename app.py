from flask import Flask, render_template, request
import pickle
import csv

app = Flask(__name__)

# Load model
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/', methods=['GET', 'POST'])
def home():
    result = None
    error = None
    hours = ''
    attendance = ''
    internal = ''

    if request.method == 'POST':
        hours = request.form['hours']
        attendance = request.form['attendance']
        internal = request.form['internal']

        try:
            h = float(hours)
            a = float(attendance)
            i = float(internal)
            prediction = model.predict([[h, a, i]])
            if prediction[0] == 1:
                result = "✅ Pass 🎉"
            else:
                result = "❌ Fail 😟"

            # Save to CSV
            with open('predictions.csv', 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([hours, attendance, internal, result])

        except ValueError:
            error = "⚠️ Please enter only numbers!"

    return render_template('index.html', result=result, error=error, 
                           hours=hours, attendance=attendance, internal=internal)

@app.route('/history')
def history():
    rows = []
    try:
        with open('predictions.csv', newline='') as f:
            reader = csv.reader(f)
            rows = list(reader)
    except FileNotFoundError:
        pass
    return render_template('history.html', rows=rows)

if __name__ == '__main__':
    app.run(debug=True)
