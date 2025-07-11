from flask import Flask, render_template, request
import pickle
import csv

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

        # Save to CSV
        with open('predictions.csv', 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow([hours, attendance, internal, result])

    return render_template('index.html', result=result, hours=hours, attendance=attendance, internal=internal)

if __name__ == '__main__':
    app.run(debug=True)
