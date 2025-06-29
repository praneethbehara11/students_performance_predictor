import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load data
df = pd.read_csv('student_data.csv')

# Convert 'Pass'/'Fail' to 1/0
df['Final_Result'] = df['Final_Result'].map({'Pass': 1, 'Fail': 0})

# Features and target
X = df[['Hours_Studied', 'Attendance', 'Internal_Score']]
y = df['Final_Result']

# Split data (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)

# Save the trained model
with open('model.pkl', 'wb') as f:
    pickle.dump(model, f)

print("✅ Model trained and saved as model.pkl")
