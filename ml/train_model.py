import joblib
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# Sample Data: [Fever, Cough, Headache]
X = np.array([[1,1,0], [0,0,1], [1,0,1], [0,1,0]]) 
y = ['Flu', 'Migraine', 'Flu', 'Common Cold']

model = RandomForestClassifier()
model.fit(X, y)

# Save the model
joblib.dump(model, 'model.pkl')
print("Model saved as model.pkl")