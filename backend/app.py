import os
from flask import Flask, render_template, request
import joblib

# Hum Flask ko bata rahe hain ki HTML 'frontend' folder mein hai
app = Flask(__name__, 
            template_folder='../frontend', 
            static_folder='../frontend')

# Model loading logic
# Path check karna: agar app.py 'backend' folder ke andar hai
model_path = os.path.join(os.path.dirname(__file__), 'model.pkl')
model = joblib.load(model_path)

symptoms_list = ['Fever', 'Cough', 'Headache']

@app.route('/')
def home():
    return render_template('index.html', symptoms=symptoms_list)

@app.route('/predict', methods=['POST'])
def predict():
    selected = request.form.getlist('symptoms_chosen')
    input_vector = [1 if s in selected else 0 for s in symptoms_list]
    prediction = model.predict([input_vector])[0]
    return render_template('index.html', symptoms=symptoms_list, result=prediction)

if __name__ == '__main__':
    app.run(debug=True)