from flask import Flask, render_template, request
import pickle

# Flask ilovasini yaratish
app = Flask(__name__)

# Modelni yuklash
with open('chatbot_model.pkl', 'rb') as f:
    vectorizer, model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    text = request.form.get('text', '')

    if not text:
        return render_template('index.html', prediction="Matnni kiriting!")

    # Matnni vektorizatsiya qilish
    vectorized_text = vectorizer.transform([text])

    # Model orqali bashorat qilish
    #Soxta yangiliklar uchun 1-yorliq, aks holda 0
    prediction = model.predict(vectorized_text)[0]
    result = "Soxta" if prediction == 1 else "Haqiqiy"

    return render_template('index.html', prediction=result)

if __name__ == '__main__':
    app.run(debug=True)
