from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# salary.pkl dosyasının aynı klasörde olduğuna emin olun
model = pickle.load(open("salary.pkl", "rb"))

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Form’dan gelen metinleri int’e çeviriyoruz
    Experience = int(request.form.get('Experience'))
    Exam       = int(request.form.get('Exam'))
    Interwiev  = int(request.form.get('Interwiev'))

    # model.predict tek bir satır (örnek) için yine iç içe liste bekler
    tahmin_array = model.predict([[Experience, Exam, Interwiev]])
    # tahmin_array bir NumPy dizisi (ör. array([52300.0])), bunu Python float’a çevirelim:
    sonuc = float(tahmin_array[0])

    # Şimdi saf bir float değeri elimizde var, bunu dilediğimiz gibi formatlayabiliriz
    tahmin_metni = f'Your predicted salary: ${sonuc:.2f}'

    return render_template('index.html', tahmin=tahmin_metni)

if __name__ == "__main__":
    app.run(debug=True)
