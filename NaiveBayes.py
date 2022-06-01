from flask import Flask, request, render_template
import pickle
import numpy as np 

app = Flask(__name__, template_folder="templates")

model_file = open('model/prediksi.pkl', 'rb')
model = pickle.load(model_file, encoding='bytes')

@app.route('/')
def index():
    # return "success"
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    '''
    Predict the insurance cost based on user inputs
    and render the result to the html page
    '''
    tanggal=float(request.form['tanggal'])
    
    jumlah=float(request.form['jumlah'])

    bahan=float(request.form['bahan'])

    permintaan=float(request.form['permintaan'])

    day=float(request.form['day'])

    X=np.array([[day,tanggal,jumlah,bahan,permintaan]])
    
    prediction = model.predict(X)

    output = round(prediction[0],0)
    if (output==0):
        kelas="Mempertahankan Stock"
    elif (output==1):
        kelas="Menambah Stock"

    return render_template('index.html',kelas=kelas, tanggal=tanggal, jumlah=jumlah, bahan=bahan, permintaan=permintaan)


if __name__ == '__main__':
    app.run(debug=True)