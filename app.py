from flask import Flask, request, app,render_template
from flask import Response
import pickle
import numpy as np
import pandas as pd
import joblib

application = Flask(__name__)
app=application
# model = joblib.load('liver/random_forest_model.pkl')
model_kidney=pickle.load(open("Model/modelForCkd.pkl", "rb"))
model = pickle.load(open("Model/modelForPrediction.pkl", "rb"))

## Route for homepage
@app.route('/predictForCkd', methods=['POST'])
def predictckd():
    # Read the form data
    data = [float(request.form[field]) for field in [
        'age', 'bp', 'sg', 'al', 'su', 'rbc', 'pc', 'pcc', 'ba', 'bgr', 'bu', 'sc', 'sod', 'pot', 'hemo', 
        'pcv', 'wc', 'rc', 'htn', 'dm', 'cad', 'appet', 'pe', 'ane'
    ]]

    # Convert data to a 2D numpy array (because the model expects a 2D input)
    data = np.array([data])

    # Make the prediction
    prediction = model_kidney.predict(data)

    # Interpret the prediction
    result = "Chronic Kidney Disease" if prediction[0] == 0 else "No Chronic Kidney Disease"

    # Render the form again, with the prediction result
    return render_template('kidney.html', prediction=result)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/liver')
def liver():
    return render_template('liver.html')

@app.route('/kidney')
def kideny():
    return render_template('kidney.html')

## Route for Single data point prediction
@app.route('/predictdata',methods=['GET','POST'])
def predict_datapoint():
    result=""

    if request.method=='POST':

        Pregnancies=int(request.form.get("Pregnancies"))
        Glucose = float(request.form.get('Glucose'))
        BloodPressure = float(request.form.get('BloodPressure'))
        SkinThickness = float(request.form.get('SkinThickness'))
        Insulin = float(request.form.get('Insulin'))
        BMI = float(request.form.get('BMI'))
        DiabetesPedigreeFunction = float(request.form.get('DiabetesPedigreeFunction'))
        Age = float(request.form.get('Age'))

        new_data=scaler.transform([[Pregnancies,Glucose,BloodPressure,SkinThickness,Insulin,BMI,DiabetesPedigreeFunction,Age]])
        predict=model.predict(new_data)
       
        if predict[0] ==1 :
            result = 'Diabetic'
        else:
            result ='Non-Diabetic'
            
        return render_template('single_prediction.html',result=result)

    else:
        return render_template('home.html')

@app.route('/predict', methods=['POST','GET'])
def predict():
    # Get the form data from the POST request
    age = request.form['age']
    gender = request.form['gender']
    total_bilirubin = request.form['total_bilirubin']
    direct_bilirubin = request.form['direct_bilirubin']
    alkaline_phosphotase = request.form['alkaline_phosphotase']
    alamine_aminotransferase = request.form['alamine_aminotransferase']
    aspartate_aminotransferase = request.form['aspartate_aminotransferase']
    total_proteins = request.form['total_proteins']
    albumin = request.form['albumin']
    albumin_globulin_ratio = request.form['albumin_globulin_ratio']

    # Create a DataFrame with the input data
    data = pd.DataFrame([{
        'Age': int(age),
        'Gender': int(gender),
        'Total_Bilirubin': float(total_bilirubin),
        'Direct_Bilirubin': float(direct_bilirubin),
        'Alkaline_Phosphotase': float(alkaline_phosphotase),
        'Alamine_Aminotransferase': float(alamine_aminotransferase),
        'Aspartate_Aminotransferase': float(aspartate_aminotransferase),
        'Total_Protiens': float(total_proteins),
        'Albumin': float(albumin),
        'Albumin_and_Globulin_Ratio': float(albumin_globulin_ratio),
    }])

    # Make a prediction with the model
    prediction = model.predict(data)[0]  # Get the first (and only) prediction

    # Convert the prediction to a human-readable format
    prediction_text = "Liver Disease Detected" if prediction == 1 else "No Liver Disease Detected"

    # Render the result back to the form with the prediction
    return render_template('liver.html', prediction=prediction_text)


if __name__=="__main__":
    app.run(host="0.0.0.0")