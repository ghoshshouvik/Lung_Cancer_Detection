from flask import Flask, render_template, request
import pandas as pd
import joblib

# create a Flask app
app = Flask(__name__)

# load the trained model using joblib
model = joblib.load('../pickle/cancer_lr.pkl')


@app.route('/')
def index():
    return render_template('form.html')


@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']

    data = pd.read_csv(file)
    y_pred = model.predict(data)

    data['Result'] = y_pred

    return render_template('result.html', data=data.to_html())


if __name__ == '__main__':
    app.run(debug=True)
