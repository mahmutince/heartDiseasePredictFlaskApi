from flask import Flask, request, jsonify
import joblib

HTTP_BAD_REQUEST = 400
app = Flask(__name__)


# Load the model
MODEL = joblib.load('heartPredict.pkl')
MODEL_LABELS = ['1', '0']

@app.route('/predict')
def predict():
    # Retrieve query parameters related to this request.
    age = request.args.get('age')
    sex = request.args.get('sex')
    cp = request.args.get('cp')
    trestbps = request.args.get('trestbps')
    chol = request.args.get('chol')
    fbs = request.args.get('fbs')
    restecg = request.args.get('restecg')
    thalach = request.args.get('thalach')
    exang = request.args.get('exang')
    oldpeak = request.args.get('oldpeak')
    slope = request.args.get('slope')
    ca = request.args.get('ca')
    thal = request.args.get('thal')
    # Our model expects a list of records
    features = [[age, sex, cp, trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal]]
    
    # Use the model to predict the class
    # label_index = MODEL.predict(features)
 
    try:
        label_index = MODEL.predict(features)
    except Exception as err:
        message = ('Failed to score the model. Exception: {}'.format(err))
        response = jsonify(status='error', error_message=message)
        response.status_code = HTTP_BAD_REQUEST
        return response

     # Retrieve the iris name that is associated with the predicted class
    label = MODEL_LABELS[label_index[0]]
   
    isRisky = False
    if label == '0':
        isRisky = False
    elif label == '1':
        isRisky = True

    # Create and send a response to the API caller
    return jsonify(status='complete', isRisky=isRisky)

    if __name__ == '__main__':
        app.run(debug=True)

