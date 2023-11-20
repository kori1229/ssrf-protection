from flask import Flask, request, render_template, redirect
import url_features
import predicty_lstm
import keras, joblib
import numpy as np

app = Flask(__name__)

loaded_model = keras.models.load_model('trained_model.h5')
scaler = joblib.load('lstm_scaler.joblib')


def make_prediction(features):
    
    feat = np.array(features)
    featy = feat.reshape(1,-1)
    normalized_input = scaler.transform(featy)
    values = np.array(normalized_input)
    my_array_reshaped = np.expand_dims(values, axis=1)

    # Print the reshaped array
    print(my_array_reshaped.shape)


    predictions = loaded_model.predict(my_array_reshaped)
    print(f'PREDICTION FOR LSTM IS {predictions}')
    y_pred_binary = (predictions > 0.5).astype(int)
    print(f' PREDICTION ISSSSSSSSSSSSSSSSS {y_pred_binary}')
    result = 'malicious' if y_pred_binary == 1 else 'benign'
    return result
    


if __name__ == '__main__':
    app.run(debug=True)


