from flask import Flask, request
import joblib
import numpy

MODEL_PATH = 'mlmodels/model.pkl'
SCALER_X_PATH = 'mlmodels/scaler_x.pkl'
SCALER_Y_PATH = 'mlmodels/scaler_y.pkl'

model = joblib.load(MODEL_PATH)
sc_x = joblib.load(SCALER_X_PATH)
sc_y = joblib.load(SCALER_Y_PATH)

app = Flask(__name__)

@app.route("/predict_price", methods = ['GET'])
def predict():
    args = request.args

    open_plan = args.get('open_plan', default=-1, type = int)
    area = args.get('area', default=-1, type = float)
    rooms = args.get('rooms', default=-1, type = int)
    renovation = args.get('renovation', default=-1, type=int)
    # response = 'open_plan:{}, area:{}, rooms:{}, renovation:{}'.format(open_plan, area, rooms, renovation)

    x = numpy.array([open_plan, area, rooms, renovation]).reshape(1, -1)
    x = sc_x.transform(x)
    result = model.predict(x)
    result = sc_y.inverse_transform(result.reshape(1, -1))

    return str(result[0][0])

if __name__ == '__main__':
    app.run(debug=True, port=1235, host='0.0.0.0')