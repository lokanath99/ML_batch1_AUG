from flask import Flask, render_template, request
import joblib


app = Flask(__name__)

@app.route("/", methods=['GET'])
def prediction():
    if request.method == 'GET':
        knn = joblib.load('./Ml_model/iris_knn')
        if request.args:
            petal_width =   float(request.args.get("petal_width"))
            sepal_width = float(request.args.get("sepal_width"))
            sepal_length = float(request.args.get("sepal_lenth"))
            petal_length = float(request.args.get("petal_lenth"))
            predict = knn.predict([[sepal_length,sepal_width,petal_length,petal_width]])
            print(predict)           
            return render_template("index.html", prediction=predict[0])
        else:
            return render_template("index.html")
    