from flask import Flask, render_template, request, redirect, url_for
from Model import Recommendation_system
import pickle
import warnings
warnings.filterwarnings("ignore")

app = Flask(__name__)
recommend_products = Recommendation_system()


@app.route('/')
def home():
	return render_template('index.html')

@app.route('/predict',methods=['POST'])
def predict():
	Input = request.form.get('Username')
	prediction,flag = recommend_products.top_4_products(Input)
	if flag:
		return render_template('index.html',  OUTPUT= str(prediction))
	else:
		return render_template('index.html',  OUTPUT= prediction.to_html())

if __name__ == "__main__":
    app.run(debug=True)

