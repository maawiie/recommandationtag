import flask
from flask import render_template, redirect, request, jsonify
from mafonction import formulaire

app = flask.Flask(__name__)
app.config["DEBUG"] = True


@app.route('/', methods=['GET'])
def home():
    return render_template('template.html')

@app.route('/result', methods=['GET','POST'])
def result():
    result = formulaire(request)
    # Use the jsonify function from Flask to convert our list of
    # Python dictionaries to the JSON format.
    #return result
    return render_template('template.html', result=result)
	
if __name__ == "__main__":
    app.run()

