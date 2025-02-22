from flask import Flask, render_template, request
from models.classes import NLIModel  # Import the NLIModel class

# Initialize the Flask app
app = Flask(__name__)

# Initialize the NLI model
nli_model = NLIModel()

# Web routes
@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        premise = request.form["premise"]
        hypothesis = request.form["hypothesis"]
        prediction = nli_model.nli_prediction(premise, hypothesis)  # Use the NLIModel for prediction
        return render_template("index.html", premise=premise, hypothesis=hypothesis, prediction=prediction)
    return render_template("index.html")

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9999)
