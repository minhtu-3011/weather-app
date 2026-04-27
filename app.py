from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle

app = Flask(__name__)
CORS(app)  


with open("model_final3.pkl", "rb") as f:
    model = pickle.load(f)

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.json["input"]
        print("INPUT:", data)

        result = model.predict([data])

        return jsonify({
            "result": result[0].tolist() if hasattr(result[0], "tolist") else result[0]
        })
    except Exception as e:
        print("ERROR:", e)
        return jsonify({"error": str(e)})

app.run(debug=True)