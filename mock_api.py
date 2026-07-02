from flask import Flask, jsonify
from flask_cors import CORS
import random
from datetime import datetime

app = Flask(__name__)
CORS(app)

@app.route("/api/eeg/latest")
def latest():
    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "alpha_power": round(random.uniform(0.3, 0.6), 3),
        "beta_power": round(random.uniform(0.2, 0.5), 3),
        "amplitude": [round(random.uniform(-1.5, 1.5), 2) for _ in range(10)]
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
