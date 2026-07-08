from flask import Flask, jsonify
from flask_cors import CORS
import numpy as np
from datetime import datetime

app = Flask(__name__)
CORS(app)

# Standard 19-channel 10-20 system layout
CHANNELS = ["Fp1", "Fp2", "F7", "F3", "Fz", "F4", "F8", "T7", "C3", "Cz",
            "C4", "T8", "P7", "P3", "Pz", "P4", "P8", "O1", "O2"]
SFREQ = 128  # samples per second (matches real EEG headset rates)


@app.route("/api/eeg/latest")
def latest():
    n = SFREQ  
    t = np.arange(n) / SFREQ

    signal = {}
    for i, ch in enumerate(CHANNELS):
        rng = np.random.default_rng()
        sig = np.zeros(n)
        
        # Add spatial variation: multiply base amplitude by a random factor (e.g., 0.1 to 1.5)
        for freq, base_amp in [(2, 8), (6, 5), (10, 6), (20, 3), (40, 2)]:
            amp = base_amp * rng.uniform(0.1, 1.5) # Creates hot/cold spots on the map
            phase = rng.uniform(0, 2 * np.pi)
            sig += amp * np.sin(2 * np.pi * freq * t + phase)
            
        pink = np.cumsum(rng.normal(0, 1, n))
        pink = (pink - pink.mean()) / pink.std() * 4
        signal[ch] = (sig + pink).round(2).tolist()

    return jsonify({
        "timestamp": datetime.now().isoformat(),
        "sampling_rate": SFREQ,
        "channels": CHANNELS,
        "signal": signal,
        "alpha_power": round(np.random.uniform(0.3, 0.6), 3),
        "beta_power": round(np.random.uniform(0.2, 0.5), 3),
        "gamma_power": round(np.random.uniform(0.1, 0.3), 3),
    })


if __name__ == "__main__":
    app.run(port=5000, debug=True)
