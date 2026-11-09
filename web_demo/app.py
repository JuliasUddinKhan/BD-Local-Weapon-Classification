"""BD-LW Local Weapon Recognition: web interface.

Run:  python app.py   then open http://127.0.0.1:5000
"""
import io

from flask import Flask, jsonify, render_template, request

import model as M

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024


@app.route("/")
def index():
    models = [(k, M.MODELS[k][0]) for k in M.available()]
    return render_template("index.html", models=models, classes=[(c, M.ENGLISH[c]) for c in M.CLASSES])


@app.route("/predict", methods=["POST"])
def predict():
    if not M.available():
        return jsonify(error="No model found. Put convnext_tiny_fold0.pt ... convnext_tiny_fold4.pt into the weights folder."), 400
    f = request.files.get("image")
    if not f:
        return jsonify(error="No image received."), 400
    try:
        rgb = M.read_image(io.BytesIO(f.read()))
    except Exception:
        return jsonify(error="This file is not a readable image."), 400
    name = request.form.get("model") or M.available()[0]
    if name not in M.available():
        name = M.available()[0]
    bg = "remove" if request.form.get("bg") == "remove" else "keep"
    try:
        return jsonify(M.predict(rgb, name, bg))
    except ImportError:
        return jsonify(error="Background removal needs rembg: run  pip install rembg  and restart."), 400


if __name__ == "__main__":
    print("Models found:", M.available() or "none (see README)")
    app.run(host="127.0.0.1", port=5000, debug=False)
