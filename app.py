from flask import Flask, jsonify, render_template, request
from corrector import SpellCorrector, evaluate

app = Flask(__name__)
corrector = SpellCorrector()


@app.get("/")
def index():
    return render_template("index.html")


@app.post("/api/correct")
def correct():
    text = (request.get_json(silent=True) or {}).get("text", "")
    if not text.strip():
        return jsonify(error="Enter some text to check."), 400
    if len(text) > 20000:
        return jsonify(error="Text is too long. Limit is 20,000 characters."), 413
    return jsonify(corrector.correct(text))


@app.get("/api/suggest")
def suggest():
    word = request.args.get("word", "").strip()
    return jsonify(word=word, suggestions=corrector.suggest(word) if word else [])


@app.get("/api/eval")
def run_eval():
    acc, misses = evaluate(corrector)
    return jsonify(accuracy=acc, misses=misses)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
