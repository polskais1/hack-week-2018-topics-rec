from flask import Flask, jsonify, request
from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
import pickle
import re
import tensorflow as tf
import urllib.request

import pdb

app = Flask(__name__)

# loading
with open("../article_topic_model/model/tokenizer.pickle", "rb") as handle:
  tok = pickle.load(handle)

print("* Loading Keras model and Flask starting server...")
model = load_model("../article_topic_model/model/topic_model.h5")
app.run()

graph = tf.get_default_graph()

@app.route("/", methods=["GET", "POST"])
def index():
  data = {"success": False}
  if request.method == "GET":
    return "Hello"

  if request.method == "POST":
    raw_article = request.form["article"]
    article = prepare_article(raw_article)

    with graph.as_default():
      predictions = model.predict(article)[0]

    sorted_predictions = []
    for i in range(len(predictions)):
      sorted_predictions.append({
        "label": topics[i],
        "score": predictions[i]
      })

    data["predictions"] = sorted_predictions
    data["success"] = True
  return jsonify(data)

def prepare_article(article):
  sanitized_article = sanitize_article(article)

  # Tokenize, sequence, and pad the article
  sequences = Tokenizer.texts_to_sequences(tok, [sanitized_article])
  return pad_sequences(sequences, maxlen=2000)
  

def sanitize_article(article):
  return re.sub(r"<[^>]*>", "", article)
