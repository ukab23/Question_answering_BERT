from flask import Flask, request, jsonify
# from sklearn.externals import joblib
import traceback
import pandas as pd
import numpy as np
from qa2 import Classifier
# Your API definition
app = Flask(__name__)
result = Classifier()
@app.route('/')
def query_example():
    return 'Go to  http://127.0.0.1:5000/qa to ask question.'


@app.route('/qa', methods=['POST']) #GET requests will be blocked
def json_example():
    req_data = request.get_json()
    # print(req_data)
    paragraph = req_data['paragraph']
    question = req_data['question']
    
    ans = result.get_answer(para = paragraph, qr = question)

    return {'paragraph':paragraph, 'question':question, 'Answer':ans[0][0]}


if __name__ == '__main__':
    try:
        port = int(sys.argv[1])
    except:
        port = 5000 
    app.run(port=port, debug=True)
