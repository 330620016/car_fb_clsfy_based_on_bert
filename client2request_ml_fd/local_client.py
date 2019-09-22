from __future__ import print_function

# This is a placeholder for a Google-internal import.
import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import os
from flask import Flask, globals, request, Response
import json
from bert import run_classifier, tokenization, optimization
from config import MAX_SEQ_LENGTH, LABELS_LIST, VOCAB_FILE_PATH
import numpy as np

app = Flask(__name__)


server = 'localhost:8500'


@app.route('/', methods=['POST', 'GET'])
def main():
    # get model from the server
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # get the sentence of input
    sentence = str(globals.request.headers.getlist('Text')[0])
    print(sentence)

    # convert single sentence to feature
    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE_PATH, do_lower_case=True)
    example = run_classifier.InputExample(
        guid="test-0", text_a=tokenization.convert_to_unicode(sentence), text_b=None, label=LABELS_LIST[0])
    feature = run_classifier.convert_single_example(
        0, example, LABELS_LIST, MAX_SEQ_LENGTH, tokenizer)

    # get the input of model
    input_ids = np.reshape([feature.input_ids], (1, MAX_SEQ_LENGTH))
    input_mask = np.reshape([feature.input_mask], (1, MAX_SEQ_LENGTH))
    segment_ids = np.reshape([feature.segment_ids], (MAX_SEQ_LENGTH))
    label_ids = [feature.label_id]

    # Construct the request to tensorflow serving
    request = predict_pb2.PredictRequest()
    request.model_spec.name = 'car_classification'
    request.model_spec.signature_name = 'serving_default'

    # package the input into request, Note the format of the input(follow the model)
    request.inputs['input_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(input_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
    request.inputs['input_mask'].CopyFrom(
        tf.contrib.util.make_tensor_proto(input_mask, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))
    request.inputs['label_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(label_ids, shape=[1], dtype=tf.int32))
    request.inputs['segment_ids'].CopyFrom(
        tf.contrib.util.make_tensor_proto(segment_ids, shape=[1, MAX_SEQ_LENGTH], dtype=tf.int32))

    # do predict
    result = stub.Predict(request, 10.0)  # 10 secs timeout

    # parse the result
    probabilities_tensor_proto = result.outputs["probabilities"]
    probabilities = list(probabilities_tensor_proto.float_val)
    probabilities_np = np.array(probabilities)
    top3_index_np = probabilities_np.argsort()[-3:][::-1]
    probabilities_top3 = probabilities_np[top3_index_np]
    label_top3 = np.array(LABELS_LIST)[top3_index_np]
    # shape = tf.TensorShape(probabilities_tensor_proto.tensor_shape)
    # probabilities = np.array(probabilities_tensor_proto.float_val).reshape(
    #     shape.as_list())
    print(probabilities_top3, label_top3)
    return Response(json.dumps({"123": "456"}), mimetype='application/json')


port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.debug = not os.getenv('PORT')
    app.run(host='0.0.0.0', port=int(port))
