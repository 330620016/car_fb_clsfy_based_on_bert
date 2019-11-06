import grpc
import requests
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc

import os
from flask import Flask, globals, request, Response
import json
from bert import run_classifier, tokenization, optimization
import numpy as np

app = Flask(__name__)


server = 'localhost:8500'
model_name = "car_fb_clsfy_cn_batch"


def get_config(language: str):
    MAX_SEQ_LENGTH = 128
    LABELS_LIST = []
    VOCAB_FILE_PATH = ""
    if language == "chinese" or language == "cn" or language == "CN":
        LABELS_LIST = ['外观', '操控', '动力', '安全辅助', '空间', '能耗', '内饰']
        VOCAB_FILE_PATH = "../bert_pretrain_model/BERT_Base_Chinese/chinese_L-12_H-768_A-12/vocab.txt"
    if language == "english" or language == "en" or language == "EN":
        LABELS_LIST = [
            'control', 'interior', 'power', 'energy', 'appearance', 'safety',
            'space'
        ]
        VOCAB_FILE_PATH = "../bert_pretrain_model/BERT_Base_Uncased/uncased_L-12_H-768_A-12/vocab.txt"
    return MAX_SEQ_LENGTH, LABELS_LIST, VOCAB_FILE_PATH


@app.route('/classify', methods=['POST', 'GET'])
def main():
    MAX_SEQ_LENGTH, LABELS_LIST, VOCAB_FILE_PATH = get_config("cn")
    # get model from the server
    channel = grpc.insecure_channel(server)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # get the sentences of input
    sentences = globals.request.form.to_dict()
    # examples = []
    # keys = []
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
    request.model_spec.signature_name = 'serving_default'
    for key, sentence in sentences.items():
        example = run_classifier.InputExample(
            guid="test-0", text_a=tokenization.convert_to_unicode(sentence), text_b=None, label=LABELS_LIST[0])
        # examples.append(example)
        # keys.append(key)

        # Construct the request to tensorflow serving
        request.inputs['examples'].CopyFrom(
            tf.contrib.util.make_tensor_proto(example, shape=[1], dtype=tf.string))
        # do predict
        result = stub.Predict(request, 10.0)  # 10 secs timeout

    # # parse the result
    # probabilities_tensor_proto = result.outputs["probabilities"]
    # probabilities = list(probabilities_tensor_proto.float_val)
    # probabilities_np = np.array(probabilities)
    # top3_index_np = probabilities_np.argsort()[-3:][::-1]
    # probabilities_top3 = probabilities_np[top3_index_np]
    # label_top3 = np.array(LABELS_LIST)[top3_index_np]
    # # shape = tf.TensorShape(probabilities_tensor_proto.tensor_shape)
    # # probabilities = np.array(probabilities_tensor_proto.float_val).reshape(
    # #     shape.as_list())
    # result_list = []
    # for index in range(3):
    #     result_list.append(
    #         {"label": label_top3[index], "score": str(probabilities_top3[index])})
    # results[key] = result_list
        print(1)
    return Response(json.dumps({1: 2}), mimetype='application/json')


port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.debug = not os.getenv('PORT')
    app.run(host='0.0.0.0', port=int(port))
