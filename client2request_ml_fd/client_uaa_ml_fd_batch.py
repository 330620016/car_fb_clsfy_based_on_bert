import os
# from grpc.beta import implementations
import grpc
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2_grpc
import json
import requests
import flask
from flask import Flask, globals, Response, request, g

from bert import run_classifier, tokenization, optimization
import numpy as np

from sapjwt import jwtValidation
from sap import xssec
from functools import wraps
from cfenv import AppEnv

app = Flask(__name__)
env = AppEnv()
uaaCredentials = env.get_service(label='xsuaa').credentials

MAX_SEQ_LENGTH = 128

LABELS_LIST = str(os.getenv('LABELS', '')).replace(" ", "").split(",")
VOCAB_FILE_PATH = str(os.getenv('VOCAB_FILE_PATH', ''))
MODEL_NAME = str(os.getenv('MODEL_NAME', ''))
MODEL_SERVER_HOST = str(os.getenv('MODEL_SERVER_HOST', ''))
MODEL_SERVER_PORT = int(os.getenv('MODEL_SERVER_PORT', ''))
ROOT_CERT = str(os.getenv('ROOT_CERT', '')).replace('\\n', '\n')


@app.before_request
def before_request():
    g._uaaCredentials = uaaCredentials


def validatejwt(encodedJwtToken):
    """JWT offline validation"""
    xs_security = getattr(g, '_sap_xssec', None)
    if xs_security is None:
        xs_security = xssec.create_security_context(
            encodedJwtToken, g._uaaCredentials)
    if xs_security.get_grant_type is None:
        return False
    g._sap_xssec = xs_security
    return True


def checktoken():
    """check JWT"""
    authHeader = request.headers.get('Authorization')
    if authHeader is None:
        return False
    encodedJwtToken = authHeader.replace('Bearer ', '').strip()
    if encodedJwtToken == '':
        return False
    return validatejwt(encodedJwtToken)


def sendauth():
    """Sends a 403 response"""
    return Response('Unauthorized', 401)


def authenticated(func):
    """ JWT token check decorator """

    @wraps(func)
    def decorated(*args, **kwargs):
        if not checktoken():
            return sendauth()
        return func(*args, **kwargs)
    return decorated


def get_access_token():
    url = "https://inno-demo.authentication.sap.hana.ondemand.com/oauth/token"
    querystring = {"grant_type": "client_credentials"}
    headers = {
        'Authorization': "Basic c2ItZDZhYTZjY2QtY2QyYy00ZGFiLTk1NDYtZTMwNzkxYjQ0MGQyIWI1MjgzfG1sLWZvdW5kYXRpb24teHN1YWEtc3RkIWIzMTM6M00ybnRYdjZyKzRLbTlKWXkwRWxZQnBtcWdZPQ=="
    }
    response = requests.request(
        "POST", url, headers=headers, params=querystring)
    if response.status_code == 200:
        return 'Bearer ' + json.loads(response.text)['access_token']


def metadata_transformer():
    metadata = []
    token = get_access_token()
    metadata.append(('authorization', token))
    return tuple(metadata)


@app.route('/classify', methods=['POST', 'GET'])
@authenticated
def main():

    credentials = grpc.ssl_channel_credentials(
        root_certificates=ROOT_CERT.encode())
    channel = grpc.secure_channel('{}:{}'.format(
        MODEL_SERVER_HOST, MODEL_SERVER_PORT), credentials)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # get the sentences of input
    sentences = globals.request.form.to_dict()

    # convert single sentence to feature
    tokenizer = tokenization.FullTokenizer(
        vocab_file=VOCAB_FILE_PATH, do_lower_case=True)

    # Construct the request to tensorflow serving
    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = 'serving_default'

    results = {}
    for key, sentence in sentences.items():
        example = run_classifier.InputExample(
            guid="test-0", text_a=tokenization.convert_to_unicode(sentence), text_b=None, label=LABELS_LIST[0])
        feature = run_classifier.convert_single_example(
            0, example, LABELS_LIST, MAX_SEQ_LENGTH, tokenizer)

        # get the input of model
        input_ids = np.reshape([feature.input_ids], (1, MAX_SEQ_LENGTH))
        input_mask = np.reshape([feature.input_mask], (1, MAX_SEQ_LENGTH))
        segment_ids = np.reshape([feature.segment_ids], (MAX_SEQ_LENGTH))
        label_ids = [feature.label_id]

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
        result = stub.Predict(
            request, 100, metadata=metadata_transformer())  # 100 secs timeout

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
        result_list = []
        for index in range(3):
            result_list.append(
                {"label": label_top3[index], "score": str(probabilities_top3[index])})
        results[key] = result_list
    return Response(json.dumps(results), mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
