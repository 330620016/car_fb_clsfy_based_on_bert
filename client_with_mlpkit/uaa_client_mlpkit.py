import os
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
# from tensorflow_serving.apis import prediction_service_pb2
import json
import requests
from functools import wraps
from flask import Flask, globals, Response, request, g, jsonify
import logging
import sys

from bert import run_classifier, tokenization, optimization
import numpy as np

from sap import xssec
from cfenv import AppEnv

import json
from mlpkitsecurity import SecurityError
from mlpkit.tfs_model_utils import ModelQuerier


app = Flask(__name__)
LOG = logging.getLogger(__name__)
LOG.setLevel(logging.INFO)
LOG.addHandler(logging.StreamHandler(stream=sys.stdout))

env = AppEnv()
uaaCredentials = env.get_service(label='xsuaa').credentials

# parameters for model
LABELS_LIST = [
    'control', 'interior', 'power', 'appearance', 'safety', 'energy', 'space'
]
MAX_SEQ_LENGTH = 128
VOCAB_FILE_PATH = './vocab.txt'


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


def authenticated(func):
    """ JWT token check decorator """
    @wraps(func)
    def decorated(*args, **kwargs):
        if not checktoken():
            return Response('Unauthorized', 401)

        return func(*args, **kwargs)

    return decorated


@app.errorhandler(SecurityError)
def handle_error(error):
    # handle `SecurityError`
    response = jsonify({'message': error.detail_info})
    response.status_code = error.code
    return response


def get_access_token():
    url = "https://inno-demo.authentication.sap.hana.ondemand.com/oauth/token"
    querystring = {"grant_type": "client_credentials"}
    headers = {
        'Authorization': "Basic c2ItYWZiN2RmMTctZjI3NS00MzZkLThiMzgtYzNkN2NjNmYyYjA3IWI1MjgzfG1sLWZvdW5kYXRpb24teHN1YWEtc3RkIWIzMTM6b1ZiTmdRS3FISE1zNGVvZWhxZzhEWFJxUmxvPQ=="
    }
    response = requests.request(
        "POST", url, headers=headers, params=querystring)
    if response.status_code == 200:
        return 'Bearer ' + json.loads(response.text)['access_token']


# def metadata_transformer(metadata):
#     additions = []
#     token = get_access_token()
#     additions.append(('authorization', token))
#     return tuple(metadata) + tuple(additions)


@app.route('/classify', methods=['POST', 'GET'])
@authenticated
def main():
    foundation_service_access_token = get_access_token()
    model_name, namespace, model_version = str(os.getenv("MODEL_NAME")), str(
        os.getenv("NAMESPACE")), str(os.getenv("MODEL_VERSION"))
    MQ = ModelQuerier(model_name=model_name,
                      namespace=namespace, model_version=model_version)

    # # request ml foundation to load model
    # credentials = implementations.ssl_channel_credentials(
    #     root_certificates=ROOT_CERT)
    # channel = implementations.secure_channel(
    #     MODEL_SERVER_HOST, MODEL_SERVER_PORT, credentials)
    # stub = prediction_service_pb2.beta_create_PredictionService_stub(
    #     channel, metadata_transformer=metadata_transformer)

    # get the sentence of input
    sentence = str(globals.request.headers.getlist('Text')[0])

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
    # compose gRPC request
    request = predict_pb2.PredictRequest()
    request.model_spec.name = model_name
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

    # # do predict
    result = MQ.predict(
        request=request, access_token=foundation_service_access_token)
    # result = stub.Predict(request, 120.0)  # 10 secs timeout

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
    output_json = {"predictions": [{"results": result_list}]}
    return Response(json.dumps(output_json), mimetype='application/json')
    # output_json = {"predictions": [{"results": "123"}]}
    # return Response(json.dumps(output_json), mimetype='application/json')


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=8080, threaded=True, debug=True)
