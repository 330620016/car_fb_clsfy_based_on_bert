import os
from grpc.beta import implementations
import tensorflow as tf
from tensorflow_serving.apis import predict_pb2
from tensorflow_serving.apis import prediction_service_pb2
import json
import requests
import flask
from flask import Flask, globals, Response, request, g

from bert import run_classifier,tokenization,optimization
import numpy as np
from config import MODEL_NAME,MODEL_SERVER_HOST,MODEL_SERVER_PORT,ROOT_CERT,LABELS_LIST,MAX_SEQ_LENGTH,VOCAB_FILE_PATH
app = Flask(__name__)

def get_access_token():
    print("******  begain to get access token  ******")
    url = "https://inno-demo.authentication.sap.hana.ondemand.com/oauth/token"
    querystring = {"grant_type":"client_credentials"}
    headers = {
        'Authorization': "Basic c2ItYWZiN2RmMTctZjI3NS00MzZkLThiMzgtYzNkN2NjNmYyYjA3IWI1MjgzfG1sLWZvdW5kYXRpb24teHN1YWEtc3RkIWIzMTM6b1ZiTmdRS3FISE1zNGVvZWhxZzhEWFJxUmxvPQ=="
        }
    response = requests.request("POST", url, headers=headers, params=querystring)
    if response.status_code == 200:
        print("******   Done   *****")
        return 'Bearer ' + json.loads(response.text)['access_token'].encode('ascii', 'ignore')
    else:
        print("******   get token wrong   ******")
        return

def metadata_transformer(metadata):
    additions = []
    token = get_access_token()
    additions.append(('authorization', token))
    return tuple(metadata) + tuple(additions)


@app.route('/', methods=['POST', 'GET'])
def main():
    # request ml foundation to load model
    credentials = implementations.ssl_channel_credentials(
        root_certificates=ROOT_CERT)
    channel = implementations.secure_channel(
        MODEL_SERVER_HOST, MODEL_SERVER_PORT, credentials)
    stub = prediction_service_pb2.beta_create_PredictionService_stub(
        channel, metadata_transformer=metadata_transformer)

    tokenizer = tokenization.FullTokenizer(vocab_file=VOCAB_FILE_PATH,do_lower_case=True)

    # get the sentence of input
    sentence = globals.request.headers.getlist('Text')
    print(sentence)

    # convert single sentence to feature
    example = run_classifier.InputExample(guid="test-0",text_a=tokenization.convert_to_unicode(sentence),text_b=None,label=LABELS_LIST[0])
    feature = run_classifier.convert_single_example(0, example, LABELS_LIST, MAX_SEQ_LENGTH, tokenizer)
    input_ids = np.reshape([feature.input_ids],(1,MAX_SEQ_LENGTH))
    input_mask = np.reshape([feature.input_mask],(1,MAX_SEQ_LENGTH))
    segment_ids =  np.reshape([feature.segment_ids],(MAX_SEQ_LENGTH))
    label_ids =[feature.label_id]


    request = predict_pb2.PredictRequest()
    request.model_spec.name = MODEL_NAME
    request.model_spec.signature_name = 'predicted_labels'
    request.inputs['input_ids'].CopyFrom(input_ids)
    request.inputs['input_mask'].CopyFrom(input_mask)
    request.inputs['segment_ids'].CopyFrom(segment_ids)
    request.inputs['label_ids'].CopyFrom(label_ids)

    grpc_result = stub.Predict(request, 20.0)
    print(grpc_result)
    return Response(json.dumps({"123": "456"}), mimetype='application/json')


port = os.getenv('PORT', '5000')
if __name__ == "__main__":
    app.debug = not os.getenv('PORT')
    app.run(host='0.0.0.0', port=int(port))
