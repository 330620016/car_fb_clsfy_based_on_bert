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
from functools import wraps

app = Flask(__name__)

MODEL_NAME = "car_fb_clsfy_cn"
MODEL_SERVER_HOST = "ms-35406304-ec04-43d2-b00a-f0493c801b24.byom.internalprod.eu-central-1.aws.ml.hana.ondemand.com"
MODEL_SERVER_PORT = 443
ROOT_CERT = "-----BEGIN CERTIFICATE-----\nMIIEizCCA3OgAwIBAgIQDI7gyQ1qiRWIBAYe4kH5rzANBgkqhkiG9w0BAQsFADBh\nMQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3\nd3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH\nMjAeFw0xMzA4MDExMjAwMDBaFw0yODA4MDExMjAwMDBaMEQxCzAJBgNVBAYTAlVT\nMRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxHjAcBgNVBAMTFURpZ2lDZXJ0IEdsb2Jh\nbCBDQSBHMjCCASIwDQYJKoZIhvcNAQEBBQADggEPADCCAQoCggEBANNIfL7zBYZd\nW9UvhU5L4IatFaxhz1uvPmoKR/uadpFgC4przc/cV35gmAvkVNlW7SHMArZagV+X\nau4CLyMnuG3UsOcGAngLH1ypmTb+u6wbBfpXzYEQQGfWMItYNdSWYb7QjHqXnxr5\nIuYUL6nG6AEfq/gmD6yOTSwyOR2Bm40cZbIc22GoiS9g5+vCShjEbyrpEJIJ7RfR\nACvmfe8EiRROM6GyD5eHn7OgzS+8LOy4g2gxPR/VSpAQGQuBldYpdlH5NnbQtwl6\nOErXb4y/E3w57bqukPyV93t4CTZedJMeJfD/1K2uaGvG/w/VNfFVbkhJ+Pi474j4\n8V4Rd6rfArMCAwEAAaOCAVowggFWMBIGA1UdEwEB/wQIMAYBAf8CAQAwDgYDVR0P\nAQH/BAQDAgGGMDQGCCsGAQUFBwEBBCgwJjAkBggrBgEFBQcwAYYYaHR0cDovL29j\nc3AuZGlnaWNlcnQuY29tMHsGA1UdHwR0MHIwN6A1oDOGMWh0dHA6Ly9jcmw0LmRp\nZ2ljZXJ0LmNvbS9EaWdpQ2VydEdsb2JhbFJvb3RHMi5jcmwwN6A1oDOGMWh0dHA6\nLy9jcmwzLmRpZ2ljZXJ0LmNvbS9EaWdpQ2VydEdsb2JhbFJvb3RHMi5jcmwwPQYD\nVR0gBDYwNDAyBgRVHSAAMCowKAYIKwYBBQUHAgEWHGh0dHBzOi8vd3d3LmRpZ2lj\nZXJ0LmNvbS9DUFMwHQYDVR0OBBYEFCRuKy3QapJRUSVpAaqaR6aJ50AgMB8GA1Ud\nIwQYMBaAFE4iVCAYlebjbuYP+vq5Eu0GF485MA0GCSqGSIb3DQEBCwUAA4IBAQAL\nOYSR+ZfrqoGvhOlaOJL84mxZvzbIRacxAxHhBsCsMsdaVSnaT0AC9aHesO3ewPj2\ndZ12uYf+QYB6z13jAMZbAuabeGLJ3LhimnftiQjXS8X9Q9ViIyfEBFltcT8jW+rZ\n8uckJ2/0lYDblizkVIvP6hnZf1WZUXoOLRg9eFhSvGNoVwvdRLNXSmDmyHBwW4co\natc7TlJFGa8kBpJIERqLrqwYElesA8u49L3KJg6nwd3jM+/AVTANlVlOnAM2BvjA\njxSZnE0qnsHhfTuvcqdFuhOWKU4Z0BqYBvQ3lBetoxi6PrABDJXWKTUgNX31EGDk\n92hiHuwZ4STyhxGs6QiA\n-----END CERTIFICATE-----\n-----BEGIN CERTIFICATE-----\nMIIDjjCCAnagAwIBAgIQAzrx5qcRqaC7KGSxHQn65TANBgkqhkiG9w0BAQsFADBh\nMQswCQYDVQQGEwJVUzEVMBMGA1UEChMMRGlnaUNlcnQgSW5jMRkwFwYDVQQLExB3\nd3cuZGlnaWNlcnQuY29tMSAwHgYDVQQDExdEaWdpQ2VydCBHbG9iYWwgUm9vdCBH\nMjAeFw0xMzA4MDExMjAwMDBaFw0zODAxMTUxMjAwMDBaMGExCzAJBgNVBAYTAlVT\nMRUwEwYDVQQKEwxEaWdpQ2VydCBJbmMxGTAXBgNVBAsTEHd3dy5kaWdpY2VydC5j\nb20xIDAeBgNVBAMTF0RpZ2lDZXJ0IEdsb2JhbCBSb290IEcyMIIBIjANBgkqhkiG\n9w0BAQEFAAOCAQ8AMIIBCgKCAQEAuzfNNNx7a8myaJCtSnX/RrohCgiN9RlUyfuI\n2/Ou8jqJkTx65qsGGmvPrC3oXgkkRLpimn7Wo6h+4FR1IAWsULecYxpsMNzaHxmx\n1x7e/dfgy5SDN67sH0NO3Xss0r0upS/kqbitOtSZpLYl6ZtrAGCSYP9PIUkY92eQ\nq2EGnI/yuum06ZIya7XzV+hdG82MHauVBJVJ8zUtluNJbd134/tJS7SsVQepj5Wz\ntCO7TG1F8PapspUwtP1MVYwnSlcUfIKdzXOS0xZKBgyMUNGPHgm+F6HmIcr9g+UQ\nvIOlCsRnKPZzFBQ9RnbDhxSJITRNrw9FDKZJobq7nMWxM4MphQIDAQABo0IwQDAP\nBgNVHRMBAf8EBTADAQH/MA4GA1UdDwEB/wQEAwIBhjAdBgNVHQ4EFgQUTiJUIBiV\n5uNu5g/6+rkS7QYXjzkwDQYJKoZIhvcNAQELBQADggEBAGBnKJRvDkhj6zHd6mcY\n1Yl9PMWLSn/pvtsrF9+wX3N3KjITOYFnQoQj8kVnNeyIv/iPsGEMNKSuIEyExtv4\nNeF22d+mQrvHRAiGfzZ0JFrabA0UWTW98kndth/Jsw1HKj2ZL7tcu7XUIOGZX1NG\nFdtom/DzMNU+MeKNhJ7jitralj41E6Vf8PlwUHBHQRFXGU7Aj64GxJUTFy8bJZ91\n8rGOmaFvE7FBcf6IKshPECBV1/MUReXgRPTqh5Uykw7+U0b6LJ3/iyK5S9kJRaTe\npLiaWN0bfVKfjllDiIGknibVb63dDcY3fe0Dkhvld1927jyNxF1WW6LZZm6zNTfl\nMrY=\n-----END CERTIFICATE-----"


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
    credentials = grpc.ssl_channel_credentials(
        root_certificates=ROOT_CERT.encode())
    channel = grpc.secure_channel('{}:{}'.format(
        MODEL_SERVER_HOST, MODEL_SERVER_PORT), credentials)
    stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)

    # get the sentence of input
    # sentence = str(globals.request.headers.getlist('Text')[0])
    # sentence = globals.request.form.to_dict()
    sentence = "配置很不错，有很多的贴心配置，让人感到很温暖"

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
    request.model_spec.name = MODEL_NAME
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
    result = stub.Predict(
        request, 100, metadata=metadata_transformer())  # 120 secs timeout

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


if __name__ == "__main__":
    app.run(host='0.0.0.0', port=5001)
