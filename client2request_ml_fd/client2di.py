import tensorflow as tf
from bert import run_classifier, tokenization
import requests
from flask import Flask, globals, request, Response
import json

app = Flask(__name__)

# LABELS_LIST = str(os.getenv('LABELS', '')).replace(" ", "").split(",")
# DEPLOYMENT_URL_DI = str(os.getenv('DEPLOYMENT_URL_DI', ''))
# VOCAB_FILE_PATH = str(os.getenv('VOCAB_FILE_PATH', ''))

LABELS_LIST = ["control", "interior", "power",
               "appearance", "safety", "energy", "space"]
DEPLOYMENT_URL_DI = "https://vsystem.ingress.dh-qx9h8hsuh.dh-canary.shoot.live.k8s-hana.ondemand.com/app/pipeline-modeler/openapi/service/16cc40f0-9ba7-4925-a9ab-9fd8f6f2db1f/clsfy/"
VOCAB_FILE_PATH = "C:/Users/I342202/softwares/share_folder/car_fb_clsfy_based_on_bert/client2request_ml_fd/vocab_en.txt"
TOKENIZER = tokenization.FullTokenizer(
    vocab_file=VOCAB_FILE_PATH, do_lower_case=True)
MAX_SEQ_LENGTH = 128


def generateInferenceRequest(sentence: str):
    req = {}
    req['signature_name'] = "serving_default"
    req['inputs'] = {}
    example = run_classifier.InputExample(
        guid="test-0", text_a=tokenization.convert_to_unicode(sentence), text_b=None, label=LABELS_LIST[0])
    feature = run_classifier.convert_single_example(
        0, example, LABELS_LIST, MAX_SEQ_LENGTH, TOKENIZER)
    req['inputs']['input_ids'] = feature.input_ids
    req['inputs']['input_mask'] = feature.input_mask
    req['inputs']['segment_ids'] = feature.segment_ids
    req['inputs']['label_ids'] = feature.label_id
    req = json.dumps(req)
    return req


@app.route('/', methods=['POST', 'GET'])
def main():
    predict_result = {}
    data = globals.request.form.to_dict()
    for key, sentence in data.items():
        req = generateInferenceRequest(sentence=sentence)
        response = requests.post(DEPLOYMENT_URL_DI,
                                 data=req,
                                 headers={
                                     "X-Requested-With": "XMLHttpRequest",
                                     "Authorization": "Basic ZGVmYXVsdFxpMzQyMjAyOkdiY2h1ITAx"})
    return Response(response, mimetype='application/json')


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5001)
