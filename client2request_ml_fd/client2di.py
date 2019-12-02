from flask import Flask, globals, request

app = Flask(__name__)


@app.route('/', methods=['POST', 'GET'])
def main():
    data = globals.request.form.to_dict()
    pass


if __name__ == "__main__":
    app.debug = True
    app.run(host='0.0.0.0', port=5001)
