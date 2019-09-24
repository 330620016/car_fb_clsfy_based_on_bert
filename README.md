# text_classify_based_on_bert

## trian the model

pls follow the [train_based_on_sample.ipynb](train_based_on_sample.ipynb)

## export the model

for this train, we **Using SavedModel with Estimators**

``` python
estimator.export_savedmodel(export_dir_base, serving_input_receiver_fn)
```

`export_dir_base: ` the folder to savd model

`serving_input_receiver_fn:` serving input_fn

1. Prepare serving inputs

for bert model, the input is `label_ids` , `input_ids` , `input_mask` , `segment_ids` , reference the [https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators](https://www.tensorflow.org/guide/saved_model#using_savedmodel_with_estimators)

``` python
def serving_input_fn():
        label_ids = tf.placeholder(tf.int32, [None], name='label_ids')
        input_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH],name='input_ids')
        input_mask = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH],name='input_mask')
        segment_ids = tf.placeholder(tf.int32, [None, MAX_SEQ_LENGTH],name='segment_ids')
        input_fn=tf.estimator.export.build_raw_serving_input_receiver_fn({
            'label_ids':label_ids,
            'input_ids':input_ids,
            'input_mask':input_mask,
            'segment_ids': segment_ids,
        })()
        return input_fn
```

after export, get the files like these:

``` 
1538687457
    ├── saved_model.pb
    └── variables
        ├── variables.data-00000-of-00001
        └── variables.index
```

## deploy the model

1. in order to create request for serving, check the input and ouput structure of model

``` shell
$ saved_model_cli show --dir 1568787840 --all

MetaGraphDef with tag-set: 'serve' contains the following SignatureDefs:

signature_def['serving_default']:
  The given SavedModel SignatureDef contains the following input(s):
    inputs['input_ids'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 128)
        name: input_ids_1:0
    inputs['input_mask'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 128)
        name: input_mask_1:0
    inputs['label_ids'] tensor_info:
        dtype: DT_INT32
        shape: (-1)
        name: label_ids_1:0
    inputs['segment_ids'] tensor_info:
        dtype: DT_INT32
        shape: (-1, 128)
        name: segment_ids_1:0
  The given SavedModel SignatureDef contains the following output(s):
    outputs['predicted_labels'] tensor_info:
        dtype: DT_INT32
        shape: unknown_rank
        name: loss/Squeeze:0
    outputs['probabilities'] tensor_info:
        dtype: DT_FLOAT
        shape: (-1, 7)
        name: loss/Softmax:0
  Method name is: tensorflow/serving/predict
```

2. we use docker to deploy model and provide inference service based on Grpc or REST ful api
* download the docker image of tensorflow serving

 

``` shell
$ docker pull tensorflow/serving
```

* check the image

  

``` shell
$ docker images
REPOSITORY           TAG                 IMAGE ID            CREATED             SIZE
tensorflow/serving   latest              d4b221a3f345        2 months ago        220MB
tensorflow/serving   latest-devel        e88481fe4fbe        2 months ago        3.77GB
```

* create and run container (mount the folder of model in host to container)

    

``` shell
    docker run -it --name=tf_serving_with_dev -p 8500:8500 -p 8501:8501 -v /home/wiki/share_folder/text_classification_based_on_bert/output/pb:/models e88481fe4fbe
```

* start serving

    after last step, you should have login into the container, run the command in container shell to start serving

``` shell
root@f12ffc88646d:/tensorflow-serving# tensorflow_model_server --port=8500 --rest_api_port=8501 --model_name=car_classification --model_base_path=/models/
```

**Note:** 8500 port for grpc serving, 8501 port for HTTP/REST API serving

## inference by using tf serving

Pure version: [client2request_ml_fd/local_client.py](client2request_ml_fd/local_client.py)

## using sap ml-foundation to deploy model

when use sap ml-foundation, we just need to export the model(follow the **export model** step), the upload and deploy the model
for how to upload/upload the model, follow [SAP Help Portal - SAP Leonardo Machine Learning Foundation - Bring Your Own Model](https://help.sap.com/viewer/70cdad3d4f2f4af08c795a7c44081827/1908B/en-US), and there is a step by step tutorial by sap developer:

[ML Foundation documentation](https://github.wdf.sap.corp/ICN-ML/docs)

[Deploying your trained model in ML Foundation](https://github.wdf.sap.corp/ICN-ML/docs/blob/master/docs/byom/tutorials/byom-deploying-model.md)

## inference by ml-foundation serving

local client for request to ml-foundation:[client2request_ml_fd/client.py](client2request_ml_fd/client.py)

client publish to cloud foundary request to ml-foundation (add uaa authorization):[client2request_ml_fd/uaa_client.py](client2request_ml_fd/uaa_client.py)

