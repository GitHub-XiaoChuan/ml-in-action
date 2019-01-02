import requests
import tensorflow as tf
import basic.mnist_input_data as mnist_input_data

headers = {"content-type": "application/json"}
# json_response = requests.post('http://localhost:8501/v1/models/half_plus_two:predict',
#                               data='{"instances": [1.0, 2.0, 5.0]}',
#                               headers=headers)
# print(json_response.text)

url = 'http://localhost:8501/v1/models/mnist:predict'
# json_response = requests.post(url,
#                               data='{"instances": [1.0, 2.0, 5.0]}',
#                               headers=headers)
# print(json_response.text)
test_data_set = mnist_input_data.read_data_sets('./tmp').test
print(type(test_data_set))

for _ in range(10):
    image, label = test_data_set.next_batch(1)
    print(tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))

    json_response = requests.post(url,
                              data='{"inputs": [image[0]]}',
                              headers=headers)
    print(json_response.text)

# channel = grpc.insecure_channel(hostport)
# stub = prediction_service_pb2_grpc.PredictionServiceStub(channel)
# result_counter = _ResultCounter(num_tests, concurrency)
# for _ in range(num_tests):
#     request = predict_pb2.PredictRequest()
#     request.model_spec.name = 'mnist'
#     request.model_spec.signature_name = 'predict_images'
#     image, label = test_data_set.next_batch(1)
#     request.inputs['images'].CopyFrom(tf.contrib.util.make_tensor_proto(image[0], shape=[1, image[0].size]))
#     result_counter.throttle()
#     result_future = stub.Predict.future(request, 5.0)  # 5 seconds
#     result_future.add_done_callback(_create_rpc_callback(label[0], result_counter))
# return result_counter.get_error_rate()
