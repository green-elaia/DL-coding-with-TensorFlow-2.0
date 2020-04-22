import tensorflow as tf
import tensorflow_hub as hub
import pathlib
import os



def model_load():
    global model
    mobile_net_url = "http://tfhub.dev/google/tf2-preview/mobilenet_v2/classification/2"
    model = tf.keras.Sequential()
    model.add(hub.KerasLayer(handle=mobile_net_url, input_shape=(224, 224, 3), trainable=False))
    model.summary()

    from tensorflow.keras.applications import MobileNetV2
    mobilev2 = MobileNetV2()
    tf.keras.utils.plot_model(mobilev2)


def dataset_load():
    content_data_url = '/content/sample_data'
    data_root_orig = tf.keras.utils.get_file('imagenetV2', 'https://s3-us-west-2.amazonaws.com/imagenetv2public/imagenetv2-topimages.tar.gz', cache_dir=content_data_url, extract=True)
    data_root = pathlib.Path(content_data_url + '/datasets/imagenetv2-topimages')
    print(data_root)
    for idx, item in enumerate(data_root.iterdir()):
        print(item)
        if idx == 9:
            break


if __name__ == "__main__":
    model_load()
    dataset_load()
