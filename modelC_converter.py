# Brief: this code using to convert model to model lite using tflite
# Author : Van Quan PHAM
# Created day : 2 Jan 2023
# Status :  
import tensorflow as tf
import numpy as np
import train
from test import model_name
tfmodel_path = 'trained_models/pretrainedResnet.h5'
#tfmodelLite_path = 'trained_models/pretrainedResnet.tflite'
#tfmodel = tf.keras.models.load_model(tfmodel_path)
#converter = tf.lite.TFLiteConverter.from_keras_model(tfmodel)
#tflite_model = converter.convert()
tfmodel = tf.keras.models.load_model(tfmodel_path)

cifar_10_dir = 'cifar-10-batches-py'
def representative_dataset_generator():
    train_data, train_filenames, train_labels, test_data, test_filenames, test_labels, label_names = \
        train.load_cifar_10_data(cifar_10_dir)
    _idx = np.load('calibration_samples_idxs.npy')
    for i in _idx:
        sample_img = np.expand_dims(np.array(test_data[i], dtype=np.float32), axis=0)
        yield [sample_img]


converter = tf.lite.TFLiteConverter.from_keras_model(tfmodel)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.representative_dataset = representative_dataset_generator
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8
tflite_model_int8 = converter.convert()
from tensorflow.lite.python.util import convert_bytes_to_c_source
source_text,header_text = convert_bytes_to_c_source(tflite_model_int8, "modelLite")
with open('modelLite.h', 'w') as file:
	file.write(header_text)
with open('modelLite.cc', 'w') as file:
	file.write(source_text)
