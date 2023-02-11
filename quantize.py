# Keras quantization
# Author: Van Quan PHAM
# date: 5/2/2023

import numpy as np
#import os  
import tensorflow as tf

# Find max, min element in array 2D
def min_value(arr):
    return max([value for row in arr for value in row])
def min_value(arr):
    return min([value for row in arr for value in row])


# Find number bits of data type of input value
# def data_checkType(inputs):
#     # Check data type of inputs
#     d_type = type(inputs[0][0][0]) 
#     if d_type == 'float':
#         nBits_int = 8 
#         nBits_frac = 23
#     # elif d_type == double:
#     #     nBits_int = 11
#     #     nBits_frac = 52
#     #else:
#         #raise DataType_Invalid("Data type of inputs is neither float nor double")
#     return [nBits_int, nBits_frac]

"""
Brief: Quantization Convolution 2D.
Input:
    - inputs: data in type of floating point
    - weights: weights in type of floating point
    - quant_bits: Number of bits in quantization
Output:
    - weights: weights in type of integer with quant_bits
    - outputs: Dequantized output n type of floating point
Note: 
    - Zf = (fmax-fmin)/2
    - Zq = 0
"""

# Quantization and dequantization custom
class Quantization(tf.keras.layers.Layer):
    def __init__(self, num_bits, **kwargs):
        super(Quantization, self).__init__(**kwargs)
        self.num_bits = num_bits

    def call(self, inputs):
        # #[nBits_int, nBits_frac] = data_checkType(inputs)
        # [nBits_int, nBits_frac] = [8,23]
        # # Find max, min of inputs
        # min_in = min(inputs)*nBits_frac   # convert to integer 31 bits
        # max_in = max(inputs)*nBits_frac
        # med_in = (max_in + min_in)*nBits_frac/2
        # scale_in = (max_in - min_in)/(2**self.num_bits)
        # return [scale_in, np.round((inputs*nBits_frac - med_in)/scale_in)]

        scale = 2**self.num_bits - 1
        return tf.round(inputs * scale) / scale

class Dequantization(tf.keras.layers.Layer):
    def __init__(self, num_bits, **kwargs):
        super(Dequantization, self).__init__(**kwargs)
        self.num_bits = num_bits

    def call(self, inputs):
        scale = 2**self.num_bits - 1
        return inputs / scale


# class DeQuantization(tf.keras.layers.Layer):
#     def __init__(self, scale_in, **kwargs):   # TODO: check again!
#         super(Dequantization, self).__init__(**kwargs)
#         self.scale_in = scale_in

#     def call(self, inputs):
#         scale = scale_in
#         return inputs * scale

# # Get weights
# import numpy as np

# def quantize_weights(weights, num_bits):
#     scale = 2**(num_bits - 1) - 1
#     quantized_weights = np.round(weights * scale) / scale
#     return quantized_weights

# model = ...  # Your Keras model
# layers = model.layers
# for layer in layers:
#     if hasattr(layer, 'weights'):
#         weights = layer.get_weights()
#         quantized_weights = [quantize_weights(w, 8) for w in weights]
#         layer.set_weights(quantized_weights)


# # fixing
# def quantize_conv2d(inputs, weights, quant_bits):*
#     # Analyze histogram and select range of great probability data
#     # TODO: to improve quantization

#     # Get number of bit integer and fraction of data input
#     [nBits_int, nBits_frac] = data_checkType(inputs)

#     # Find max, min of inputs
#     min_in = min(inputs)*nBits_frac   # convert to integer 31 bits
#     max_in = max(inputs)*nBits_frac
#     med_in = (max_in + min_in)*nBits_frac/2
#     scale_in = (max_in - min_in)/(2**(quant_bits)

#     # Find max, min of weights
#     min_w = min(weights)*nBits_frac   # convert to integer 31 bits
#     max_w = max(weights)*nBits_frac
#     med_w = (max_w + min_w)*nBits_frac/2
#     scale_w = (max_w - min_w)/(2**(quant_bits)

#     # Quantize inputs
#     inputs = np.round((inputs*nBits_frac - med_in)/scale_in)

#     # Quantize weights
#     weights = np.round((weights*nBits_frac - med_w) / scale_w)
#     # TODO:
#     # Perform convolution
#     outputs = np.zeros_like(inputs)
#     for i in range(inputs.shape[0]):
#         for j in range(weights.shape[0]):
#             outputs[i, j] = np.sum(inputs[i] * weights[j])
    
#     # Dequantize outputs
#     outputs = outputs * 2**(quant_bits - 1)
    
#     return [weights, outputs]   


