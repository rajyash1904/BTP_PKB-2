import os
import argparse
import numpy as np
import tensorflow as tf
import time
import importlib 
import subprocess
tf.enable_eager_execution()


import models.model_voxception as model
from models.entropy_model import EntropyBottleneck
from models.conditional_entropy_model import SymmetricConditional


################### Compression Network (with factorized entropy model) ##################

def compress_factorized(cubes, model, ckpt_dir):
    """Compress cubes to bitstream.
    Input: cubes with shape [batch size, length, width, height, channel(1)].
    Output: compressed bitstearm.
    """

    print('===== Compress =====')
    # load model.
    #model = importlib.import_module(model)
    analysis_transform = model.AnalysisTransform()
    # synthesis_transform = model.SynthesisTransform()
    entropy_bottleneck = EntropyBottleneck()
    checkpoint = tf.train.checkpoint(analysis_transform=analysis_transform, estimator=entropy_bottleneck)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))


    x = tf.convert_to_tensor(cubes, "float32")

    def loop_analysis(x):
        x = tf.expand_dims(x, 0)
        y = analysis_transform(x)
        return tf.squeeze(y)


    start = time.time()
    ys = tf.map_fn(loop_analysis, x, dtype=tf.float32, parallel_iterations=1, back_prop=False)
    print("Analysis Transform: {}s".format(round(time.time()-start,4)))

    start = time.time()
    strings, min_v, max_v = entropy_bottleneck.compress(ys)
    shape = tf.shape(ys)[:]
    print("Entropy Encode: {}s".format(round(time.time()-start, 4)))

    return strings, min_v, max_v, shape

def decompress_factorized(strings, min_v, max_v, shape, model, ckpt_dir):
    """Decompress bitstream to cubes.
    Input: compressed bitstream.
    Output: cubes with shape [batch size, length, width, height, channel(1)]
    """

    print('===== Decompress =====')
    # load model.
    #model = importlib.import_module(model)
    # analysis_transform = model.AnalysisTransform()
    synthesis_transform = model.SynthesisTransform()
    entropy_bottleneck = EntropyBottleneck()
    checkpoint = tf.train.Checkpoint(synthesis_transform=synthesis_transform, 
                                    estimator=entropy_bottleneck)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

    start = time.time()
    ys = entropy_bottleneck.decompress(strings, min_v, max_v, shape, shape[-1])
    print("Entropy Decode: {}s".format(round(time.time()-start, 4)))

    def loop_synthesis(y):  
        y = tf.expand_dims(y, 0)
        x = synthesis_transform(y)
        return tf.squeeze(x, [0])

    start = time.time()
    xs = tf.map_fn(loop_synthesis, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)
    print("Synthesis Transform: {}s".format(round(time.time()-start, 4)))

    return xs

################### Compression Network (with conditional entropy model) ###################

def compress_hyper(cubes, model, ckpt_dir, decompress=False):
    """Compress cubes to bitstream.
    Input: cubes with shape [batch size, length, width, height, channel(1)].
    Output: compressed bitstream.
    """

    print('===== Compress =====')
    # load model.
    #model = importlib.import_module(model)
    analysis_transform = model.AnalysisTransform()
    synthesis_transform = model.SynthesisTransform()
    hyper_encoder = model.HyperEncoder()
    hyper_decoder = model.HyperDecoder()
    entropy_bottleneck = EntropyBottleneck()
    conditional_entropy_model = SymmetricConditional()

    checkpoint = tf.train.Checkpoint(analysis_transform=analysis_transform, 
                                    synthesis_transform=synthesis_transform, 
                                    hyper_encoder=hyper_encoder, 
                                    hyper_decoder=hyper_decoder, 
                                    estimator=entropy_bottleneck)
    status = checkpoint.restore(tf.train.latest_checkpoint(ckpt_dir))

    x = tf.convert_to_tensor(cubes, "float32")

    def loop_analysis(x):
        x = tf.expand_dims(x, 0)
        y = analysis_transform(x)
        return tf.squeeze(y)

    start = time.time()
    ys = tf.map_fn(loop_analysis, x, dtype=tf.float32, parallel_iterations=1, back_prop=False)
    print("Analysis Transform: {}s".format(round(time.time()-start, 4)))

    def loop_hyper_encoder(y):
        y = tf.expand_dims(y, 0)
        z = hyper_encoder(y)
        return tf.squeeze(z)

    start = time.time()
    zs = tf.map_fn(loop_hyper_encoder, ys, dtype=tf.float32, parallel_iterations=1, back_prop=False)     
    print("Hyper Encoder: {}s".format(round(time.time()-start, 4)))

    z_hats, _ = entropy_bottleneck(zs, False)
    print("Quantize hyperprior.")

