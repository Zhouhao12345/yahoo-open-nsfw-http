#!/usr/bin/env python

import numpy
import PIL
from PIL import Image
import caffe
from io import BytesIO
import os
import grpc
import data_pb2
import data_pb2_grpc
import time
import base64
from concurrent import futures

_ONE_DAY_IN_SECONDS = 60 * 60 * 24

def resize_image(image_stream, sz=(256, 256)):
    im = PIL.Image.open(image_stream)
    if im.mode != 'RGB':
        im = im.convert('RGB')
    imr = im.resize(sz, resample=PIL.Image.BILINEAR)
    imrh = BytesIO()
    imr.save(imrh, format='JPEG')
    imrh.seek(0)
    return imrh

def score_image(image_stream, net=None, transformer=None):
    if net is None:
        return []

    output_layers=['prob']

    img_data_rs = resize_image(image_stream, sz=(256, 256))
    image = caffe.io.load_image(img_data_rs)

    H, W, _ = image.shape
    _, _, h, w = net.blobs['data'].data.shape
    h_off = max((H - h) / 2, 0)
    w_off = max((W - w) / 2, 0)
    crop = image[h_off:h_off + h, w_off:w_off + w, :]
    transformed_image = transformer.preprocess('data', crop)
    transformed_image.shape = (1,) + transformed_image.shape

    input_name = net.inputs[0]
    all_outputs = net.forward_all(blobs=output_layers, **{input_name: transformed_image})

    outputs = all_outputs[output_layers[0]][0].astype(float)
    return outputs[1]

def make_transformer(net):
    transformer = caffe.io.Transformer({'data': net.blobs['data'].data.shape})
    transformer.set_transpose('data', (2, 0, 1))                # move image channels to outermost
    transformer.set_mean('data', numpy.array([104, 117, 123]))  # subtract the dataset-mean value in each channel
    transformer.set_raw_scale('data', 255)                      # rescale from [0, 1] to [0, 255]
    transformer.set_channel_swap('data', (2, 1, 0))             # swap channels from RGB to BGR
    return transformer

class Greeter(data_pb2_grpc.GreeterServicer):

    def CheckImageFace(self, request, context):
        model_def = os.getenv('MODEL_DEF', 'model_def.prototxt')
        pretrained_model = os.getenv('PRETRAINED_MODEL', 'pretrained_model.caffemodel')
        img_str = base64.b64decode(request.imageBuffer)
        net = caffe.Net(model_def, pretrained_model, caffe.TEST)
        transformer = make_transformer(net)
        buffer = BytesIO(img_str)
        score = score_image(buffer, net=net, transformer=transformer)
        return data_pb2.CheckReply(checkResult=str(score))

def main():
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10))
    data_pb2_grpc.add_GreeterServicer_to_server(Greeter(), server)
    server.add_insecure_port('[::]:50052')
    server.start()
    try:
        while True:
            time.sleep(_ONE_DAY_IN_SECONDS)
    except KeyboardInterrupt:
        server.stop(0)

if __name__ == '__main__':
    main()
