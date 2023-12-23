# -*- coding: UTF-8 -*-
import pycnnl
import time
import numpy as np
import os
import scipy.io


class VGG19(object):
    def __init__(self):
        # set up net

        self.net = pycnnl.CnnlNet()
        self.input_quant_params = []
        self.filter_quant_params = []

    def build_model(self, param_path='../../imagenet-vgg-verydeep-19.mat'):
        self.param_path = param_path

        # TODO: 使用net的createXXXLayer接口搭建VGG19网络
        # creating layers
        self.net.setInputShape(1, 3, 224, 224)

        # conv1, relu1
        input_shape11 = pycnnl.IntVector(4)
        input_shape11[0] = 1
        input_shape11[1] = 3
        input_shape11[2] = 224
        input_shape11[3] = 224
        self.net.createConvLayer('conv1_1', input_shape11, 64, 3, 1, 1, 1)
        self.net.createReLuLayer('relu1_1')
        input_shape12 = pycnnl.IntVector(4)
        input_shape12[0] = 1
        input_shape12[1] = 64
        input_shape12[2] = 224
        input_shape12[3] = 224
        self.net.createConvLayer('conv1_2', input_shape12, 64, 3, 1, 1, 1)
        self.net.createReLuLayer('relu1_2')

        # pool1
        self.net.createPoolingLayer('pool1', input_shape12, 2, 2)

        # conv2, relu2
        input_shape21 = pycnnl.IntVector(4)
        input_shape21[0] = 1
        input_shape21[1] = 64
        input_shape21[2] = 112
        input_shape21[3] = 112
        self.net.createConvLayer('conv2_1', input_shape21, 128, 3, 1, 1, 1)
        self.net.createReLuLayer('relu2_1')
        input_shape22 = pycnnl.IntVector(4)
        input_shape22[0] = 1
        input_shape22[1] = 128
        input_shape22[2] = 112
        input_shape22[3] = 112
        self.net.createConvLayer('conv2_2', input_shape22, 128, 3, 1, 1, 1)
        self.net.createReLuLayer('relu2_2')

        # pool2
        self.net.createPoolingLayer('pool2', input_shape22, 2, 2)

        # conv3, relu3
        input_shape31 = pycnnl.IntVector(4)
        input_shape31[0] = 1
        input_shape31[1] = 128
        input_shape31[2] = 56
        input_shape31[3] = 56
        self.net.createConvLayer('conv3_1', input_shape31, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_1')
        input_shape32 = pycnnl.IntVector(4)
        input_shape32[0] = 1
        input_shape32[1] = 256
        input_shape32[2] = 56
        input_shape32[3] = 56
        self.net.createConvLayer('conv3_2', input_shape32, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_2')
        self.net.createConvLayer('conv3_3', input_shape32, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_3')
        self.net.createConvLayer('conv3_4', input_shape32, 256, 3, 1, 1, 1)
        self.net.createReLuLayer('relu3_4')

        # pool3
        self.net.createPoolingLayer('pool3', input_shape32, 2, 2)

        # conv4, relu4
        input_shape41 = pycnnl.IntVector(4)
        input_shape41[0] = 1
        input_shape41[1] = 256
        input_shape41[2] = 28
        input_shape41[3] = 28
        self.net.createConvLayer('conv4_1', input_shape41, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_1')
        input_shape42 = pycnnl.IntVector(4)
        input_shape42[0] = 1
        input_shape42[1] = 512
        input_shape42[2] = 28
        input_shape42[3] = 28
        self.net.createConvLayer('conv4_2', input_shape42, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_2')
        self.net.createConvLayer('conv4_3', input_shape42, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_3')
        self.net.createConvLayer('conv4_4', input_shape42, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu4_4')

        # pool4
        self.net.createPoolingLayer('pool4', input_shape42, 2, 2)

        # conv5, relu5
        input_shape51 = pycnnl.IntVector(4)
        input_shape51[0] = 1
        input_shape51[1] = 512
        input_shape51[2] = 14
        input_shape51[3] = 14
        self.net.createConvLayer('conv5_1', input_shape51, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_1')
        self.net.createConvLayer('conv5_2', input_shape51, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_2')
        self.net.createConvLayer('conv5_3', input_shape51, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_3')
        self.net.createConvLayer('conv5_4', input_shape51, 512, 3, 1, 1, 1)
        self.net.createReLuLayer('relu5_4')

        # pool5
        self.net.createPoolingLayer('pool5', input_shape51, 2, 2)

        # fc6 relu6
        input_shapem1 = pycnnl.IntVector(4)
        input_shapem1[0] = 1
        input_shapem1[1] = 1
        input_shapem1[2] = 1
        input_shapem1[3] = 25088
        weight_shapem1 = pycnnl.IntVector(4)
        weight_shapem1[0] = 1
        weight_shapem1[1] = 1
        weight_shapem1[2] = 25088
        weight_shapem1[3] = 4096
        output_shapem1 = pycnnl.IntVector(4)
        output_shapem1[0] = 1
        output_shapem1[1] = 1
        output_shapem1[2] = 1
        output_shapem1[3] = 4096
        self.net.createMlpLayer('fc6', input_shapem1, weight_shapem1, output_shapem1)
        self.net.createReLuLayer('relu6')

        # fc7 relu7
        input_shapem2 = pycnnl.IntVector(4)
        input_shapem2[0] = 1
        input_shapem2[1] = 1
        input_shapem2[2] = 1
        input_shapem2[3] = 4096
        weight_shapem2 = pycnnl.IntVector(4)
        weight_shapem2[0] = 1
        weight_shapem2[1] = 1
        weight_shapem2[2] = 4096
        weight_shapem2[3] = 4096
        output_shapem2 = pycnnl.IntVector(4)
        output_shapem2[0] = 1
        output_shapem2[1] = 1
        output_shapem2[2] = 1
        output_shapem2[3] = 4096
        self.net.createMlpLayer('fc7', input_shapem2, weight_shapem2, output_shapem2)
        self.net.createReLuLayer('relu7')

        # fc8
        input_shapem3 = pycnnl.IntVector(4)
        input_shapem3[0] = 1
        input_shapem3[1] = 1
        input_shapem3[2] = 1
        input_shapem3[3] = 4096
        weight_shapem3 = pycnnl.IntVector(4)
        weight_shapem3[0] = 1
        weight_shapem3[1] = 1
        weight_shapem3[2] = 4096
        weight_shapem3[3] = 1000
        output_shapem3 = pycnnl.IntVector(4)
        output_shapem3[0] = 1
        output_shapem3[1] = 1
        output_shapem3[2] = 1
        output_shapem3[3] = 1000
        self.net.createMlpLayer('fc8', input_shapem3, weight_shapem3, output_shapem3)

        # softmax
        input_shapes = pycnnl.IntVector(3)
        input_shapes[0] = 1
        input_shapes[1] = 1
        input_shapes[2] = 1000
        self.net.createSoftmaxLayer('softmax', input_shapes, 1)

    def load_model(self):
        # loading params ...
        print('Loading parameters from file ' + self.param_path)
        params = scipy.io.loadmat(self.param_path)
        self.image_mean = params['normalization'][0][0][0]
        self.image_mean = np.mean(self.image_mean, axis=(0, 1))

        count = 0
        for idx in range(self.net.size()):
            if 'conv' in self.net.getLayerName(idx):
                weight, bias = params['layers'][0][idx][0][0][0][0]
                # TODO：调整权重形状
                # matconvnet: weights dim [height, width, in_channel, out_channel]
                # ours: weights dim [out_channel, height, width,in_channel]
                weight = np.transpose(weight, [3, 0, 1, 2]).flatten().astype(np.float) # 转换为一维数组
                bias = bias.reshape(-1).astype(np.float)
                self.net.loadParams(idx, weight, bias)
                count += 1
            if 'fc' in self.net.getLayerName(idx):
                # Loading params may take quite a while. Please be patient.
                weight, bias = params['layers'][0][idx][0][0][0][0]
                weight = weight.reshape([weight.shape[0]*weight.shape[1]*weight.shape[2], weight.shape[3]])
                weight = weight.flatten().astype(np.float)
                bias = bias.reshape(-1).astype(np.float)
                self.net.loadParams(idx, weight, bias)
                count += 1

    def load_image(self, image_dir):
        # loading image
        self.image = image_dir
        image_mean = np.array([123.68, 116.779, 103.939])
        print('Loading and preprocessing image from ' + image_dir)
        input_image = scipy.misc.imread(image_dir)
        input_image = scipy.misc.imresize(input_image, [224, 224, 3])
        input_image = np.array(input_image).astype(np.float32)
        input_image -= image_mean
        input_image = np.reshape(input_image, [1]+list(input_image.shape))
        # input dim [N, height, width, channel] 2
        # TODO：调整输入数据
        input_data = input_image.flatten().astype(np.float)  # input as a vector

        self.net.setInputData(input_data)

    def forward(self):
        return self.net.forward()

    def get_top5(self, label):
        start = time.time()
        self.forward()
        end = time.time()

        result = self.net.getOutputData()

        # loading labels
        labels = []
        with open('../synset_words.txt', 'r') as f:
            labels = f.readlines()

        # print results
        top1 = False
        top5 = False
        print('------ Top 5 of ' + self.image + ' ------')
        prob = sorted(list(result), reverse=True)[:6]
        if result.index(prob[0]) == label:
            top1 = True
        for i in range(5):
            top = prob[i]
            idx = result.index(top)
            if idx == label:
                top5 = True
            print('%f - ' % top + labels[idx].strip())

        print('inference time: %f' % (end - start))
        return top1, top5

    def evaluate(self, file_list):
        top1_num = 0
        top5_num = 0
        total_num = 0

        start = time.time()
        with open(file_list, 'r') as f:
            file_list = f.readlines()
            total_num = len(file_list)
            for line in file_list:
                image = line.split()[0].strip()
                label = int(line.split()[1].strip())
                vgg.load_image(image)
                top1, top5 = vgg.get_top5(label)
                if top1:
                    top1_num += 1
                if top5:
                    top5_num += 1
        end = time.time()

        print('Global accuracy : ')
        print('accuracy1: %f (%d/%d) ' %
              (float(top1_num)/float(total_num), top1_num, total_num))
        print('accuracy5: %f (%d/%d) ' %
              (float(top5_num)/float(total_num), top5_num, total_num))
        print('Total execution time: %f' % (end - start))


if __name__ == '__main__':
    vgg = VGG19()
    vgg.build_model()
    vgg.load_model()
    vgg.evaluate('../file_list')
