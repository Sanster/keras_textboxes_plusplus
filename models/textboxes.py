import numpy as np
from keras.layers import MaxPooling2D, Conv2D, ZeroPadding2D
from keras.models import Model


class TextBoxes(object):
    def __init__(self):
        self.build_loss()

    def build_model(self, data):
        conv1_1 = self._conv2d(data, 64, (3, 3), name='conv1_1')
        conv1_2 = self._conv2d(conv1_1, 64, (3, 3), name='conv1_2')
        pool1 = self._pooling(conv1_2, (2, 2), (2, 2), name='pool1')

        conv2_1 = self._conv2d(pool1, 128, (3, 3), name='conv2_1')
        conv2_2 = self._conv2d(conv2_1, 128, (3, 3), name='conv2_2')
        pool2 = self._pooling(conv2_2, (2, 2), (2, 2), name='pool2')

        conv3_1 = self._conv2d(pool2, 256, (3, 3), name='conv3_1')
        conv3_2 = self._conv2d(conv3_1, 256, (3, 3), name='conv3_2')
        conv3_3 = self._conv2d(conv3_2, 256, (3, 3), name='conv3_3')
        pool3 = self._pooling(conv3_3, (2, 2), (2, 2), name='pool3')

        conv4_1 = self._conv2d(pool3, 512, (3, 3), name='conv4_1')
        conv4_2 = self._conv2d(conv4_1, 512, (3, 3), name='conv4_2')
        conv4_3 = self._conv2d(conv4_2, 512, (3, 3), name='conv4_3')
        pool4 = self._pooling(conv4_3, (2, 2), (2, 2), name='pool4')

        conv5_1 = self._conv2d(pool4, 512, (3, 3), name='conv5_1')
        conv5_2 = self._conv2d(conv5_1, 512, (3, 3), name='conv5_2')
        conv5_3 = self._conv2d(conv5_2, 512, (3, 3), name='conv5_3')
        # SSD change pool5 from 2x2-s2 to 3x3-s1
        pool5 = self._pooling(conv5_3, (3, 3), (1, 1), 'pool5')

        fc6 = Conv2D(1024, (3, 3), dilation_rate=(6, 6), activation='relu', padding='same',
                     kernel_initializer='glorot_uniform', name='fc6')(pool5)

        fc7 = Conv2D(1024, (1, 1), activation='relu', padding='same',
                     kernel_initializer='glorot_uniform', name='fc7')(fc6)

        # TODO: 比较 SSD 和 Textboxes 的 caffe model
        conv6_1 = self._conv2d(fc7, 256, (1, 1), name='conv6_1')
        conv6_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv6_padding')(conv6_1)
        conv6_2 = self._conv2d(conv6_1, 512, (3, 3), (2, 2), 'valid', 'conv6_2')

        conv7_1 = self._conv2d(conv6_2, 128, (1, 1), name='conv7_1')
        conv7_1 = ZeroPadding2D(padding=((1, 1), (1, 1)), name='conv7_padding')(conv7_1)
        conv7_2 = self._conv2d(conv7_1, 256, (3, 3), (2, 2), 'valid', 'conv7_2')

        conv8_1 = self._conv2d(conv7_2, 128, (1, 1), name='conv8_1')
        conv8_2 = self._conv2d(conv8_1, 256, (3, 3), (1, 1), 'valid', name='conv8_2')

        conv9_1 = self._conv2d(conv8_2, 128, (1, 1), name='conv9_1')
        conv9_2 = self._conv2d(conv9_1, 256, (3, 3), (1, 1), 'valid', name='conv9_2')


    def build_loss(self):
        pass

    def _conv2d(self, data, filters, kernel_size, strides=(1, 1), padding='same', name=''):
        return Conv2D(filters, kernel_size,
                      activation='relu',
                      padding=padding,
                      strides=strides,
                      kernel_regularizer=None,  # TODO: which to use
                      kernel_initializer='glorot_uniform',
                      bias_initializer='zeros',
                      name=name)(data)

    def _pooling(self, data, pool_size, strides, name):
        return MaxPooling2D(pool_size=pool_size,
                            strides=strides,
                            padding='same',
                            name=name)(data)


if __name__ == '__main__':
    model = TextBoxes()
