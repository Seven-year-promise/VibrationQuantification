import sys

from tensorflow.python.tools import freeze_graph

sys.path.append('../..')
from Methods.Wellplate_6.util import *
import time
from datetime import timedelta
from Methods.Wellplate_6.data import *
import numpy as np
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
from PIL import Image
import tensorflow as tf
#tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from tensorflow.python.ops import clip_ops
from tensorflow.python.ops import math_ops

class UNet(object):

    def __init__(self, sess, conf):
        """
        本函数是UNet网络的初始化文件，用于构建网络结构、损失函数、优化方法。
        :param sess: tensorflow会话
        :param conf: 配置
        """
        self.sess = sess
        self.conf = conf
        # 输入图像
        self.images = tf.placeholder(tf.float32, shape=[None, self.conf.im_size, self.conf.im_size, 1], name='x')
        # 标注
        self.annotations = tf.placeholder(tf.float32, shape=[None, self.conf.im_size, self.conf.im_size, 2], name='annotations')
        # 构建UNet网络结构
        self.predict = self.inference() #self.LightCNN2() #
        # 损失函数，分类精度
        self.loss_op = self.combined_loss()
        self.accuracy_op = self.accuracy()
        # 优化方法
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            self.optimizer = self.train_op()
        # 初始化参数
        self.sess.run(tf.global_variables_initializer())
        # 保存所有可训练的参数
        trainable_vars = tf.trainable_variables()
        g_list = tf.global_variables()
        bn_moving_vars = [g for g in g_list if 'moving_mean' in g.name]
        bn_moving_vars += [g for g in g_list if 'moving_variance' in g.name]
        # 模型保存和保存summary的工具
        self.saver = tf.train.Saver(var_list=trainable_vars + bn_moving_vars, max_to_keep=0)
        self.writer = tf.summary.FileWriter(self.conf.logdir, self.sess.graph)
        self.train_summary = self.config_summary('train')

        self.eval_im_anno_list = []
        self.eval_FLAG = False

    def config_summary(self, name):
        summarys = [tf.summary.scalar(name + '/loss', self.loss_op)]
        summary = tf.summary.merge(summarys)
        return summary

    def save_ckpt(self, directory, filename):

        if not os.path.exists(directory):
            os.makedirs(directory)
        filepath = os.path.join(directory, filename + '.ckpt')
        self.saver.save(self.sess, filepath)
        return filepath

    def save(self, step):
        print('saving', end=' ')
        if not os.path.exists(self.conf.modeldir):
            os.makedirs(self.conf.modeldir)

        # Save check point for graph frozen later
        ckpt_filepath = self.save_ckpt(directory=self.conf.modeldir, filename=self.conf.model_name)
        pbtxt_filename = self.conf.model_name + str(step) + '.pbtxt'
        pbtxt_filepath = os.path.join(self.conf.modeldir, pbtxt_filename)
        pb_filepath = os.path.join(self.conf.modeldir, self.conf.model_name + str(step) + '.pb')
        # This will only save the graph but the variables will not be saved.
        # You have to freeze your model first.


        self.saver.save(self.sess, ckpt_filepath, global_step=step)

        tf.train.write_graph(graph_or_graph_def=self.sess.graph_def, logdir=self.conf.modeldir, name=pbtxt_filename, as_text=True)


    def reload(self, step):
        checkpoint_path = os.path.join(
            self.conf.modeldir, self.conf.model_name + ".ckpt")
        model_path = checkpoint_path + '-' + str(step)
        if not os.path.exists(model_path + '.meta'):
            print('------- no such checkpoint', model_path)
            return False
        self.saver.restore(self.sess, model_path)
        return True

    def save_summary(self, summary, step):
        print('summarizing', end=' ')
        self.writer.add_summary(summary, step)

    def inference(self):
        conv1 = conv(self.images, shape=[3, 3, 1, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv1 = conv(conv1, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool1 = max_pool(conv1, size=2)

        conv2 = conv(pool1, shape=[3, 3, 64, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv2 = conv(conv2, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool2 = max_pool(conv2, size=2)

        conv3 = conv(pool2, shape=[3, 3, 128, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv3 = conv(conv3, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool3 = max_pool(conv3, size=2)

        conv4 = conv(pool3, shape=[3, 3, 256, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv4 = conv(conv4, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        pool4 = max_pool(conv4, size=2)

        conv5 = conv(pool4, shape=[3, 3, 512, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv5 = conv(conv5, shape=[3, 3, 1024, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up6 = deconv(conv5, shape=[2, 2, 512, 1024], stride=2, stddev=0.1)
        merge6 = concat(up6, conv4)
        conv6 = conv(merge6, shape=[3, 3, 1024, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv6 = conv(conv6, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up7 = deconv(conv6, shape=[2, 2, 256, 512], stride=2, stddev=0.1)
        merge7 = concat(up7, conv3)
        conv7 = conv(merge7, shape=[3, 3, 512, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv7 = conv(conv7, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up8 = deconv(conv7, shape=[2, 2, 128, 256], stride=2, stddev=0.1)
        merge8 = concat(up8, conv2)
        conv8 = conv(merge8, shape=[3, 3, 256, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv8 = conv(conv8, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1)

        up9 = deconv(conv8, shape=[2, 2, 64, 128], stride=2, stddev=0.1)
        merge9 = concat(up9, conv1)
        conv9 = conv(merge9, shape=[3, 3, 128, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)
        conv9 = conv(conv9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1)

        predict = conv(conv9, shape=[3, 3, 64, 2], stddev=0.1,
                       is_training=self.conf.is_training, stride=1,
                       name="cnn/output",
                       activation=False)

        return predict

    def LightCNN(self):
        conv1a = conv(self.images, shape=[3, 3, 1, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv1a", activation=False)
        MFM1a = MFM(conv1a, name = "MFM1a") # out 32 channels
        conv1b = conv(MFM1a, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv1b", activation=False)
        MFM1b = MFM(conv1b, name = "MFM1b") # out 32 channels
        pool1 = max_pool(MFM1b, size=2)

        conv2a = conv(pool1, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv2a", activation=False)
        MFM2a = MFM(conv2a, name="MFM2a") # out 32 channels
        conv2b = conv(MFM2a, shape=[3, 3, 32, 128], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv2b", activation=False)
        MFM2b = MFM(conv2b, name="MFM2b") # out 64 channels
        pool2 = max_pool(MFM2b, size=2)

        conv3a = conv(pool2, shape=[3, 3, 64, 128], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv3a", activation=False)
        MFM3a = MFM(conv3a, name="MFM3a")  # out 64 channels
        conv3b = conv(MFM3a, shape=[3, 3, 64, 256], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv3b", activation=False)
        MFM3b = MFM(conv3b, name="MFM3b")  # out 128 channels
        pool3 = max_pool(MFM3b, size=2)

        conv4a = conv(pool3, shape=[3, 3, 128, 256], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv4a", activation=False)
        MFM3a = MFM(conv4a, name="MFM4a")  # out 128 channels
        conv4b = conv(MFM3a, shape=[3, 3, 128, 512], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv4b", activation=False)
        MFM4b = MFM(conv4b, name="MFM4b")  # out 256 channels
        pool4 = max_pool(MFM4b, size=2)

        conv5a = conv(pool4, shape=[3, 3, 256, 512], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv5a", activation=False)
        MFM5a = MFM(conv5a, name="MFM5a")  # out 256 channels
        conv5b = conv(MFM5a, shape=[3, 3, 256, 1024], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv5b", activation=False)
        MFM5b = MFM(conv5b, name="MFM5b")  # out 512 channels

        up6 = deconv(MFM5b, shape=[2, 2, 256, 512], stride=2, stddev=0.1)
        merge6 = concat(up6, MFM4b)
        conv6a = conv(merge6, shape=[3, 3, 512, 512], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv6b", activation=False)
        MFM6a = MFM(conv6a, name="MFM6a")  # out 256 channels
        conv6b = conv(MFM6a, shape=[3, 3, 256, 512], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv6b", activation=False)
        MFM6b = MFM(conv6b, name="MFM6b")  # out 256 channels

        up7 = deconv(MFM6b, shape=[2, 2, 128, 256], stride=2, stddev=0.1)
        merge7 = concat(up7, MFM3b)
        conv7a = conv(merge7, shape=[3, 3, 256, 256], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv7a", activation=False)
        MFM7a = MFM(conv7a, name="MFM7a")  # out 128 channels
        conv7b = conv(MFM7a, shape=[3, 3, 128, 256], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv7b", activation=False)
        MFM7b = MFM(conv7b, name="MFM7b")  # out 128 channels

        up8 = deconv(MFM7b, shape=[2, 2, 64, 128], stride=2, stddev=0.1)
        merge8 = concat(up8, MFM2b)
        conv8a = conv(merge8, shape=[3, 3, 128, 128], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv8a", activation=False)
        MFM8a = MFM(conv8a, name="MFM8a")  # out 64 channels
        conv8b = conv(MFM8a, shape=[3, 3, 64, 128], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv8b", activation=False)
        MFM8b = MFM(conv8b, name="MFM8b")  # out 64 channels

        up9 = deconv(MFM8b, shape=[2, 2, 32, 64], stride=2, stddev=0.1)
        merge9 = concat(up9, MFM1b)
        conv9a = conv(merge9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv9a", activation=False)
        MFM9a = MFM(conv9a, name="MFM8a")  # out 32 channels
        conv9b = conv(MFM9a, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv9b", activation=False)
        MFM9b = MFM(conv9b, name="MFM8b")  # out 32 channels

        light_predict = conv(MFM9b, shape=[3, 3, 32, 2], stddev=0.1,
                       is_training=self.conf.is_training, stride=1,
                       name="cnn/output",
                       activation=False)

        return light_predict

    def LightCNN2(self):
        conv1a = conv(self.images, shape=[3, 3, 1, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv1a", activation=False)
        MFM1a = MFM(conv1a, name = "MFM1a") # out 32 channels
        conv1b = conv(MFM1a, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv1b", activation=False)
        MFM1b = MFM(conv1b, name = "MFM1b") # out 32 channels
        pool1 = max_pool(MFM1b, size=2)

        conv2a = conv(pool1, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv2a", activation=False)
        MFM2a = MFM(conv2a, name="MFM2a") # out 32 channels
        conv2b = conv(MFM2a, shape=[3, 3, 32, 128], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv2b", activation=False)
        MFM2b = MFM(conv2b, name="MFM2b") # out 64 channels

        up9 = deconv(MFM2b, shape=[2, 2, 32, 64], stride=2, stddev=0.1)
        merge9 = concat(up9, MFM1b)
        conv9a = conv(merge9, shape=[3, 3, 64, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv9a", activation=False)
        MFM9a = MFM(conv9a, name="MFM8a")  # out 32 channels
        conv9b = conv(MFM9a, shape=[3, 3, 32, 64], stddev=0.1, is_training=self.conf.is_training, stride=1, name = "conv9b", activation=False)
        MFM9b = MFM(conv9b, name="MFM8b")  # out 32 channels

        light_predict = conv(MFM9b, shape=[3, 3, 32, 2], stddev=0.1,
                       is_training=self.conf.is_training, stride=1,
                       name="cnn/output",
                       activation=False)

        return light_predict

    def loss(self, scope='loss'):
        """
        :return: 损失函数及分类精确度
        """
        # 标注 1通道-num_classes通道 one-hot
        with tf.variable_scope(scope):
            losses = tf.nn.softmax_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)
            loss_op = tf.reduce_mean(losses, name='loss/loss_op')
        return loss_op

    def binary_crossentropy(self, from_logits=False):
        """Binary crossentropy between an output tensor and a target tensor.
        Arguments:
          target: A tensor with the same shape as `output`.
          output: A tensor.
          from_logits: Whether `output` is expected to be a logits tensor.
              By default, we consider that `output`
              encodes a probability distribution.
        Returns:
          A tensor.
        """
        # Note: nn.sigmoid_cross_entropy_with_logits
        # expects logits, Keras expects probabilities.
        if not from_logits:
            # transform back to logits
            epsilon_ = 1e-4 #_to_tensor(epsilon(), self.predict.dtype.base_dtype)
            output = clip_ops.clip_by_value(self.predict, epsilon_, 1 - epsilon_)
            output = math_ops.log(output / (1 - output))
            output = tf.cast(output, dtype=tf.float32)
        return tf.nn.sigmoid_cross_entropy_with_logits(labels=self.annotations, logits=output)

    def dice_loss(self, scope='dice_loss', smooth=1.):
        with tf.variable_scope(scope):
            y_true = tf.cast(self.annotations, tf.float32)
            y_pred = tf.math.sigmoid(self.predict)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            loss_op = 1 - (numerator+smooth) / (denominator + smooth)

        return loss_op

    def combined_loss(self, scope='combine_loss', smooth=1.):
        with tf.variable_scope(scope):
            y_true = tf.cast(self.annotations, tf.float32)
            y_pred = tf.math.sigmoid(self.predict)
            numerator = 2 * tf.reduce_sum(y_true * y_pred)
            denominator = tf.reduce_sum(y_true + y_pred)
            #numerator = 2 * tf.reduce_mean(tf.reduce_sum(y_true * y_pred, axis=2), axis=2)
            #denominator = tf.reduce_sum(tf.reduce_sum(y_true, axis=2), axis=2) + tf.reduce_sum(tf.reduce_sum(y_pred, axis=2), axis=2)
            dice_loss = 1 - (numerator+smooth) / (denominator + smooth)
            dice_loss = tf.reduce_mean(dice_loss)
            dice_loss *= 1-self.conf.bce_weight

            #epsilon_ = 1e-4  # _to_tensor(epsilon(), self.predict.dtype.base_dtype)
            #output = clip_ops.clip_by_value(self.predict, epsilon_, 1 - epsilon_)
            #output = math_ops.log(output / (1 - output))
            #output = tf.cast(output, dtype=tf.float32)
            binary_loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)
            #binary_loss = tf.nn.softmax_cross_entropy_with_logits(labels=self.annotations, logits=self.predict)

            binary_loss = tf.reduce_mean(binary_loss)
            binary_loss *= self.conf.bce_weight
            loss_op = dice_loss + binary_loss#tf.concat([dice_loss, binary_loss], axis=-1)
            #loss_op = tf.reduce_mean(loss_op)
        return loss_op

    def accuracy(self, scope='accuracy'):
        with tf.variable_scope(scope):
            preds = tf.cast(self.predict, dtype=tf.int64) #tf.argmax(self.predict, -1, name='accuracy/decode_pred')
            targets = tf.cast(self.annotations, dtype=tf.int64)
            acc = 1.0 - tf.nn.zero_fraction(
                tf.cast(tf.equal(preds, targets), dtype=tf.int32))
        return acc

    def train_op(self):
        # params = tf.trainable_variables()
        # gradients = tf.gradients(self.loss_op, params, name='gradients')
        # optimizer = tf.train.MomentumOptimizer(self.conf.rate, 0.9)
        # update = optimizer.apply_gradients(zip(gradients, params))
        # with tf.control_dependencies([update]):
        #     train_op = tf.no_op(name='train_op')
        optimizer = tf.train.AdamOptimizer(learning_rate=self.conf.rate).minimize(self.loss_op)
        return optimizer

    def random_rotate(self, x, y0, y1):
        x_img = Image.fromarray(x)
        y_img0 = Image.fromarray(y0)
        y_img1 = Image.fromarray(y1)
        rotate = np.random.randint(0, 360)
        x_img_rotate = x_img.rotate(rotate, expand=False)
        y_img0_rotate = y_img0.rotate(rotate, expand=False)
        y_img1_rotate = y_img1.rotate(rotate, expand=False)

        x_cv_gray = np.asarray(x_img_rotate)

        y_cv0_gray = np.asarray(y_img0_rotate)
        y_cv1_gray = np.asarray(y_img1_rotate)

        return x_cv_gray, y_cv0_gray, y_cv1_gray

    def contrast_brightness(self, x):
        alpha = np.random.normal(0, 0.005, 1)[0] + 1.0
        #print(alpha)
        beta = np.random.randint(0, 10)

        return np.array(alpha * x + beta, np.uint8) #

    def gaussian_noise(self, x):
        gaussian_noise = x.copy()
        cv2.randn(gaussian_noise, 0, 100)

        return x + gaussian_noise

    def augmentation(self, im_size, x, y, random_rotate = True, contrast = True, noise = False):
        out_x = np.ones((x.shape[0], im_size, im_size, 1), x.dtype)
        out_y = np.ones((y.shape[0], im_size, im_size, y.shape[3]), y.dtype)
        num = x.shape[0]
        for n in range(num):
            #print(x.shape)
            x_copy = np.array(x[n, :, :, :], np.uint8)
            y_copy0 = np.array(y[n, :, :, 0], np.uint8)
            y_copy1 = np.array(y[n, :, :, 1], np.uint8)

            x_gray = cv2.cvtColor(x_copy, cv2.COLOR_BGR2GRAY)
            _, (well_x, well_y, _), im_well = well_detection(x_copy, x_gray)
            im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)

            if well_x < 240:
                x_d_edge = well_x
            else:
                x_d_edge = 480 - well_x
            if well_y < 240:
                y_d_edge = well_y
            else:
                y_d_edge = 480 - well_y
            if x_d_edge > y_d_edge:
                d_edge = y_d_edge
            else:
                d_edge = x_d_edge

            x_min = int(well_x - d_edge)
            x_max = int(well_x + d_edge)
            y_min = int(well_y - d_edge)
            y_max = int(well_y + d_edge)
            x_copy_block = im_well[y_min:y_max, x_min:x_max]
            y_copy_block0 = y_copy0[y_min:y_max, x_min:x_max]
            y_copy_block1 = y_copy1[y_min:y_max, x_min:x_max]
            #cv2.imshow("im_well", x_copy_block)
            if random_rotate:
                x_cv_gray, y_cv0_gray, y_cv1_gray = self.random_rotate(x_copy_block, y_copy_block0, y_copy_block1)
            else:
                x_cv_gray = x_copy_block
                y_cv0_gray = y_copy_block0
                y_cv1_gray = y_copy_block1

            x_min = int(d_edge - im_size / 2)
            x_max = int(d_edge + im_size / 2)
            y_min = int(d_edge - im_size / 2)
            y_max = int(d_edge + im_size / 2)
            #print(d_edge, y_min, y_max, x_min, x_max)
            x_block = x_cv_gray[y_min:y_max, x_min:x_max]

            if contrast:
                x_block = self.contrast_brightness(x_block)

            if noise:
                x_block = self.gaussian_noise(x_block)

            y_block0 = y_cv0_gray[y_min:y_max, x_min:x_max]
            y_block1 = y_cv1_gray[y_min:y_max, x_min:x_max]
            #cv2.imshow("x_block", x_block)
            #cv2.imshow("y_block0", y_block0*255)
            #cv2.imshow("y_block1", y_block1*255)
            #cv2.waitKey(0)

            out_x[n, :, :, 0] = np.array(x_block, dtype=x.dtype)/255 - 0.5
            out_y[n, :, :, 0] = np.array(y_block0, dtype=y.dtype)
            out_y[n, :, :, 1] = np.array(y_block1, dtype=y.dtype)
        """
        x = np.array(x, dtype = np.uint8)
        x
        _, (well_x, well_y, _), im_well = well_detection(x, x)
        im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
        x_min = int(well_x - 240 / 2)
        x_max = int(well_x + 240 / 2)
        y_min = int(well_y - 240 / 2)
        y_max = int(well_y + 240 / 2)
        im_block = im_well[y_min:y_max, x_min:x_max]
        img = np.array(im_block, dtype=np.float32)
        """
        return out_x, out_y

    def train(self):
        if self.conf.reload_step > 0:
            if not self.reload(self.conf.reload_step):
                return
            print('reload', self.conf.reload_step)

        images, labels = read_record(self.conf.datadir, self.conf.ori_size, self.conf.batch_size)
        #with tf.device("/device:XLA_GPU:0"):
        tf.train.start_queue_runners(sess=self.sess)
        print('Begin Train')
        print('Augmentation rotation:', self.conf.rotation, ' contrast:', self.conf.contrast, ' noise:',
              self.conf.noise)
        needle_accs = []
        needle_ius = []
        fish_accs = []
        fish_ius = []
        steps = []
        self.sess.graph.finalize()
        start_time = time.time()
        for train_step in range(self.conf.reload_step, self.conf.max_step + 1):


            x, y = self.sess.run([images, labels])
            x, y = self.augmentation(self.conf.im_size, x, y, random_rotate = self.conf.rotation, contrast = self.conf.contrast, noise = self.conf.noise)
            # summary
            #show = np.array(y[0, :, :, 1]*255, dtype = np.uint8)
            #cv2.imshow("show", show)

            #show2 = np.array(y[0, :, :, 0] * 255, dtype=np.uint8)
            #cv2.imshow("show2", show2)

            #show3 = np.array(x[0, :, :, 0] * 255 + 128, dtype=np.uint8)
            #cv2.imshow("shows3", show3)
            #cv2.waitKey(0)

            if train_step == 1 or train_step % self.conf.summary_interval == 0:
                feed_dict = {self.images: x,
                             self.annotations: y}
                loss, _, summary = self.sess.run(
                    [self.loss_op, self.optimizer, self.train_summary],
                    feed_dict=feed_dict)



                self.save_summary(summary, train_step + self.conf.reload_step)
                end_time = time.time()
                time_diff = end_time - start_time
                print(str(train_step), '----Training loss:', loss, "Time usage: " + str(time_diff))
            # print 损失和准确性
            else:
                feed_dict = {self.images: x,
                             self.annotations: y}
                loss, _ = self.sess.run(
                    [self.loss_op, self.optimizer], feed_dict=feed_dict)
                #print(str(train_step), '----Training loss:', loss, ' accuracy:', acc, end=' ')
            # 保存模型
            if train_step % self.conf.save_interval == 0:
                print("saving for ", self.conf.modeldir)
                self.save(train_step)
        """
                steps.append(train_step)
                needle_acc, needle_iu, fish_acc, fish_iu = self. eval(2, train_step)

                needle_accs.append(needle_acc)
                needle_ius.append(needle_iu)
                fish_accs.append(fish_acc)
                fish_ius.append(fish_iu)

                plt.plot(steps, needle_accs, marker=".")
                plt.plot(steps, needle_ius, marker="s")
                plt.plot(steps, fish_accs, marker="*")
                plt.plot(steps, fish_ius, marker="h")

                plt.legend(labels=["PC Needle", "JI Needle", "PC Larva", "JI Larva"], loc="best")
                plt.xlabel("Training Steps")
                plt.ylabel("Performance Indexes")
                plt.title("The performance od U-Net on testing dataset when training")
                plt.pause(0.05)
        plt.show()
        """

    def eval(self, batch_size, train_step):
        if not self.eval_FLAG:
            test_im_path = "data/test/Images/"
            test_anno_path = "data/test/annotation/"
            ims_name = os.listdir(test_im_path)
            annos_name = os.listdir(test_anno_path)
            im_anno_list = []
            for im_name in ims_name:
                name = im_name[:-4]
                im = cv2.imread(test_im_path + im_name)
                anno = cv2.imread(test_anno_path + name + "_label.tif")
                # anno = cv2.erode(anno, (3, 3), iterations=2)
                anno = anno[:, :, 1]
                anno_needle = np.zeros(anno.shape, dtype=np.uint8)
                anno_needle[np.where(anno == 1)] = 1
                anno_fish = np.zeros(anno.shape, dtype=np.uint8)
                anno_fish[np.where(anno == 2)] = 1

                im_anno_list.append([im, anno_needle, anno_fish])
            self.eval_im_anno_list = im_anno_list

        eval_num = len(self.eval_im_anno_list)

        split_num = int(eval_num / batch_size)

        #self.load_graph(
        #   os.path.join(self.conf.modeldir, self.conf.model_name + str(train_step + self.conf.reload_step) + '.pb'))

        if eval_num != (batch_size * split_num):
            split_num = split_num-1

        ave_needle_acc = 0
        ave_fish_acc = 0
        ave_needle_iu = 0
        ave_fish_iu = 0

        for n in range(1):
            im_patch, anno_patch = load_im(self.eval_im_anno_list[(n * batch_size):((n+1) * batch_size)])

            label = self.sess.run([self.predict],
                                  feed_dict={
                                      self.images: im_patch
                                  })
            label = np.squeeze(label)
            label = tf.convert_to_tensor(label)
            label = tf.nn.sigmoid(label, name='sigmoid')
            with tf.Session() as sess:
                label = sess.run(label)
            label = np.array(label)
            segmen_patch = label.reshape((-1, self.conf.im_size, self.conf.im_size, 2))

            needle_acc, fish_acc, needle_iu, fish_iu = eval(segmen_patch, anno_patch)
            ave_needle_acc += needle_acc
            ave_fish_acc += fish_acc
            ave_needle_iu += needle_iu
            ave_fish_iu += fish_iu

        return ave_needle_acc / split_num, \
           ave_fish_acc / split_num, \
           ave_needle_iu / split_num, \
           ave_fish_iu / split_num


    def segmen_patch(self, img_patch):
        label = self.sess.run([self.output],
                              feed_dict={
                                  self.input: img_patch
                              })
        label = np.squeeze(label)
        label = tf.convert_to_tensor(label)
        label = tf.nn.sigmoid(label, name='sigmoid')
        with tf.Session() as sess:
            label = sess.run(label)
        label = np.array(label)
        label = label.reshape((-1, self.conf.im_size, self.conf.im_size, 2))
        return label


    def segmen(self, img):
        label = self.sess.run([self.output],
                              feed_dict={
                                  self.input: img
                              })
        label = np.squeeze(label)
        label = tf.convert_to_tensor(label)
        label = tf.nn.sigmoid(label, name='sigmoid')
        with tf.Session() as sess:
            label = sess.run(label)
        label = np.array(label)
        label = label.reshape((self.conf.im_size, self.conf.im_size, 2))
        return label

    def predicts(self, threshold = 0.9):
        model_path = "LightCNN2/models_rotate_contrast/"
        self.load_graph_step(model_path, 3000)
        standard = self.images/255 - 0.5 #tf.image.per_image_standardization(self.images)

        print(time.clock())
        for n in range(0, 200):
            img = os.path.join('data/train/Images/2020/', str(n) + '.jpg')
            img = cv2.imread(img)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            _, (well_x, well_y, _), im_well = well_detection(img, gray)
            im_well = cv2.cvtColor(im_well, cv2.COLOR_BGR2GRAY)
            x_min = int(well_x - 240 / 2)
            x_max = int(well_x + 240 / 2)
            y_min = int(well_y - 240 / 2)
            y_max = int(well_y + 240 / 2)
            im_block = im_well[y_min:y_max, x_min:x_max]
            img = np.array(im_block, dtype=np.float32)
            print("SHAPE-------------", img.shape)
            img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

            img = img/255 - 0.5
            img = np.reshape(img, (1, self.conf.im_size, self.conf.im_size, 1))

            label = self.sess.run([self.output],
                                  feed_dict={
                                      self.input: img
                                  })
            label = np.squeeze(label)
            #print(label, label.shape)
            label = tf.convert_to_tensor(label)
            label = tf.nn.sigmoid(label, name='sigmoid')
            with tf.Session() as sess:
                label = sess.run(label)
            label = np.array(label)
            label = label.reshape((self.conf.im_size, self.conf.im_size, 2))
            #label = np.argmax(label, axis=2)
            label[label > threshold] = 255
            label[label <= threshold] = 0
            #label = label[:, :, 1] * 255
            #cv2.imshow("label", label[:, :, 1])
            #cv2.waitKey(0)
            print(n)
            print(time.clock())
            #im = Image.fromarray(label.astype('uint8'))
            #im.save(os.path.join('data/render_test/predict/', str(n) + '.png'))

    def load_graph_frozen(self, model_path):
        '''
        Lode trained model.
        '''
        #print('Loading model...')
        tf.reset_default_graph()
        self.graph = tf.Graph()

        with tf.gfile.GFile(model_path, 'rb') as f:
            graph_def = tf.GraphDef()
            graph_def.ParseFromString(f.read())

        #print('Check out the input placeholders:')
        nodes = [n.name + ' => ' + n.op for n in graph_def.node if n.op in ('Placeholder')]
        #for node in nodes:
        #print(node)

        with self.graph.as_default():
            # Define input tensor
            self.input = tf.placeholder(np.float32, shape=[None, 240, 240, 1], name='x')
            tf.import_graph_def(graph_def, {'x': self.input})



        #print('Model loading complete!')

        # Get layer names
        layers = [op.name for op in self.graph.get_operations()]
        """
        for layer in layers:
            print(layer)

        
        # Check out the weights of the nodes
        weight_nodes = [n for n in graph_def.node if n.op == 'Const']
        for n in weight_nodes:
            print("Name of the node - %s" % n.name)
            # print("Value - " )
            # print(tensor_util.MakeNdarray(n.attr['value'].tensor))
        """

        # In this version, tf.InteractiveSession and tf.Session could be used interchangeably.
        # self.sess = tf.InteractiveSession(graph = self.graph)
        self.output = self.graph.get_tensor_by_name("import/cnn/output:0")
        self.sess = tf.Session(graph=self.graph)
        self.graph.finalize()

    def load_graph_step(self, model_path, steps):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_path + "UNet.ckpt-" + str(steps) + ".meta")
        print(model_path + "UNet.ckpt-" + str(steps) + ".meta")
        saver.restore(self.sess, tf.train.latest_checkpoint(model_path)) # add the latest model, not current model

        self.graph = tf.get_default_graph()

        #print(self.graph.get_operations())
        self.input = self.graph.get_tensor_by_name("x:0")

        self.output = self.graph.get_tensor_by_name("cnn/output:0")

    def load_graph(self, model_path, ckpt_path):
        self.sess = tf.Session()
        saver = tf.train.import_meta_graph(model_path)
        print(model_path, ckpt_path)
        saver.restore(self.sess, tf.train.load_checkpoint(ckpt_path))

        self.graph = tf.get_default_graph()

        #print(self.graph.get_operations())
        self.input = self.graph.get_tensor_by_name("x:0")

        self.output = self.graph.get_tensor_by_name("cnn/output:0")

    