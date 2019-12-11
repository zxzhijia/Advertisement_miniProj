from base.base_model import BaseModel
import tensorflow as tf
import numpy as np
import os

class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.compat.v1.placeholder(tf.bool)

        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3952])

        # network architecture
        d1 = tf.compat.v1.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")

        self.d2 = tf.compat.v1.layers.dense(d1, 3952, name="dense2")

        l2_loss = tf.nn.l2_loss(tf.compat.v1.trainable_variables()[0]) + tf.nn.l2_loss(tf.compat.v1.trainable_variables()[1]) +\
                  tf.nn.l2_loss(tf.compat.v1.trainable_variables()[2]) +\
                  tf.nn.l2_loss(tf.compat.v1.trainable_variables()[3])

        with tf.name_scope("loss"):
            # self.meansq = tf.reduce_mean(tf.square(self.y - self.d2))
            self.meansq = self.masked_mse(self.y, self.d2)
            self.meansq += 0.0005*l2_loss
            self.rmse_clip = self.masked_rmse_clip(self.y, self.d2)
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.compat.v1.train.AdamOptimizer(self.config.learning_rate).minimize(self.meansq,
                                                                                         global_step=self.global_step_tensor)

            correct_prediction = tf.equal(tf.argmax(self.d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

    def masked_mse(self, y_true, y_pred):
        # masked function
        mask_true = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        # masked squared error
        masked_squared_error = tf.square(mask_true * (y_true - y_pred))
        masked_mse = tf.math.reduce_sum(masked_squared_error, axis=-1) / tf.maximum(tf.math.reduce_sum(mask_true, axis=-1), 1)
        return tf.math.reduce_mean(masked_mse)

    def masked_rmse_clip(self, y_true, y_pred):
        # masked function
        mask_true = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1, clip_value_max=5)
        # masked squared error
        masked_squared_error = tf.square(mask_true * (y_true - y_pred))
        masked_mse = tf.sqrt(tf.math.reduce_sum(masked_squared_error, axis=-1) / tf.maximum(tf.math.reduce_sum(mask_true, axis=-1), 1))
        return tf.math.reduce_mean(masked_mse)

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)

