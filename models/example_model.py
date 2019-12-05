from base.base_model import BaseModel
import tensorflow as tf


class ExampleModel(BaseModel):
    def __init__(self, config):
        super(ExampleModel, self).__init__(config)
        self.build_model()
        self.init_saver()

    def build_model(self):
        self.is_training = tf.compat.v1.placeholder(tf.bool)

        self.x = tf.compat.v1.placeholder(tf.float32, shape=[None] + self.config.state_size)
        self.y = tf.compat.v1.placeholder(tf.float32, shape=[None, 3706])

        # network architecture
        d1 = tf.compat.v1.layers.dense(self.x, 512, activation=tf.nn.relu, name="dense1")
        d2 = tf.compat.v1.layers.dense(d1, 3706, name="dense2")

        with tf.name_scope("loss"):
            self.meansq = tf.reduce_mean(tf.square(self.y - d2))
            update_ops = tf.compat.v1.get_collection(tf.compat.v1.GraphKeys.UPDATE_OPS)
            with tf.control_dependencies(update_ops):
                self.train_step = tf.compat.v1.train.AdamOptimizer(self.config.learning_rate).minimize(self.meansq,
                                                                                         global_step=self.global_step_tensor)

            correct_prediction = tf.equal(tf.argmax(d2, 1), tf.argmax(self.y, 1))
            self.accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))


    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.compat.v1.train.Saver(max_to_keep=self.config.max_to_keep)

