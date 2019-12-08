from base.base_evaluate import BaseEvaluator
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class Evaluator(BaseEvaluator):
    def __init__(self, sess, model, data, config, logger):
        super(Evaluator, self).__init__(sess, model, data, config, logger)

    def evaluate(self):
        self.sample_test = self.data.users_items_matrix_test
        self.sample_user = self.data.users_items_matrix_train_average
        self.sample_user_predict = self.sess.run(self.model.d2, feed_dict={self.model.x: self.sample_user})
        self.test_loss = self.masked_rmse_clip(self.data.users_items_matrix_test, self.sample_user_predict)

        print("The predicted rating is {}".format(self.sample_user_predict))
        print("The default rating is {}".format(self.sample_user))


    def analysis_results(self):
        for example_id in range(1, 10):
            plt.figure()
            plt.plot(self.sample_test[example_id, np.where(self.data.users_items_matrix_test[example_id, :]>0)[0]], '.g')
            plt.plot(self.sample_user_predict[example_id, np.where(self.data.users_items_matrix_test[example_id, :]>0)[0]], '.r')
            plt.show()
            # plt.savefig('../figures/prediction_truth.png')

    def masked_rmse_clip(self, y_true, y_pred):
        # masked function
        mask_true = tf.cast(tf.math.not_equal(y_true, 0), tf.float32)
        y_pred = tf.clip_by_value(y_pred, clip_value_min=1, clip_value_max=5)
        # masked squared error
        masked_squared_error = tf.square(mask_true * (y_true - y_pred))
        masked_mse = tf.sqrt(tf.math.reduce_sum(masked_squared_error, axis=-1) / tf.maximum(tf.math.reduce_sum(mask_true, axis=-1), 1))
        return masked_mse