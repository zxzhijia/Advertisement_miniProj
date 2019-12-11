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
        self.sample_user_predict, self.rmse_clip = self.sess.run([self.model.d4, self.model.rmse_clip], feed_dict={self.model.x: self.sample_user,
                                                                                                                   self.model.y: self.sample_test,
                                                                                                                   self.model.is_training: False})

        print("The predicted rating is {}".format(self.sample_user_predict))
        print("The default rating is {}".format(self.sample_user))
        print("The clip rmse is {}".format(self.rmse_clip))

    def analysis_results(self):
        for example_id in range(1, 10):
            plt.figure()
            plt.plot(self.sample_test[example_id, np.where(self.sample_test[example_id, :]>0)[0]], '.g')
            plt.plot(self.sample_user_predict[example_id, np.where(self.sample_test[example_id, :]>0)[0]], '.r')
            plt.show()
            # plt.savefig('../figures/prediction_truth.png')

