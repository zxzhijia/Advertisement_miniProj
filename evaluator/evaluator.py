from base.base_evaluate import BaseEvaluator
import tensorflow as tf
from matplotlib import pyplot as plt
import numpy as np

class Evaluator(BaseEvaluator):
    def __init__(self, sess, model, data, config, logger):
        super(Evaluator, self).__init__(sess, model, data, config, logger)

    def evaluate(self):
        self.sample_user = self.data.input.iloc[99:100, :]
        self.sample_user_predict = self.sess.run([self.model.d2], feed_dict={self.model.x: self.sample_user})
        print("The predicted rating is {}".format(self.sample_user_predict))
        print("The default rating is {}".format(self.sample_user))


    def analysis_results(self):
        plt.figure()
        plt.plot(self.sample_user.values[0, :])
        plt.plot(self.sample_user_predict[0][0, :])
        plt.show()
        plt.savefig('../figures/prediction_truth.png')