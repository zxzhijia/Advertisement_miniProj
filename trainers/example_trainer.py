from base.base_train import BaseTrain
from tqdm import tqdm
import numpy as np


class ExampleTrainer(BaseTrain):
    def __init__(self, sess, model, data, config, logger):
        super(ExampleTrainer, self).__init__(sess, model, data, config, logger)

    def train_epoch(self):
        loop = tqdm(range(self.config.num_iter_per_epoch))
        losses = []
        accs = []
        rmse_tot = []
        for _ in loop:
            loss, rmse = self.train_step()
            losses.append(loss)
            rmse_tot.append(rmse)
        loss = np.mean(losses)
        rmse = np.mean(rmse_tot)
        val_loss, rmse_val = self.val_step()

        cur_it = self.model.global_step_tensor.eval(self.sess)
        summaries_dict = {
            'loss': loss,
            'val_loss': val_loss,
            'rmse': rmse,
            'rmse_val': rmse_val
        }
        self.logger.summarize(cur_it, summaries_dict=summaries_dict)
        self.model.save(self.sess)


    def train_step(self):
        batch_x, batch_y = next(self.data.next_batch(self.config.batch_size))
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: True}
        _, loss, rmse = self.sess.run([self.model.train_step, self.model.meansq, self.model.rmse_clip],
                                     feed_dict=feed_dict)
        return loss, rmse

    def val_step(self):
        batch_x, batch_y = self.data.users_items_matrix_train_average, self.data.users_items_matrix_validate
        feed_dict = {self.model.x: batch_x, self.model.y: batch_y, self.model.is_training: False}
        loss, rmse = self.sess.run([self.model.meansq, self.model.rmse_clip],
                                     feed_dict=feed_dict)
        return loss, rmse
