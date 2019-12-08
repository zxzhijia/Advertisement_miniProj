import tensorflow as tf

from data_loader.data_generator_v2 import DataGenerator
from models.example_model import ExampleModel
from trainers.example_trainer import ExampleTrainer
from evaluator.evaluator import Evaluator
from utils.config import process_config
from utils.dirs import create_dirs
from utils.logger import Logger
from utils.utils import get_args
from sys import exit

tf.compat.v1.disable_eager_execution()

def main():
    # capture the config path from the run arguments
    # then process the json configuration file
    try:
        args = get_args()
        json_file = '../configs/example.json'
        # config = process_config(args.config)
        config = process_config(json_file)

    except:
        print("missing or invalid arguments")
        exit(0)

    # create the experiments dirs
    create_dirs([config.summary_dir, config.checkpoint_dir])
    # create tensorflow session
    sess = tf.compat.v1.Session()
    # create your data generator
    data = DataGenerator(config)
    data.generate_data()
    
    # create an instance of the model you want
    model = ExampleModel(config)
    # create tensorboard logger
    logger = Logger(sess, config)
    # create trainer and pass all the previous components to it
    trainer = ExampleTrainer(sess, model, data, config, logger)
    #load model if exists
    model.load(sess)
    # here you train your model
    trainer.train()
    # here you evaluate your model
    evaluator = Evaluator(trainer.sess, trainer.model, data, config, logger)
    evaluator.evaluate()
    evaluator.analysis_results()



if __name__ == '__main__':
    main()
