import os
import yaml
import argparse
from pprint import pprint

from utils.video_trainer import VideoTrainer
from utils.fake_trainer import FakeTrainer
from utils.tester import Tester


def main(config_path):
    # Read training configuration
    with open(config_path) as f:
        config = yaml.full_load(f)
        pprint(config)

    mode = config['main']['mode']
    phase = config['main']['phase']

    if mode == 'train':
        if phase == 1:
            trainer = VideoTrainer(config)
        elif phase == 2:
            trainer = FakeTrainer(config)
        else:
            raise NotImplementedError
        trainer.train()
    elif mode == 'test':
        tester = Tester(config)
        tester.test()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-c",
                        "--config",
                        required=True,
                        help="path to configuration file")
    args = vars(parser.parse_args())
    main(args['config'])
