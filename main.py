from inbetween import DraftRefine
import argparse
import os
import yaml
from pprint import pprint
from easydict import EasyDict



def parse_args():
    parser = argparse.ArgumentParser(
        description='Anime segment matching')
    parser.add_argument('--config', default='')
    # exclusive arguments
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument('--train', action='store_true')
    group.add_argument('--eval', action='store_true')
    group.add_argument('--gen', action='store_true')


    return parser.parse_args()


def main():
    # parse arguments and load config
    args = parse_args()
    with open(args.config) as f:
        config = yaml.load(f)

    for k, v in vars(args).items():
        config[k] = v
    pprint(config)

    config = EasyDict(config)
    agent = DraftRefine(config)
    print(config)

    if args.train:
        agent.train()
    elif args.eval:
        agent.eval()
    elif args.gen:
        agent.gen()


if __name__ == '__main__':
    main()
