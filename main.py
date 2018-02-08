import pickle
import os
import argparse
from agent import Agent
from pprint import PrettyPrinter

def main():
    args = get_args()
    
    if not os.path.exists(args.save_dir + 'data/'):
        os.mkdir(args.save_dir + 'data/')
    
    agent = Agent(args.run_dir, args.save_dir)
    
    agent.train()

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--run_dir', help='run directory', default='./')
    parser.add_argument('--save_dir', help='save directory', default='./')
    return parser.parse_args()

if __name__ == '__main__':
    main()