'''
doc:
    https://docs.python.org/3/library/argparse.html
'''
import argparse

parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('-d', '--data-dir', type=str, default='data_', help='data directory')
parser.add_argument('-b',
                    '--block-size',
                    type=int,
                    default=65536,
                    help='the maximum number of tokens to be processed per block')
parser.add_argument('--cuda', action='store_true')

args = parser.parse_args()