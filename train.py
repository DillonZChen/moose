#!/usr/bin/env python3

from moose import train
from moose.options import get_train_parser

if __name__ == "__main__":
    parser = get_train_parser()
    opts = parser.parse_args()
    train(opts)
