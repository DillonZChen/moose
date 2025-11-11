#!/usr/bin/env python3

from moose import execute_policy
from moose.options import get_policy_parser

if __name__ == "__main__":
    parser = get_policy_parser()
    opts = parser.parse_args()
    execute_policy(opts)
