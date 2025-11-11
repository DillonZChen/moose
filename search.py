#!/usr/bin/env python3

from moose import execute_search
from moose.options import get_search_parser

if __name__ == "__main__":
    parser = get_search_parser()
    opts = parser.parse_args()
    execute_search(opts)
