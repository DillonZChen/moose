#!/bin/bash

cd src/search/bliss-0.73
make
cd -

python3 build.py release64 -j8
