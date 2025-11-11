#!/bin/bash

git submodule update --init --recursive

pip install -r requirements.txt

cd ext
sh build_planners.sh
cd nraxioms2disjpres
pip install .
cd ..
cd ..
