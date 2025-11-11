#!/bin/bash

set -e

for planner in "scorpion" "downward-numeric" "symk"; do
    cd planners/$planner
    echo "*******************************************************************************"
    echo "Building $planner"
    echo "*******************************************************************************"
    sh build.sh
    cd -
done
