#! /bin/bash
set -ex

for f in "SGFunction" "SAGFunction" "SAGAFunction" "SVRGFunction" "LSVRGFunction";
do
  for num_subsets in "21" "42" "63"; 
  do
    STOCHASTIC_FUNCTION=$f NUM_SUBSETS=$num_subsets python petric.py
  done
done
