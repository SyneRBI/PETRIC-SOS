#! /bin/bash
source ~/.bashrc
set -ex

for f in "SGFunction" "SAGFunction" "SAGAFunction" "SVRGFunction" "LSVRGFunction";
# for f in "SGFunction";
do
  # for num_subsets in "21" ;
  for num_subsets in "21" "42" "63"; 
  do
    #  --log DEBUG
    STOCHASTIC_FUNCTION=$f NUM_SUBSETS=$num_subsets python petric.py
  done
done
