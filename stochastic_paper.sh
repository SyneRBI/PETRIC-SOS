#! /bin/bash
source ~/.bashrc
set -ex

# for f in "SGFunction" "SAGFunction" "SAGAFunction" "SVRGFunction" "LSVRGFunction" "SOS-SAGAfinal2";
for f in "SOS-SAGAfinal2";
do
  echo "Using $f"
  if [ $f == "SOS-SAGAfinal2" ]; then
    # echo ">>> ln -s SOS_SAGAfinal2.py main.py"
    rm main.py
    ln -s SOS_SAGAfinal2.py main.py
  else
    # echo ">>> ln -s SAGA.py main.py"
    rm main.py
    ln -s SAGA.py main.py
  fi
  # for num_subsets in "21" ;
  for num_subsets in "21" "42" "63"; 
  do
    #  --log DEBUG
    STOCHASTIC_FUNCTION=$f NUM_SUBSETS=$num_subsets python petric.py
  done
done
