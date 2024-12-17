# conda create -n stochastic_paper tensorboard numpy matplotlib-base ipykernels -c conda-forge
# Data in C:\Users\ofn77899\Dev\StochasticFrameworkPaper\data
# https://www.tensorflow.org/install/pip#windows-native
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import os

# https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file


# docker run -it --rm -v ~/workdir/PETRIC-SOS/:/root/workdir -w /root/workdir tensorflow/tensorflow bash
tags = [
"RMSE_whole_object",
"RMSE_background",
"AEM_VOI_sphere1",
"AEM_VOI_sphere3",
"AEM_VOI_sphere5",
]

results = {
    tags[0]: [],
    tags[1]: [],
    tags[2]: [],
    tags[3]: [],
    tags[4]: [],
}

path_to_events_file = os.path.abspath("../output_test1/SAGA/42/mMR_NEMA/events.out.tfevents.1734352210.57d4f72f70fb")
path_to_events_file = os.path.abspath("../output/SAGA/63/mMR_NEMA/events.out.tfevents.1734438645.57d4f72f70fb")
for e in summary_iterator(path_to_events_file):
    for v in e.summary.value:
        if v.tag in tags:
            results[v.tag].append(v.simple_value)

print (results)

import numpy as np
data_passes = np.load(os.path.abspath("../output/SAG/21/mMR_NEMA/data_passes.npy"))
print (len(data_passes))
print (data_passes)
print(data_passes[20::21])
print([data_passes[21-1], data_passes[21*2 -1]])
# What to plot
# 1) Objective value vs data_passes
# 2) Metrics vs data_passes
# 2.1) whole object RMSE, 
# 2.2) background 
# 2.3) ROIs 
