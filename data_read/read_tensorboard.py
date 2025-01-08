# conda create -n stochastic_paper tensorboard numpy matplotlib-base ipykernels -c conda-forge
# Data in C:\Users\ofn77899\Dev\StochasticFrameworkPaper\data
# https://www.tensorflow.org/install/pip#windows-native
import tensorflow as tf
from tensorflow.python.summary.summary_iterator import summary_iterator
import os

# https://stackoverflow.com/questions/37304461/tensorflow-importing-data-from-a-tensorboard-tfevent-file


# docker run -it --rm -v ~/workdir/PETRIC-SOS/:/root/workdir -w /root/workdir tensorflow/tensorflow bash

def read_tensorboard_events(path_to_events_file: os.path.abspath, tags: list):
    results = {}
    for i,tag in enumerate(tags):
        results[tags[i]] = []
        
    for e in summary_iterator(path_to_events_file):
        for v in e.summary.value:
            if v.tag in tags:
                results[v.tag].append(v.simple_value)
    return results

def are_there_results(results: dict, tags:list):
    data_len = []
    for k,v in results.items():
        if k not in tags:
            raise ValueError(f"Tag {k} not found in expected tags")
        data_len.append(len(v))
    
    return sum(data_len) > 0

tags = [
"RMSE_whole_object",
"RMSE_background",
"AEM_VOI_sphere1",
"AEM_VOI_sphere3",
"AEM_VOI_sphere5",
]

if __name__ == "__main__":

    path_to_events_file = os.path.abspath("../output_test1/SAGA/42/mMR_NEMA/events.out.tfevents.1734352210.57d4f72f70fb")
    path_to_events_file = os.path.abspath("../output/SAGA/42/mMR_NEMA/events.out.tfevents.1734550654.57d4f72f70fb")
    # results = {
    #     tags[0]: [],
    #     tags[1]: [],
    #     tags[2]: [],
    #     tags[3]: [],
    #     tags[4]: [],
    # }
    # for e in summary_iterator(path_to_events_file):
    #     for v in e.summary.value:
    #         if v.tag in tags:
    #             results[v.tag].append(v.simple_value)

    results = read_tensorboard_events(path_to_events_file, tags)

    print (results)
    for k,v in results.items():
        print (k, len(v))

    import numpy as np
    data_passes = np.load(os.path.abspath("../output/SAGA/42/mMR_NEMA/data_passes.npy"))
    print ("len(data_passes)", len(data_passes))
    print (data_passes)
    print(data_passes[20::21])
    print([data_passes[21-1], data_passes[21*2 -1]])
    # What to plot
    # 1) Objective value vs data_passes
    # 2) Metrics vs data_passes
    # 2.1) whole object RMSE, 
    # 2.2) background 
    # 2.3) ROIs 
