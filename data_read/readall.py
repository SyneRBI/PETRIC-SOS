import glob
import os
import numpy as np
from read_tensorboard import read_tensorboard_events, tags, are_there_results

algorithms = ['LSVRG', 'SAG', 'SAGA', 'SG', 'SOS-SAGAfinal2', 'SVRG']
subsets = [21,42,63]
dataset = ['mMR_NEMA']

allresults = {}
# path to the result directory
basepath = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "output"))
print (basepath)
for alg in algorithms:
    for sub in subsets:
        events_path = os.path.join(basepath, alg, str(sub), dataset[0])
        print (f"Checking {events_path}")
        results = []
        for f in glob.glob(os.path.join(events_path , "*")):
            # print (f"Checking {f}")
            if "events.out.tfevents" in f:
                event_file = f
                # print (f"Found {f}")
                print (f"Reading {event_file}")
                res = read_tensorboard_events(event_file, tags)
                if are_there_results(res, tags):
                    print(f"Found results in event file {event_file}")
                    results.append(res)
                else:
                    print (f"No results in event file {event_file}")

        if len(results) != 1:
            raise ValueError(f"{alg} {sub} Expected 1 result, found {len(results)}")
        # add the objective values and data passes to the results
        data_passes = np.load(os.path.join(events_path, "data_passes.npy"))
        results[0]["data_passes"] = data_passes
        import csv
        csvfile = os.path.join(events_path, "objectives.csv")
        iteration_num = []
        objective = []
        with open(csvfile, mode='r') as csvf:
            reader = csv.reader(csvf, delimiter=",")
            try:
                for i,row in enumerate(reader):
                    if i>0:
                        iteration_num.append(int(row[0]))
                        objective.append(float(row[1]))
            except csv.Error as e:
                sys.exit('file {}, line {}: {}'.format(filename, reader.line_num, e))
        
        results[0]["iteration_num"] = iteration_num
        results[0]["objective"] = objective

        allresults[f"{alg}_{dataset[0]}_{sub}"] = results[0]

        
for k,v in allresults.items():
    print (k)

import pickle
pickle.dump(allresults, open("allresults.pkl", "wb"))