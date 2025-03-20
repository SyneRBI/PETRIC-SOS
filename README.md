# Running the reconstructions

## Software required

The software required is availble as docker image `synerbi/sirf:edge-gpu`, see https://github.com/SyneRBI/SIRF-SuperBuild/blob/master/docker/README.md for further details.

To run the code for the various algorithms/configuration one needs 
- this repository on the `nema_paper` branch
- to start the docker container.
- to launch the computation by launching the file `stochastic_paper.sh` attaching to the docker container

### Launching the docker container
```bash
docker run -d --gpus all -p 6006:6006 \
  -v /path/to/data:/mnt/share/petric:ro \
  -v .:/workdir -w /workdir synerbi/sirf:edge-gpu
```

### Launching computation
You can start the computation by launching the file `stochastic_paper.sh` attaching to the docker container as

```bash
nohup docker exec <container-id> bash stochastic_paper.sh > sos_submission_lowcounts.log &
```

## Configuration

In practice one needs to actually run `petric.py` to launch the computation, which in turns executes the file `main.py`. To be able to switch from the `SOS-SAGA_final2` I created a file that creates a symbolic link to the `SAGA.py` and `SOS_SAGAfinal2.py` to `main.py`.
It also sets the environment variables `STOCHASTIC_FUNCTION` and `NUM_SUBSETS` which set additional parameters.

In particular `STOCHASTIC_FUNCTION` is only used by `SAGA.py` to instantiate the appropriate stochastic estimator.