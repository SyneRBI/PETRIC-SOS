# 1. git clone & cd to your submission repository
# 2. mount `.` to container `/workdir`:
docker run --rm -it --gpus all -p 6006:6006 \
  -v /path/to/data:/mnt/share/petric:ro \
  -v .:/workdir -w /workdir synerbi/sirf:edge-gpu /bin/bash
# 3. install metrics & GPU libraries
conda install monai tensorboard tensorboardx jupytext cudatoolkit=11.8
pip uninstall torch # monai installs pytorch (CPU), so remove it
pip install tensorflow[and-cuda]==2.14  # last to support cu118
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install git+https://github.com/TomographicImaging/Hackathon-000-Stochastic-QualityMetrics
# 4. optionally, conda/pip/apt install environment.yml/requirements.txt/apt.txt
# 5. run your submission
python petric.py &
# 6. optionally, serve logs at <http://localhost:6006>
tensorboard --bind_all --port 6006 --logdir ./output