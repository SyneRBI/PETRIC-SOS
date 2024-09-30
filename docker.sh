# 2. mount `.` to container `/workdir`:
docker run --rm -it --gpus all -p 6006:6006 \
  -v /path/to/data:/mnt/share/petric:ro \
  -v .:/workdir -w /workdir synerbi/sirf:edge-gpu /bin/bash
# 3. optionally, conda/pip/apt install environment.yml/requirements.txt/apt.txt
# 4. install metrics & run your submission
pip install git+https://github.com/TomographicImaging/Hackathon-000-Stochastic-QualityMetrics
python petric.py &
# 5. optionally, serve logs at <http://localhost:6006>
tensorboard --bind_all --port 6006 --logdir ./output