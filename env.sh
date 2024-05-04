conda create -y --prefix=./.conda_env python=3.10 && \
conda activate ./.conda_env && \
conda install -y pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia  && \
conda install -y xformers -c xformers  && \
pip install easydict ipython tqdm  && \
echo "Conda environment set up successfully."
