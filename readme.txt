Please install conda first (https://conda.io/en/latest/miniconda.html#windows-installers) and ensure it is successful. 
Then run the following to create an environment:
############
conda create -n ML2  -c conda-forge -c pytorch python=3.9 pytorch=1.9 tensorflow=2.6 cudnn=8 cudatoolkit=11 scipy pandas openpyxl xlrd jupyterlab jupyter_contrib_nbextensions
conda activate ML2
pip install  tensorflow-gpu
cd [your project dir]
jupyter notebook
###################
Open the DeepKme.ipynb in your browser where your jupyter notebook works. 
Then your can run each cell to replicate the paper's work.