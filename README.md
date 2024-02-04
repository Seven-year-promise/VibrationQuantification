## Download and install the miniconda https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
## Create a new environment

- open a cmd
- conda create -n vibration python==3.7.6
- conda activate vibration
  
## Install the following libraries:

- pip install tensorflow-gpu==1.15.0
- pip install torch=1.4.0
- pip install scikit-learn==0.23.1
- pip install scipy==1.6.0
- pip install seaborn==0.12.2
- pip install pyqt5-sip==4.19.18

## How to use the repostory

- DO the quantification by

    `python quantification.py`
    
- if any libraries (modules) are missing, install them by `pip install XXXXX`
