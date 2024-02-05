## Download and install the miniconda https://docs.conda.io/projects/miniconda/en/latest/miniconda-install.html
## Create a new environment

- Firstly, open a cmd
- `conda create -n vibration python==3.7.0`
- `conda activate vibration`
  
## Install the following packages:

- `pip install tensorflow==1.15.0`
- `pip install matplotlib==3.2.1`
- `pip install scikit-image==0.18.1`
- `pip install scikit-learn==0.23.1`
- `pip install pandas==1.0.4`
- `pip install opencv-python==4.5.3.56`

For any other packages needed, you can refer to https://github.com/Seven-year-promise/VibrationQuantification/blob/master/requirements, and you can also install by simply running
`pip install -r requirements.txt`.

## How to use the repostory

- DO the quantification by

    `python Quantification.py`
    
- if any libraries (modules) are missing, install them by `pip install XXXXX`
