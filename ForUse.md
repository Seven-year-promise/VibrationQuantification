### 1. Install visual studio code (VS code) with this [link](https://code.visualstudio.com/download)

### 2. Open VS code and install the docker extensions (click the fifth button on the left, called `Extensions`, and search `docker`, click the first one and install on the right)

### 3. Put the data to `./VibrationData`, Open a `terminal` from the top of VS code and run the following codes one by one (remeber to change the path for the second code)

```
docker load -i vibration0_2.tar`
docker run -it -v {your absolute path please}/VibrationResponse:/home/data vibration0_2:no_code
conda activate vibration
python Quantification.py
```

###  The results will be at `./QuantificationResults`
### 4. exit

```
exit
```
 
