# FraudGT: A Simple, Effective, and Efficient Graph Transformer for Financial Fraud Detection
![framework](imgs/framework.png)
This repository holds the code for FraudGT framework.

## Environment Setup
You can create a conda environment to easily run the code. For example, we can create a virtual environment named `fraudGT`:
```
conda create -n fraudGT python=3.9 -y
conda activate fraudGT
```
Install the required packages using the following commands:
```
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
conda install pyg -c pyg
pip install -r requirements.txt
```

## Run the Code
You will need to firstly specify the dataset path (`./data` in this example) and log location (`./results` in this example) by editing the config file provided under `./configs/{dataset_name}/`. An example configuration is
```
......
out_dir: ./results
dataset:
  dir: ./data
......
```
Download and unzip the [Anti-Money Laundering dataset](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) into your specified dataset path (for example, put the unzipped `HI-small.csv` into `./data`).
Dataset will be automatically processed at the first run.

![experiments](imgs/experiment.png)

For convenience, a script file is created to run the experiment with specified configuration. For instance, you can edit and run the `interactive_run.sh` to start the experiment.
```
cd FraudGT
chmox +x ./run/interactive_run.sh
./run/interactive_run.sh
```

