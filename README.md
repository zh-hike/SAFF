# Self-Attention-Based Deep Feature Fusion for Remote Sensing Scene Classification
Use vgg16 and SAFF for small sample classification

## Introduction
* Extract dataset features using pretrained vgg16
![model](imgs/model.png)

* SAFF converts features into 1D tensor

![saff](imgs/saff.png)

## Environmental preparation
```bash
conda create -n zh python=3.9
conda activate zh
python3 -m pip install --upgrade pip
pip3 install -r requirements.txt
```

## Run
If your dataset is at path  */hy-tmp/data*
Suppose you want to train on the *UC* dataset.

* Feature extraction
```
python run.py 
--data_path /hy-tmp/data 
--extract
--dataset UC
```

* Train & verify

```
python run.py 
--data_path /hy-tmp/data
--train
--dataset UC
--ratio 0.8
```

## Experimental results

| dataset | train_ratio |  acc  |
|:-------:|:-----------:|:-----:|
|  NWPU   |     0.1     | 66.49 |
|  NWPU   |     0.2     | 73.13 |
|   UC    |     0.8     | 92.5  |
|   SAR   |     0.8     | 89.8  |
