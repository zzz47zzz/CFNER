# Distilling Causal Effect from Miscellaneous Other-Class for Continual Named Entity Recognition (EMNLP2022)

## Overview of the directory
- config/ : the hyper-parameters for each models
- datasets/ : the datasets
- src/ : the code
- main_CL.py : the main file for running commands
```
.
├── config
│   ├── conll2003
│   ├── ontonotes5
│   ├── i2b2
│   └── default.yaml
├── datasets
│   └── NER_data
│       ├── conll2003
│       ├── i2b2
│       └── ontonotes5
├── main_CL.py
└── src
    ├── config.py
    ├── dataloader.py
    ├── model.py
    ├── trainer.py
    ├── utils_log.py
    ├── utils_plot.py
    └── utils.py
```

### Step 1: Prepare your data
Download data and preprocess them in the following format (word + \t + NER_tag):
```
SOCCER	O
-	O
JAPAN	B-location
GET	O
LUCKY	O
WIN	O
,	O
CHINA	B-person
IN	O
SURPRISE	O
DEFEAT	O
.	O
```
Then, save the training/testing/developing set to a txt file named train.txt/test.txt/dev.txt and move them to the corresponding directory in ./datasets/NER_data/{dataset_name}/

## Step 2: Split the dataset for continual learning
Take CoNLL2003 as an example:
- modify the ./src/dataloader.py
```
if __name__ == "__main__":
   spilt_dataset(['datasets/NER_data/conll2003'], 'train', domain2entity['conll2003'], 1, 1, 'BIO')
   # the parameters nb_class_fg=1 and nb_class_pg=1 represent that the model learns 1 entity in the first CL step and learns 1 entity in the following CL steps.
```

- run the following command:
```
python ./src/dataloader.py
```
Then, the split dataset will be stored in **train_fg_1_pg_1.pth**.
Note that only training set needs to be split! 

## Step 3: Run main_CL.py
Specify your configurations (e.g., ./config/i2b2/fg_8_pg_2/i2b2_ours.yaml) and run the following command 
```
python3 main_CL.py --exp_name {your_experiment_name} --exp_id {your_experiment_id} --cfg {your_configuration}
```

If you find the code useful, please cite the following work
```
@
```

