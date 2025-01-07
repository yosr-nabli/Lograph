### How to run the experiment
* Edit `config.py` which stores the basic settings of experimental environment, including data paths, model paths, training schemes, hyper-parameters, etc.
* Download public log datasets from [LogHub](https://github.com/logpai/loghub) and put into the data folder.
* Run `main.py` to start training the model.
* Edit the last line of `main.py` to run experiments on different datasets:
 `run_experiment(<DATASET_NAME:String>, <GROUPING_STRATEGY:String>)`
