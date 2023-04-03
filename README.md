# IGCCF
![](IGCCF_architecture.png)

Paper Link: https://link.springer.com/chapter/10.1007/978-3-031-28244-7_16

### Requirements
To install all the required packages using the following command:
	
	$ pip install -r requirements.txt

### Datasets
To run the provided code is necessary to download and pre-process the datasets. To download and  preprocess one dataset run command:

    $ python datasets/create_split.py --dataset="Movielens1M"
    
The available Datasets are: **LastFM, Movielens1M, AmazonElectronics, Gowalla**.
*Note:* The dataset will be preprocessed using the exact random seeds used to obtain the results presented in the paper to let the experiments be completely reproducible.

### Train models 
To train a `model` using the validation set, run the following command with the proper args:

    $ python models/tensorflow/train_model/model/train_model.py --epochs="1000" --val_every="10" ...

Available models are: **matrix_factorization_bpr, fism, pureSVD, lightgcn, ngcf, igccf**

### Evaluate models
To evaluate the performance of a model onto the test set run:

    $ python igccf_experiments/eval_trainval.py --algorithm="igccf"

### Experiments
All the experiments file are stored inside `igccf_experiments` folder
##### Ablation studies
* igccf_experiments/ablation_studies/`convolution_depth.py`
* igccf_experiments/ablation_studies/`top_k_pruning.py`
* igccf_experiments/ablation_studies/`user_profile_dropout.py`

Files to plot the results are store inside `igccf_experiments/ablation_studies/plot`
##### Inductive performance 
* igccf_experiments/`unseen_users_performance.py`

File to plot the results `igccf_experiments/plot/plot_inductive_user_performance.py`

##### Robustness Inductive performance
* igccf_experiments/plot/plot_robustness_inductive_performance.py
