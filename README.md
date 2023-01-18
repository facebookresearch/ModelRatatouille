# Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization

Official PyTorch implementation of model ratatouille | [paper](https://arxiv.org/abs/2212.10445)

[Alexandre Ramé](https://alexrame.github.io/), [Kartik Ahuja](https://ahujak.github.io/), [Jianyu Zhang](https://scholar.google.com/citations?user=srn7ay8AAAAJ&hl=en), [Matthieu Cord](http://webia.lip6.fr/~cord/), [Léon Bottou](https://leon.bottou.org/), [David Lopez-Paz](http://lopezpaz.org/)

## TL;DR

We propose a new fine-tuning strategy that improves OOD generalization in computer vision by recycling and averaging weights specialized on diverse auxiliary tasks.

## Abstract

Foundation models are redefining how AI systems are built. Practitioners now follow a standard procedure to build their machine learning solutions: from a pre-trained foundation model, they fine-tune the weights on the target task of interest. Then, the Internet is swarmed by a handful of foundation models fine-tuned on many diverse tasks: these individual fine-tunings exist in isolation without benefiting from each other. In our opinion, this is a missed opportunity, as these specialized models contain rich and diverse features. In this paper, we thus propose model ratatouille, a new strategy to recycle the multiple fine-tunings of the same foundation model on diverse auxiliary tasks. Specifically, we repurpose these auxiliary weights as initializations for multiple parallel fine-tunings on the target task; then, we average all fine-tuned weights to obtain the final model. This recycling strategy aims at maximizing the diversity in weights by leveraging the diversity in auxiliary tasks. Empirically, it improves the state of the art on the reference DomainBed benchmark for out-of-distribution generalization. Looking forward, this work contributes to the emerging paradigm of updatable machine learning where, akin to open-source software development, the community collaborates to reliably update machine learning models.

# Setup

## Codebase and DomainBed

Our code is adapted from the open-source [DomainBed github](https://github.com/facebookresearch/DomainBed/), which is a PyTorch benchmark including datasets and algorithms evaluating OOD generalization. It was introduced in [In Search of Lost Domain Generalization, ICLR 2021](https://openreview.net/forum?id=lQdXeXDoWtI). More specifically, our code extends the [DiWA github](https://github.com/alexrame/diwa), which weight averages the models obtained from the hyperparameter search as a replacement to only selecting one single model: this was motivated and explained in [model soups, ICML 2022](https://arxiv.org/abs/2203.05482) and [DiWA, NeurIPS 2022](https://arxiv.org/abs/2205.09739) papers.

## Packages requirements

* python == 3.7.10
* torch == 1.12.1
* torchvision == 0.13.1
* numpy == 1.21.5

## Datasets

We consider the following [datasets](domainbed/datasets.py):

* VLCS ([Fang et al., 2013](https://openaccess.thecvf.com/content_iccv_2013/papers/Fang_Unbiased_Metric_Learning_2013_ICCV_paper.pdf))
* PACS ([Li et al., 2017](https://arxiv.org/abs/1710.03077))
* OfficeHome ([Venkateswara et al., 2017](https://arxiv.org/abs/1706.07522))
* A TerraIncognita ([Beery et al., 2018](https://arxiv.org/abs/1807.04975)) subset
* DomainNet ([Peng et al., 2019](http://ai.bu.edu/M3SDA/))

You can download the datasets with following command:

```sh
python3 -m domainbed.scripts.download --data_dir ${data_dir}
```

# Ratatouille: procedure details

Our procedure is in three stages.

1. Auxiliary trainings: create a pool of specialized models on various auxiliary tasks.
2. Target trainings: apply the standard hyperparameter search starting from these auxiliary initializations.
3. Weight selection: average the fine-tuned weights.

The different experiments are saved in `${expe_dir}`.

## Building a pool of specialized auxiliary weights

For real-world applications, we envision that specialized weights may be downloaded from collaborative open-source repositories of neural networks. In practice in this github, to populate the folder `${expe_dir}/aux`, we will perform fine-tunings on DomainBed's datasets. Specifically, we use the `sweep` script with either VLCS, PACS, OfficeHome, TerraIncognita or DomainNet as the `${auxiliary_dataset}`.

```sh
mkdir ${expe_dir}/lp # dir containing the linear probe runs
mkdir ${expe_dir}/aux # dir containing the auxiliary runs

for auxiliary_dataset in VLCS PACS OfficeHome TerraIncognita DomainNet
do
python -m domainbed.scripts.sweep launch\
       --data_dir ${data_dir}\
       --dataset ${auxiliary_dataset}\
       --test_env -1\ ## this means that we train on all domains simultaneously: there is no OOD test env for auxiliary trainings.
       --output_dir_lp ${expe_dir}/lp/${auxiliary_dataset}_notest\ ## where the shared linear probe is saved
       --output_dir ${expe_dir}/aux/${auxiliary_dataset}_notest\ ## where the auxiliary hyperparameter sweep is saved
       --n_hparams 4\ ## we only need 4 runs in the hyperparameter search
       --n_trials 1 ## only one data split
done
```

First, if `output_dir_lp` does not exist, we linear probe (lp) the classifier (to prevent [feature distortion](https://openreview.net/forum?id=UYneFzXSJWh)): this classifier initialization will be used in the subsequent runs. Second, we populate `output_dir` with `n_hparams` ERM runs following the hyperparameter distributions from [here](domainbed/hparams_registry.py).

Critically, this procedure is agnostic to the target task, and thus is done only once.

## Fine-tunings on the target task

Now we focus on a given `${target_dataset}`, and one `${test_env}` domain considered as the test domain: other domains are for training. As previously, we leverage the `sweep` script.

```sh
mkdir ${expe_dir}/target # dir containing the target runs
target_dataset=OfficeHome ## or any other DomainBed's dataset
test_env=0 ## or any integer between 0 and 3

python -m domainbed.scripts.sweep launch\
       --data_dir ${data_dir}\
       --dataset ${target_dataset}\
       --test_env ${test_env}\ ## domain not seen during training and kept apart for OOD evaluation
       --output_dir_lp ${expe_dir}/lp/${target_dataset}_test${test_env}\ ## where the shared linear probe is saved
       --output_dir ${expe_dir}/target/${target_dataset}_withaux\ ## where the target hyperparameter sweep is saved
       --aux_dir ${expe_dir}/aux\ ## where the pool of auxiliary weights are saved
       --n_hparams 20\  ## default number of hyperparameters, but 5 already provides good results
       --n_trials 1 ## set to 3 to test different data splits
```

The arg `aux_dir` is the directory containing the different auxiliary runs to initialize the featurizer. Obviously, to prevent any kind of information leakage, in the code we will discard from `aux_dir` the models inter-trained on `${target_dataset}`: in brief, we ensure that `${target_dataset}` $\neq$ `${auxiliary_dataset}`.

## Average the fine-tuned weights

Ratatouille's main theoretical contribution states the linear mode connectivity across models fine-tuned on the target task starting from different initializations. Thus we average the weights obtained from previous sweep.

```sh
python -m domainbed.scripts.inference\
       --data_dir ${data_dir}\
       --dataset ${target_dataset}\
       --test_env ${test_env}\
       --input_dir ${expe_dir}/target/${target_dataset}_withaux\
       --weight_selection uniform\ # or use greedy
       --trial_seed 0
```

If you want to obtain standard deviations on different data splits, set `--n_trials 3` in the sweep command. Then you can specify `trial_seed` to either `0`, `1` or `2`: you can also average all `60` weights from the `3` trials by setting`trial_seed`to`-1`, what we call`uniform`$^\dagger$.


# Baselines

### Inter-training

Inter-training selects the best model based on ID validation accuracy from previous runs. To reproduce the results, call:

````sh
python -m domainbed.scripts.collect_results --input_dir ${expe_dir}/target/${target_dataset}_withaux
````

## Vanilla fine-tuning and Soups/DiWA

You first need to launch a new sweep without specifying `aux_dir`.
```sh
python -m domainbed.scripts.sweep launch\
       ... # same as before
       --output_dir ${expe_dir}/target/${target_dataset}_noaux\ ## change the output dir
       --aux_dir none
```

Then call `collect_results.py` (for vanilla fine-tuning) or `inference.py` (for Soups/DiWA) with `--input_dir ${expe_dir}/target/${target_dataset}_noaux`. In brief, model ratatouille is to inter-training as model soups is to vanilla fine-tuning.

## Fusing

Add `--fusing_range 4` in the previous sweep command to operate linear interpolation at initialization as in [fusing](https://arxiv.org/abs/2204.03044), where rather than selecting one single checkpoint at initialization, they linearly interpolate multiple auxiliary featurizers.

```sh
python -m domainbed.scripts.sweep launch\
       ... # same as before
       --output_dir ${expe_dir}/target/${target_dataset}_withaux_fusing4\ ## change the output dir
       --fusing_range 4 #  The value `4` specifies how the interpolating coefficients are sampled.
```

# Results

Ratatouille sets a new state of the art on DomainBed.

| Algorithm        | Selection | PACS | VLCS | OfficeHome | TerraInc | DomainNet | Avg  |
|---|---|---|---|---|---|---|---|
| Vanilla fine-tuning              | ID val              | 85.5 | 77.5 | 66.5       | 46.1     | 40.9      | 63.3 |
| Coral            | ID val              | 86.2 | 78.8 | 68.7       | 47.6     | 41.5      | 64.6 |
| SWAD             | Loss-aware       | 88.1 | **79.1** | 70.6       | 50.0     | 46.5      | 66.9 |
|---|---|---|---|---|---|---|---|
| ERM              | ID val              | 85.9 | 78.1 | 69.4       | 50.4     | 44.3      | 65.6 |
| Soups/DiWA             | Greedy           | 88.0 | 78.5 | 71.5       | 51.6     | **47.7**      | 67.5 |
| Soups/DiWA             | Uniform          | 88.7 | 78.4 | 72.1       | 51.4     | 47.4      | 67.6 |
| Soups/DiWA$^{\dagger}$ | Uniform$^{\dagger}$          | 89.0 | 78.6 | 72.8       | 51.9     | **47.7**      | 68.0 |
|---|---|---|---|---|---|---|---|
| Inter-training | ID val  | 89.0 | 77.7 | 69.9 | 46.7 | 44.5 | 65.6 |
| Fusing         | ID val  | 88.0 | 78.5 | 71.5 | 46.7 | 44.4 | 65.8 |
| Ratatouille    | Uniform | 89.5 | 78.5 | 73.1 | 51.8 | 47.5 | 68.1 |
| Ratatouille    | Greedy  | **90.5** | 78.7 | 73.4 | 49.2 | **47.7** | 67.9 |
| Ratatouille$^{\dagger}$    | Uniform$^{\dagger}$ | 89.8 | 78.3 | **73.5** | **52.0** | **47.7** | **68.3** |


# Other information

## License

This source code is released under the MIT license, included [here](LICENSE).

## Why ratatouille ?

We named our method after this traditional French dish for two main reasons. Firstly, the ratatouille is often used as a way to recycle leftover vegetables. Secondly, the ratatouille is better prepared by cooking each ingredient separately before mixing them: this technique ensures that each ingredient “will taste truly of itself”, as [noted](https://www.bbc.com/travel/article/20200812-the-right-way-to-make-ratatouille) by chef Joël Robuchon.

## Citation

If you find this code useful for your research, please consider citing our work:

```
@article{rame2022recycling,
  title={Model Ratatouille: Recycling Diverse Models for Out-of-Distribution Generalization},
  author={Ram{\'e}, Alexandre and Ahuja, Kartik and Zhang, Jianyu and Cord, Matthieu and Bottou, L{\'e}on and Lopez-Paz, David},
  journal={arXiv preprint arXiv:2212.10445},
  year={2022}
}
```

Correspondence to alexandre.rame at isir.upmc.fr
