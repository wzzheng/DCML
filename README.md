# Deep Compositional Metric Learning

This repository is the official implementation of **Deep Compositional Metric Learning**. 

The majority of this codebase is built upon research and implementations provided in 
Paper: https://arxiv.org/abs/2002.08473 
Repo: https://github.com/Confusezius/Revisiting_Deep_Metric_Learning_PyTorch

## Datasets 

### CUB-200-2011

Download from (http://www.vision.caltech.edu/visipedia/CUB-200-2011.html).

### Cars196

Download from (http://ai.stanford.edu/~jkrause/cars/car_dataset.html).

### Standard Online Products

Download from (http://cvgl.stanford.edu/projects/lifted_struct/)

Organize the dataset as follows:

__CUB200-2011__
```
cub200
└───images
|    └───001.Black_footed_Albatross
|           │   Black_Footed_Albatross_0001_796111
|           │   ...
|    ...
```

__CARS196__
```
cars196
└───images
|    └───Acura Integra Type R 2001
|           │   00128.jpg
|           │   ...
|    ...
```

__Online Products__
```
online_products
└───images
|    └───bicycle_final
|           │   111085122871_0.jpg
|    ...
|
└───Info_Files
|    │   bicycle.txt
|    │   ...
```

## Requirements
* Pytorch 1.2.0+ & Faiss-Gpu
* Python 3.7
* pretrainedmodels 0.7.4
* touchvision 0.5.0

## Training
Training is done by using 'mian.py', with settings available in 'parameters.py'. The main parameters that control our compositiors and ensembles are '--compos_num' and '--ensemble_num'. 
For example, we initialize the '--compos_num' to 4 and '--ensemble_num' to 4. More compositions are worth trying.

To train the DCML model with margin loss on CUB200, run this command:

```
python main.py --dataset cub200 --tau 55 --gamma 0.2 --gpu 0 --seed 0 --compos_num 4 --ensemble_num 4 --embed_dim 128 --bs 100
                           --n_epochs 300 --samples_per_class 2 --loss margin --batch_mining distance --arch resent50_frozen_normalize
```

Besides, our architecture can be implemented in the diva framework (http://arxiv.org/abs/2004.13458) with this command:

```
python main.py --dataset cub200 --tau 55 --gamma 0.2 --gpu 0 --seed 0 --compos_num 4 --ensemble_num 4 --embed_dim 128 --bs 100
                           --n_epochs 300 --samples_per_class 2 --loss margin --batch_mining distance --arch resent50_frozen_normalize
                           --diva_ssl fast_moco --lr 0.000015 --evaltypes all --diva_rho_decorrelation 1500 1500 1500 --diva_features discriminative selfsimilarity shared intra
                           --diva_moco_temperature 0.01 --diva_moco_n_key_batches 30 --diva_aplha_ssl 0.5 diva_alpha_shared 0.3 --diva_alpha_intra 0.3
```


## Device 

We tested our code on a linux machine with an Nvidia RTX 2080ti GPU card. We recommend using a GPU card with a memory > 11GB.
