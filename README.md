# Neuron Shapley: Discovering the Responsible Neurons
*Xander Davies, Max Nadeau, Max Li*

This repository is an implementation of the biased filter removal experiment in Ghorbani et al's [Neuron Shapley: Discovering the Responsible Neurons](https://arxiv.org/abs/2002.09815) (2020).

Organization:
- `datasets.py` configures the [CelebA](https://mmlab.ie.cuhk.edu.hk/projects/CelebA.html) and [FairFace](https://github.com/joojs/fairface) datasets and dataloaders.
- `model.py` configures a [SqueezeNet](https://arxiv.org/abs/1602.07360) model for the binary gender detection class. It also defines a `epoch` function, which is a generic train/test loop.
- `train_squeeze.py` finetunes a ImageNet-pretrained SqueezeNet on a gender detection task from the CelebA dataset, and evaluates the resulting model on 500 images from each Black/White Male/Female split of FairFace.
- `mab_shapley.py` finds shapley values for each filter on the FairFace dataset using the multi-armed bandit algorithm described by Ghorbani et al., and outputs approximations of filter shapley values to `shapley_values.pkl`.
- `evaluate.py` evaluates the accuracy on FairFace (decomposed by race and gender) when removing filters with negative shapley values.
- `squeezenet.pth` stores the weights of a SqueezeNet after two epochs of fine tuning on the CelebA gender detection task.
- `shapley_values.pkl` stores the shapley values obtained after 195 iterations of the MAB algorithm.

Necessary Datasets:
- `./data/celeba` should contain the CelebA dataset, stored as `*.jpg` files within `test`, `train`, and `val` subfolders. The attributes files should be stored in `list_landmarks_align_celeba.csv`
- `./data/fairface` should contain the FairFace dataset, with `train` and `val` subfolders and a `fairface_label_val.csv` label file. 

*Reproducibility note: running `evaluate.py` will reproduce the above figure using the precomputed shapley values in `shaple_values.pkl` on a small subset of the CelebA and FairFace dataset we've included in this repository. Replicating the MAB algorithm and our full results require dataset downloads.*


## Citation
```BibTex
@article{@misc{https://doi.org/10.48550/arxiv.2002.09815,
  doi = {10.48550/ARXIV.2002.09815},
  url = {https://arxiv.org/abs/2002.09815},
  author = {Ghorbani, Amirata and Zou, James},
  keywords = {Machine Learning (stat.ML), Computer Vision and Pattern Recognition (cs.CV), Machine Learning (cs.LG), Neural and Evolutionary Computing (cs.NE), FOS: Computer and information sciences, FOS: Computer and information sciences},
  title = {Neuron Shapley: Discovering the Responsible Neurons},
  publisher = {arXiv},
  year = {2020},
  copyright = {arXiv.org perpetual, non-exclusive license}
}
}
```