# MCVAE / sMCVAE

Code related to the paper:

*Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data*

by  Luigi Antelmi, Nicholas Ayache, Philippe Robert, Marco Lorenzi.

Link: http://proceedings.mlr.press/v97/antelmi19a.html

BibTeX citation:
```bibtex
@inproceedings{Antelmi2019,
author = {Antelmi, Luigi and Ayache, Nicholas and Robert, Philippe and Lorenzi, Marco},
booktitle = {Proceedings of the 36th International Conference on Machine Learning},
editor = {Chaudhuri, Kamalika and Salakhutdinov, Ruslan},
pages = {302--311},
publisher = {PMLR},
title = {{Sparse Multi-Channel Variational Autoencoder for the Joint Analysis of Heterogeneous Data}}, 
year = {2019}
}
```

# Installation

## Linux
Install conda:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh  
bash Miniconda3-latest-Linux-x86_64.sh
```

Download this github repository and move into in:
```bash
git clone https://gitlab.inria.fr/epione_ML/mcvae
cd mcvae
```

Install the customized python environment:
`conda env create -f environment.yml`

Activate the python environment:
`conda activate py37`

Install the mcvae package:
`python setup.py install`