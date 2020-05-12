# ADRML
**A**nticancer **D**rug **R**esponse prediction using **M**anifold **L**earning is a computaional method to predict log IC50 for cell line-drug pairs. This method is used in cancer drug sensitivity prediction, which is a fundamental issue in precision medicine. 

>Fatemeh Ahmadi Moughari, Changiz Eslahchi; ADRML: Anticancer Drug Response Prediction Using Manifold Learning

This repository contains the implementated codes of ADRML, the collected and preprocessed data, in addition to the cmputed similarity matrices for cell lines and drugs.

## A giude to run ADRML
Please make sure to have the following libraries installed.
#### Required libraries
Python 3.r and upper:
- Numpy
- sklearn
- Argparse
- random
- copy
- math
- sys

#### Input parameters
To execute the codes, the user must provide three input files
- `response_dirc`: the directory to a file which contains the real values of log IC50 for all cell lines and drugs
- `simC_dirc`: the directory to a file that is a square matrix containing the similarity of cell lines
- `simD_dirc`: the directory to a file that is a square matrix containing the similarity of drugs.
- `dim`: the dimension of latent space
- `miu`: the regularization coefficient for latent matrices
- `lambda`: the coefficient that controls the similarity conservation while manifold learning
- `CV`: the number of folds in cross validation
- `repetition`: the number of repeting the cross validation 
The real matrix for log IC50 values for CCLE and GDSC are presented in `Data/CCLE/Features/LogIC50.csv` and `Data/GDSC/Features/LogIC50.csv`, respectively. Moreover, the required similarity files for CCLE and GDSC are provided in `Data/CCLE/Similarities` and `Data/GDSC/Similarities`. There are several types of cell line similarity based on Expression, Mutation, and CNV, and numerous types of drug similarities based on Chemical, Target, and KEGG pathways. 
The recommended values for hyper-parametrs are `dim=0.7`, `miu=8`, `lambda=4`, `CV=5`, `repetition=30`.

__Command__

The following command is a sample of executing ADRML
```sh
python ADRML.py response_dirc=../Data/CCLE/Features/LogIC50.csv  simC_dirc=../Data/CCLE/Similarities/Expression.csv simD_dric=../Data/CCLE/Similarities/Target.csv dim=0.7 miu=8 lambda=4 CV=5 repetition=30
```

## Contact

Please do not hesitate to contact us at (f.ahmadi.moughari@gmail.com) or (ch.eslahchi@sbu.ac.ir) if there is any question. 

