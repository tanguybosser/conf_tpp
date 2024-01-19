# Distribution-Free Conformal Joint Prediction Regions for Neural Marked Temporal Point Processes

This repository contains the code base to reproduce the main experiments of the paper 'Distribution-Free Conformal Joint Prediction Regions for Neural Marked Temporal Point Processes'.  

## Abstract

Sequences of labeled events observed at irregular intervals in continuous time are ubiquitous across various fields. Temporal Point Processes (TPPs) provide a mathematical framework for modeling these sequences, enabling inferences such as predicting the arrival time of future events and their associated label, called mark. However, due to model misspecification or lack of training data, these probabilistic models may provide a poor approximation of the true, unknown underlying process, with prediction regions extracted from them being unreliable estimates of the underlying uncertainty. This paper develops more reliable methods for uncertainty quantification in neural TPP models via the framework of conformal prediction. A primary objective is to generate a distribution-free joint prediction region for the arrival time and mark, with a finite-sample marginal coverage guarantee. A key challenge is to handle both a strictly positive, continuous response and a categorical response, without distributional assumptions. We first consider a simple but overly conservative approach that combines individual prediction regions for the event arrival time and mark. Then, we introduce a more effective method based on bivariate highest density regions derived from the joint predictive density of event arrival time and mark. By leveraging the dependencies between these two variables, this method exclude unlikely combinations of the two, resulting in sharper prediction regions while still attaining the pre-specified coverage level. We also explore the generation of individual univariate prediction regions for arrival times and marks through conformal regression and classification techniques. Moreover, we investigate the stronger notion of conditional coverage. Finally, through extensive experimentation on both simulated and real-world datasets, we assess the validity and efficiency of these methods.

## Installation

The experiments have been run in python 3.9 with the package versions listed in requirements.txt, which can be installed using:
```
pip install -r requirements.txt
```

## Reproducing the experiments

1. Run the script `run.sh` to run the conformal methods for all datasets and base models that we consider and save the results.
2. Run `make_figures.ipynb` to reproduce the corresponding figures.
3. Other figures on toy examples are self-contained in the directory `toy`
