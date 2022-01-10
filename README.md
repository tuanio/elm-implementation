# Extreme Learning Machine

## Introduction
This is the respository that implement the Extreme Learning Machine for Single Hidden-layer Feedfoward Neural Network (SLFN). This learning algorithm is extremely fast if compare to Back-propagation or Gradient-based learning algorithm.

## Benchmark

**Boston Housing dataset for Regression problem**

|Algorithm|Type|Training time (miliseconds)|RMSE on trainset|RMSE on testset|
|---|---|:---:|:---:|:---:|
|ELM|Neural Network|54.62|4.69|4.82|
|Ridge|Linear Model|2.66|4.72|4.75|
|SVR|Support Vector Machine|26.32|8.25|8.15|
|K-Nearest Neighbors|Nearest Neighbors|1.97|4.98|6.08|
|Decision Tree|Tree-based|7.2|0 (overfit)|5.1|
|Random Forest|Tree-based Esemble|293.13|1.19|3.8|
|Perceptron (Back-propagation)|Neural Network|237.5|6.94|7.43|


**MNIST dataset for Classification problem**
|Algorithm|Type|Training time (miliseconds)|RMSE on trainset|RMSE on testset|
|---|---|:---:|:---:|:---:|
|ELM|Neural Network|4488.25|4.69|4.82|
|Logistic Regression|Linear Model|21398.93|4.72|4.75|
|SVC|Support Vector Machine|16.36|8.25|8.15|
|K-Nearest Neighbors|Nearest Neighbors|1.82|4.98|6.08|
|Decision Tree|Tree-based|5.81|0 (overfit)|5.62|
|Random Forest|Tree-based Esemble|291.68|1.12|3.68|
|Perceptron (Back-propagation)|Neural Network|349.52|7.08|7.5|

*Above is just a compact comparision table, you guys can see more detail in notebook.*

## Structure
- `elm.py`: This is a file that implement `ELMBase` class and two class for Classification and Regression: `ELMClassifier` and `ELMRegressor`.
- `elm_classification.ipynb`: Notebook contain testing code for `ELMClassifier` for MNIST dataset.
- `elm_regression.ipynb`: Notebook contain testing code for `ELMRegressor` on Boston Housing dataset.
- `utils.py`: Utility functions support above 3 files.

## References

[1] Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning machine: Theory and applications, 2006. https://doi.org/10.1016/j.neucom.2005.12.126.