# Extreme Learning Machine

## Introduction
This is the respository that implement the Extreme Learning Machine for Single Hidden-layer Feedfoward Neural Network (SLFN). This learning algorithm is extremely fast if compare to Back-propagation or Gradient-based learning algorithm.

## Benchmark

**Boston Housing dataset for Regression problem**

|Algorithm|Type|Training time (miliseconds)|Trainset RMSE|Testset RMSE|
|---|---|:---:|:---:|:---:|
|ELM|Neural Network|54.62|4.69|4.82|
|Ridge|Linear Model|2.66|4.72|4.75|
|SVR|Support Vector Machine|26.32|8.25|8.15|
|K-Nearest Neighbors|Nearest Neighbors|1.97|4.98|6.08|
|Decision Tree|Tree-based|7.2|0 (overfit)|5.1|
|Random Forest|Tree-based Esemble|293.13|1.19|3.8|
|Perceptron (Back-propagation)|Neural Network|237.5|6.94|7.43|


**MNIST dataset for Classification problem**
|Algorithm|Type|Training time (miliseconds)|Trainset accuracy (%)|Testset accuracy (%)|
|---|---|:---:|:---:|:---:|
|ELM|Neural Network|4754.13|91.94|92.22|
|Logistic Regression|Linear Model|21112.03|93.39|92.55|
|SVC|Support Vector Machine|280275.89|98.99|97.92|
|K-Nearest Neighbors|Nearest Neighbors|5.07|98.19|96.88|
|Decision Tree|Tree-based|17913.2|100.0 (overfit)|87.68|
|Random Forest|Tree-based Esemble|37244.7|100.0|96.95|
|Perceptron (Back-propagation)|Neural Network|379703.05|99.73|97.85|

*Above is just a compact comparision table, you guys can see more detail in notebook.*

## Structure
- `elm.py`: This is a file that implement `ELMBase` class and two class for Classification and Regression: `ELMClassifier` and `ELMRegressor`.
- `elm_classification.ipynb`: Notebook contain testing code for `ELMClassifier` for MNIST dataset.
- `elm_regression.ipynb`: Notebook contain testing code for `ELMRegressor` on Boston Housing dataset.
- `utils.py`: Utility functions support above 3 files.

## References

[1] Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning machine: Theory and applications, 2006. https://doi.org/10.1016/j.neucom.2005.12.126.