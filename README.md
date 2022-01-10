# Extreme Learning Machine

## Introduction
This is the respository that implement the Extreme Learning Machine for Single Hidden-layer Feedfoward Neural Network (SLFN). This learning algorithm is extremely fast if compare to Back-propagation or Gradient-based learning algorithm.

## Benchmark

**Boston Housing dataset for Regression problem**

|Algorithm|Training time (miliseconds)|RMSE on trainset|RMSE on testset|
|:---|:---:|:---:|:---:|
|ELM|78.84|4.69|4.82|
|Ridge|1.86|4.72|4.75|
|Support Vector Machine|16.36|8.25|8.15|
|K-Nearest Neighbors (K=5)|1.82|4.98|6.08|
|Decision Tree|5.81|0 (overfit)|5.62|
|Random Forest|291.68|1.12|3.68|
|Single-Layer Perceptron (Back-propagation)|349.52|7.08|7.5|


**MNIST dataset for Classification problem**
- Consumed training time: 4.73 seconds
- Accuracy Score for train set: 91.94%
- Accuracy Score for test set: 92.22%

## Structure
- `elm.py`: This is a file that implement `ELMBase` class and two class for Classification and Regression: `ELMClassifier` and `ELMRegressor`.
- `elm_classification.ipynb`: Notebook contain testing code for `ELMClassifier` for MNIST dataset.
- `elm_regression.ipynb`: Notebook contain testing code for `ELMRegressor` on Boston Housing dataset.
- `utils.py`: Utility functions support above 3 files.

## References

[1] Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning machine: Theory and applications, 2006. https://doi.org/10.1016/j.neucom.2005.12.126.