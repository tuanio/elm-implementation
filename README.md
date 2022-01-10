# Extreme Learning Machine

## Introduction
This is the respository that implement the Extreme Learning Machine for Single Hidden-layer Feedfoward Neural Network (SLFN). This learning algorithm is extremely fast if compare to Back-propagation or Gradient-based learning algorithm.

## Benchmark

**Boston Housing dataset for Regression problem**
- Consumed training time: 198 miliseconds (0.198 seconds)
- Mean Square Error for train set: 22.004
- Mean Square Error for test set: 23.195

**MNIST dataset for Classification problem**
- Consumed training time: 4.73 seconds
- Accuracy Score for train set: 91.94%
- Accuracy Score for test set: 92.22%

## Structure
- `elm.py`: This is a file that implement `ELMBase` class and two class for Classification and Regression: `ELMClassifier` and `ELMRegressor`.
- `elm_classification.ipynb`: Notebook contain testing code for MNIST dataset.
- `elm_regression.ipynb`: Notebook contain testing code for Boston Housing dataset.
- `utils.py`: Utility functions support above 3 files.

## References

[1] Guang-Bin Huang, Qin-Yu Zhu, Chee-Kheong Siew, Extreme learning machine: Theory and applications, 2006. https://doi.org/10.1016/j.neucom.2005.12.126.