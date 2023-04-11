# Forward-Forward Learning of a Single Perceptron
Python code for applying the Forward-Forward Learning Algorithm to a Single Perceptron for classification of handwritten digits in the MNIST data set (requires PyTorch).

The Jupyter Notebook file also includes the text outputted during training with jittered images of the perceptron with 8000 outputs, reaching a test error  below 1 %. The following performance is obtained with this code:

| Number of Outputs  | Data Augmentation | Test Error (%)
| ------------- | ------------- | ------------- |
|  125  | none  | ≲ 2.6 |
|  500  | none  | ≲ 1.9 |
| 2000  | none  | ≲ 1.7 |
| 8000  | none  | ≲ 1.6 |
|  125  | jittered  | ≲ 2.2 |
|  500  | jittered  | ≲ 1.4 |
| 2000  | jittered  | ≲ 1.1 |
| 8000  | jittered  | ≲ 1.0 |

See the arXiv preprint: http://arxiv.org/abs/2304.03189 (2023).

### The Concept of Forward-Forward Learning Applied to a Multi Output Perceptron
**Author:** K. F. Karlsson

**Abstract:** The concept of a recently proposed Forward-Forward learning algorithm for fully connected artificial neural networks is applied to a single multi output perceptron for classification. The parameters of the system are trained with respect to increased (decreased) "goodness" for correctly (incorrectly) labelled input samples. Basic numerical tests demonstrate that the trained perceptron effectively deals with data sets that have non-linear decision boundaries. Moreover, the overall performance is comparable to more complex neural networks with hidden layers. The benefit of the approach presented here is that it only involves a single matrix multiplication.


### References
G. Hinton, The Forward-Forward Algorithm: Some Preliminary Investigations, http://arxiv.org/abs/2212.13345 (2022)
