# Forward-Forward Learning of a Single Perceptron
Python code for applying the Forward-Forward Learning Algorithm to a Single Perceptron for classification of handwritten digits in the MNIST data set (requires PyTorch).

The Jupyter Notebook file also includes the text output during training with jittered images of the perceptron with 8000 outputs.

Results obtained by this code are presented and discussed in a arXiv preprint: http://arxiv.org/abs/2304.03189 (2023)

### The Concept of Forward-Forward Learning Applied to a Multi Output Perceptron
**Author:** K. F. Karlsson

**Abstract:** The concept of a recently proposed Forward-Forward learning algorithm for fully connected artificial neural networks is applied to a single multi output perceptron for classification. The parameters of the system are trained with respect to increased (decreased) "goodness" for correctly (incorrectly) labelled input samples. Basic numerical tests demonstrate that the trained perceptron effectively deals with data sets that have non-linear decision boundaries. Moreover, the overall performance is comparable to more complex neural networks with hidden layers. The benefit of the approach presented here is that it only involves a single matrix multiplication.


### References
G. Hinton, The Forward-Forward Algorithm: Some Preliminary Investigations, http://arxiv.org/abs/2212.13345 (2022)
