---
jupyter:
  jupytext:
    formats: ipynb,qmd,md
    text_representation:
      extension: .md
      format_name: markdown
      format_version: '1.3'
      jupytext_version: 1.19.1
  kernelspec:
    display_name: Python 3 (ipykernel)
    language: python
    name: python3
---

<!-- #region cell_style="center" slideshow={"slide_type": "slide"} -->
# Intro to programming and training Neural Networks with PyTorch


<center><img src="figures/pytorch_logo.png"></center>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Introduction

* PyTorch is an end-to-end open source platform for ML 
* Allows to easily build and deploy ML powered applications.
* Not only Neural Networks



<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## PyTorch

* PyTorch is an "optimized tensor library for deep learning"
* Scientific computing, general ML, Neural Networks
* C++/python (we use the latter)
* Easy to implement complex architectures with few lines of code
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Docs: https://docs.pytorch.org/docs

* Installation inscructions (you should be already set up!)
* Tutorials from the groud up
* Reference API
  * Models and Layers
  * Data handling
  * Helper functionalities (utils)

<!-- #endregion -->

## Other ML/DL libraries

* Tensorflow/Keras
* JAX
* ...
* Most concepts translate across libraries with minor differences

<!-- #region slideshow={"slide_type": "slide"} -->
## Some terminology

* A dataset in supervised learning is made of a number of (features, label) pairs
* Example, a dataset of diabetic patients is made of:
    * Features: information describing each patient (weight, height, blood pressure...)
    * Labels: whether each patient is diabetic or not (glucose levels higher or lower than...)
* Each (features, label) pair is also called a _sample_ or _example_. Basically a data point
* Features are also sometimes called _inputs_ when referred to something you feed to a NN
* Labels are compared to the NN's _outputs_ to see how well the network is doing compared to the truth
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## What is a tensor

The main variables in PyTorch are tensors:

> A tensor is often thought of as a generalized matrix. That is, it could be a 1-D matrix (a vector), a 3-D matrix (something like a cube of numbers), even a 0-D matrix (a single number), or a higher dimensional structure that is harder to visualize. The dimension of the tensor is called its rank.
>
> Src: https://www.kaggle.com/discussions/getting-started/159424




<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## What is a tensor

The main variables in TensorFlow are, of course, tensors:

> A tensor is often thought of as a generalized matrix. That is, it could be a 1-D matrix (a vector), a 3-D matrix (something like a cube of numbers), even a 0-D matrix (a single number), or a higher dimensional structure that is harder to visualize. The dimension of the tensor is called its rank.

## TensorFlow operates on tensors

> TensorFlow computations are expressed as stateful dataflow graphs. The name TensorFlow derives from the operations that such neural networks perform on multidimensional data arrays, which are referred to as tensors.
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## The first step is to build a graph of operations

* NNs are defined in TensorFlow as graphs through which the data flows until the final result is produced
* Before we can do any operation on our data (images, etc) we need to build the graph of tensor operations
* When we have a full graph built from input to output, we can run (flow) our data (training or testing) through it.

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Tensors and data are *not* the same thing
* Tensors are, rather, a symbolic representation of the data
* Think about the function $g = f(x)$: as long as we do not assign a value to $x$, we will not have a fully computed $g$
* In this case, $g$ is the output tensor, $x$ the input tensor, $f$ the tensor operation (a Neural Network?)

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Example

* We have a set of color images of size $1000x1000$ pixels (1 megapixel) that we want to use on our NN 
* We define tensors with shape $(n, 1000, 1000, 3)$
    * $n$ is the number of images that we are presenting to our network in one go (a "batch")
    * $1000x1000$: image pixels
    * $3$ is the number of channels (RGB)
    * Grayscale images tensors would have shape $(n, 1000, 1000, 1)$
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## One thing to remember when operating on tensors

The dimensions between tensors coming out of the $i$-th node and those going into the $(i+1)$-th node *must* match:

* If each sample in our dataset is made of 10 features, the first (input) layer must accept a tensor of shape $(n, 10)$
* If the first layer in our NN outputs a 3D tensor, the second layer must accept a 3D tensor as input
* Check the documentation to make sure what input-output shapes are allowed ([example](https://keras.io/api/layers/convolution_layers/convolution1d/))
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Here's how a NN layer looks like in TensorFlow:

* 7 samples in batch
* 784 inputs
* 500 outputs

<center><img src="figures/run_metadata_graph.png"></center>
<!-- #endregion -->

<!-- #region jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": "slide"} -->
## Here is how a model is built and trained in Keras
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "-"} -->
```python
#Multi-layer perceptron (one hidden layer)
model = Sequential()
model.add(Dense(3, input_dim=3, activation='sigmoid'))
model.add(Dense(1, activation='sigmoid'))

#Gradient descent algorithm, Mean Squared Error as Loss function
model.compile(optimizer='sgd', loss='mse', metrics=['mse'])

#Training for 10 iterations of the data (epochs)
history = model.fit(data, labels, epochs=10, batch=32)
...
```
<!-- #endregion -->

What does each bit do?

<!-- #region cell_style="center" jp-MarkdownHeadingCollapsed=true slideshow={"slide_type": "slide"} -->
## A neural network in Keras is called a Model

The simplest kind of model is of the Sequential kind:
<!-- #endregion -->

```python cell_style="center" slideshow={"slide_type": "-"}
#from tensorflow.keras.models import Sequential
#model = Sequential()

import torch

torch.nn.Linear(4, 8)(torch.tensor([1.0, 1.0, 0.0, -1]))

```

<!-- #region slideshow={"slide_type": "slide"} -->
This is an "empty" model, with no layers, no inputs or outputs are defined either.

Adding layer is easy. Let's say we have data for participants to a clinical study. For participant we have recorded: blood pressure, BMI and age.

The participants have been diagnosed as healthy or sick, these will be our labels.

We could define a simple NN that predicts if a participant is healthy or sick as follows:
<!-- #endregion -->

```python cell_style="center" slideshow={"slide_type": "-"}
from tensorflow.keras.layers import Dense
model = Sequential()
#model.add(Dense(units=2, input_dim=3, activation="softmax"))
model.add(Dense(units=4, activation='relu', input_dim=3, name="input"))
model.add(Dense(units=2, activation='softmax', name="output"))

```

<!-- #region slideshow={"slide_type": "fragment"} -->

A "Dense" layer is a fully connected layer as the ones we have seen in Multi-layer Perceptrons.
The above is equal to having this network:

<center><img src="figures/simplenet_patient.png"></center>

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
If we want to see the layers in the Model this far, we can just call:
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
model.summary()
```

<!-- #region slideshow={"slide_type": "-"} -->
Notice the number of parameters, can you tell why 12 and 8 parameters for each layer?
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
Using "model.add()" keeps stacking layers on top of what we have:
<!-- #endregion -->

```python
model.add(Dense(units=2, activation=None))
model.summary()
```

<!-- #region slideshow={"slide_type": "slide"} -->
One can also declare the model in one go, by passing a list of layers to Sequential() like so:
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
model = Sequential([
    Dense(units=4, activation='relu', input_dim=3),
    Dense(units=2, activation='softmax'),
    Dense(units=2, activation=None)
])

model.summary()
```

<!-- #region slideshow={"slide_type": "slide"} -->
If we want to see the layers in the Model this far, we can just call:
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
from tensorflow.keras.utils import plot_model

plot_model(model, "figures/simplenet_model.png", show_shapes=True)
```

## Small exercise

* Can you write code to make a simple NN model on Keras?
* Open the `exercises` jupyter notebook

<!-- #region slideshow={"slide_type": "slide"} -->
## Keras layers (https://keras.io/api/layers/)

Common layers (we will cover most of these!)

* Trainable
    * <font color='red'>Dense (fully connected/MLP)</font>
    * <font color='red'>Conv1D (2D/3D)</font>
    * <font color='red'>Recurrent: LSTM/GRU/Bidirectional</font>
    * <font color='red'>Embedding</font>
    * <font color='red'>Lambda (apply your own function)</font>

* Non-trainable
    * <font color='red'>Dropout</font>
    * <font color='red'>Flatten</font>
    * BatchNormalization
    * MaxPooling1D (2D/3D)
    * Merge (add/subtract/concatenate)
    * <font color='red'>Activation (Softmax/ReLU/Sigmoid/...)</font>

<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Compiling a model

Once we have defined a model we want to "compile" it

This means chosing a Loss function and an Optimizer (the algorithm that finds the minimum loss possible).
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
model.compile(optimizer='rmsprop',                    #adaptive learning rate method
              loss='sparse_categorical_crossentropy', #loss function for classification problems with integer labels
              metrics=['accuracy'])                   #the metric doesn't influence the training

model.optimizer.get_config()
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Losses (https://keras.io/api/losses/)

These are the functions used to evaluate and train the neural network

Common losses for classification problems:
* CategoricalCrossentropy
* SparseCategoricalCrossentropy
* KLDivergence

Common losses for regression problems:
* MeanSquaredError
* MeanAbsoluteError
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Metrics (https://keras.io/api/metrics/)

Common metrics for classification:
* Accuracy/CategoricalAccuracy (respectively for integer labels or one-hot labels)
* SparseCategoricalCrossentropy/CategoricalCrossentropy (integer/one-hot labels)
* Precision/Recall
* AUC

Common metrics for regression:
* MeanSquaredError
* MeanAbsoluteError
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Metrics (https://keras.io/api/metrics/)

Notice the "metrics" parameter, which accepts a list of values. Multiple metrics can be shown during training.
Metrics are only to visualize how the training is going, they don't have an effect on training itself
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
from tensorflow.keras.optimizers import RMSprop
model.compile(optimizer=RMSprop(learning_rate=1.0),   #adaptive learning rate method
              loss='sparse_categorical_crossentropy', #loss function for classification problems with integer labels
              metrics=['accuracy', 'recall'])         #the metric doesn't influence the training
```

<!-- #region cell_style="split" slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)

* They are algorithms for gradient descent
* A few to choose from:
    * SGD (Stochastic Gradient Descent)
    * RMSprop (Root Mean Square propagation)
    * Adadelta (Adaptive delta)
    * Adam (Adaptive Moment estimation)

<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "skip"} -->
<br>
<br>
<br>
<br>
<img src="figures/gradient_descent.png">
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "slide"} -->
## Gradient Descent 

We have seen how gradient descent works:

For each epoch:
* Get predicted $y$ ($ŷ$) for all $N$ samples
* Calculate error (loss)
* Calculate all gradients (backprop)
* Apply gradients to weights
    
Pros/cons:
* Stable procedure
* Guarantees lower error at next step
* Will get stuck at local minimum
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "-"} -->
<br>
<br>
<br>
<br>
<img src="figures/gradient_descent.png">
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "slide"} -->
## Stochastic Gradient Descent
For each epoch:
* Divide data in batch blocks of size $n < N$
* For each of the $N/n$ blocks:
    * Get predicted $y$ for $n$ samples
    * Calculate partial loss
    * Calculate gradients (backprop)
    * Apply gradients to weights

Pros/cons:
* Noisy gradients
* Error will still go down overall
* Less likely to get stuck at local minimum
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "-"} -->
<br>
<br>
<br>
<br>
<img src="figures/gradient_descent.png">
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)

We need to choose a learning rate to multiply to our gradient. If it is too small, we risk taking too long to get to a minimum
<center><img src="figures/small_lr.png"></center>
<!-- #endregion -->

<!-- #region hideOutput=true slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)

If it is too large, the network risks becoming unstable, explode

<center><img src="figures/large_lr.png"></center>

Let's test different optimization strategies on Tensorflow playground: http://playground.tensorflow.org
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)

Luckily there are algorithms to address these issues:
* Increase descent speed when past gradients agree with current, slow down otherwise (momentum)
* Annealing (decrease learning rate with passing time)
* Different learning rates for different parameters
* Adaptive learning rate based on gradient
<!-- #endregion -->

<!-- #region cell_style="split" slideshow={"slide_type": "-"} -->
<br>
<br>
<br>
<br>
<img src="figures/adaptive_lr.png">
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)

* They are algorithms for gradient descent
* A few to choose from:
    * SGD (stochastic gradient descent)
        * One learning rate, fixed
        * Old, but works well with Nesterov momentum
    * RMSprop
        * One learning rate per parameter
        * Adaptive learning rate (divide by squared mean of past gradients)
    * Adadelta (adaptive learning rate)
        * Similar to RMSprop, no need to set initial learning rate
    * Adam (Adaptive moment estimation)
        * Combines pros from RMSprop, Adadelta, works well with most problems

<!-- #endregion -->

<!-- #region cell_style="center" slideshow={"slide_type": "slide"} -->
## Optimizers (https://keras.io/api/optimizers/)
<br>
<br>
<center><img src="figures/adam_et_al.png" width=500></center>
<div style="text-align: right">("Adam: A Method for Stochastic Optimization", 2015)</div>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Training the model: fit() function (https://keras.io/api/models/model_training_apis/)

* We are almost ready to train the model, I swear
* fit() is a method of the Model, actually launches training on a dataset with features and labels
* X_train, y_train: features and labels
* batch: how many samples between each weight update
* epochs: how many times we iterate through the dataset
* validation_data: used to evaluate the model at the end of every epoch, NOT used for training
<!-- #endregion -->

<!-- #region -->
```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch=32, epochs=10)
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Training the model: fit() function (https://keras.io/api/models/model_training_apis/)

* Ok, last thing we need is the actual data, then we can train the model
<!-- #endregion -->

<!-- #region -->
```python
model.fit(X_train, y_train, validation_data=(X_val, y_val), batch=32, epochs=10, validation_data=(X_val, y_val))
```
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## What is this validation thing? Do I really need it?

* Yes, yes you do
* Helps understanding if the model is learning anything useful
* Take some of your labelled data, set it aside, call it validation set and don't train on it
* Evaluate model on validation set at the end of each epoch, see if model works on unseen data
* If it works well on training set but not on validation set, you're overfitting

<img src="figures/overfitting_class.png" width=300>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## What is this validation thing? Do I really need it?

* If it works well on training set but not on validation set, you're overfitting
* Validation data is used to adapt hyperparameters, select best models
* Validation data is NOT testing data (more on this later)
* Let's try this on Tensorflow playground: http://playground.tensorflow.org

<img src="figures/early_stopping.png" width=500>
<!-- #endregion -->

<!-- #region slideshow={"slide_type": "slide"} -->
## Ok, can we PLEASE train a NN now?

* Let's generate some artificial data, see what happens
* Classification dataset, 2 classes
* Let's say 10,000 samples, three features per sample
* Random data
<!-- #endregion -->

```python
import numpy as np

# Generate dummy data
data = np.random.random((10000, 3))
labels = np.random.randint(2, size=(10000, 1))

#let's print the first sample (three floats) and its corresponding label:
print(np.hstack((data[0:10,:], labels[0:10])))
```

<!-- #region slideshow={"slide_type": "slide"} -->
## We have the data, now make the model, compile it, train it

* At the last layer of a classifier use the _softmax_ activation (more on this later)
* Batch size is 32, 10 epochs
* Take 10% of the data, reserve it for validation
<!-- #endregion -->

```python slideshow={"slide_type": "subslide"}
model = Sequential()
model.add(Dense(4, input_dim=3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.1)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Let's visualize our training curves

* Plots loss and accuracy for train and validation sets separately

<!-- #endregion -->

```python
model = Sequential()
model.add(Dense(4, input_dim=3, activation='tanh'))
model.add(Dense(3, activation='tanh'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.1)
```

```python
%matplotlib inline

import matplotlib.pyplot as plt

def plot_loss_acc(history):
    try:
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
    except:
        plt.plot(history.history['acc'])
        plt.plot(history.history['val_acc'])

    plt.plot(history.history['loss'])
    plt.plot(history.history['val_loss'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train acc', 'val acc', 'train loss', 'val loss'], loc='upper left')
    plt.show()

```

<!-- #region slideshow={"slide_type": "slide"} -->
## Let's visualize our training curves

* Plots loss and accuracy for train and validation sets separately
* The model didn't learn anything, which makes sense (data is random)
<!-- #endregion -->

```python
plot_loss_acc(history)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Do it again, but with data that actually means something

* A XOR function is not linear
* A perceptron is not able to separate XOR classes
* A MLP should be able to

<img src="figures/3-IP-TRUTH-TABLE2.jpg">

<!-- #endregion -->

<!-- #region cell_style="center" slideshow={"slide_type": "slide"} -->
Let's generate data that is not just binary, but behaves like it:

* A positive (+) input behaves like a 1
* A negative (-) input behaves like a 0
* -0.5 $\oplus$ 0.2 $\oplus$ -0.1 => 1
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
# Generate XOR data
data = np.random.random((1000000, 3)) - 0.5
labels = np.zeros((1000000, 1))

labels[np.where(np.logical_xor(np.logical_xor(data[:,0] > 0, data[:,1] > 0), data[:,2] > 0))] = 1

#let's print some data and the corresponding label to check that they match the table above
for x in range(3):
    print("{0: .2f} xor {1: .2f} xor {2: .2f} equals {3:}".format(data[x,0], data[x,1], data[x,2], labels[x,0]))
```

```python slideshow={"slide_type": "skip"}
from sklearn.decomposition import PCA
pca = PCA(n_components=2)
transformed = pca.fit_transform(data)
plt.scatter(transformed[:,0], transformed[:,1], c=labels)
```

<!-- #region slideshow={"slide_type": "slide"} -->
Now let's fit a model to the data:
<!-- #endregion -->

```python slideshow={"slide_type": "-"}
from keras.layers import LeakyReLU, Dropout
model = Sequential()
model.add(Dense(16, input_dim=3, activation="tanh"))
model.add(Dense(8, activation="tanh"))
model.add(Dense(4, activation="tanh"))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=2, batch_size=128, validation_split=0.1, shuffle=True)
```

```python
from keras.layers import LeakyReLU, Dropout
model = Sequential()
model.add(Dense(16, input_dim=3, activation="tanh"))
model.add(Dense(8, activation="tanh"))
model.add(Dense(4, activation="tanh"))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=100, batch_size=128, validation_split=0.1, shuffle=True)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## XOR data

* Better than random!
* Notice the difference between train and validation curves
<!-- #endregion -->

```python
plot_loss_acc(history)
```

<!-- #region slideshow={"slide_type": "slide"} -->
## Exercise: can you do better?

* Check the exercise notebook!
<!-- #endregion -->

```python slideshow={"slide_type": "slide"}
model = Sequential()
model.add(Dense(4, input_dim=3, activation='sigmoid'))
model.add(Dense(3, activation='sigmoid'))
model.add(Dense(2, activation='softmax'))

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
# Train the model, iterating on the data in batches of 32 samples
history = model.fit(data, labels, epochs=10, batch_size=32, validation_split=0.1)
```
