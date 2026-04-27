---
jupyter:
  jupytext:
    formats: ipynb,md
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

<!-- #region editable=true slideshow={"slide_type": ""} -->
# Rigorous splitting of datasets into train and validation
<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
In this lab we will try our hand at protein structure prediction. Given a few thousands protein sequences, for each of the amino acids in the sequences we will try to predict if in the protein structure they will be part of one of three classes:

* $\alpha$-helix
* $\beta$-sheet
* none of the above
<img src="figures/secondary_structure.png">

So the input to our predictor is a protein sequence string such as this one:

```
>APF29063.1 spike protein [Human coronavirus NL63]
MKLFLILLVLPLASCFFTCNSNANLSMLQLGVPDNSSTIVTGLLPTHWFCANQSTSVYSANGFFYIDVGN
HRSAFALHTGYYDVNQYYIYVTNEIGLNASVTLKICKFGINTTFDFLSNSSSSFDCIVNLLFTEQLGAPL
```

for each letter in the sequence, we want to make a classification in the three classes mentioned above.

I have prepared a dataset where all protein sequences have been pre-split into windows of 31 amino acids. We want to predict the class for the amino acid in the center of the window, like so:


predict("MKLFLILLVLPLASCF<font color="red">F</font>TCNSNANLSMLQLG") -> [p(H), p(S), p(C)]

Of course, a neural network will not accept a string input as it is, so we will have to deal with this by converting each letter in our alphabet into an integer. Then, we will use word embeddings to translate the integers into vectors of floating points.

<!-- #endregion -->

<!-- #region editable=true slideshow={"slide_type": ""} -->
## To work on google colab

[Click on this link](https://colab.research.google.com/github/NBISweden/workshop_NN_DL/blob/main/good_practices/labs/data_splits/rigorous_train_validation_splitting.ipynb)
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
#from google.colab import output
#output.enable_custom_widget_manager()
```

```python editable=true slideshow={"slide_type": ""}
#!pip install torchmetrics
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
## Data download

First, let's setup the colab environment, download dataset and other relevant data:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
!mkdir -p data
!wget -v -O data/dataset_sseq_singleseq.pic -L https://zenodo.org/records/19821394/files/dataset_sseq_singleseq.pic?download=1
!wget -v -O data/trainset_distance_matrix.tsv -L https://zenodo.org/records/19821394/files/trainset_distance_matrix.tsv?download=1
!wget -O data/train_set -L https://github.com/NBISweden/workshop_NN_DL/raw/refs/heads/basics/good_practices/labs/data_splits/data/train_set
```

Now let's load libraries and plotting functions:

```python editable=true slideshow={"slide_type": ""}
import random
import numpy as np
import pandas as pd
import sys
from scipy.cluster.hierarchy import dendrogram, linkage, fcluster
from scipy.spatial.distance import squareform
from typing import Optional
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.io as pio

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, ConcatDataset
import torchmetrics

class LivePlot():
    def __init__(self, left_label="Loss", right_label="Accuracy"):
        self.fig = go.FigureWidget(
            make_subplots(specs=[[{"secondary_y": True}]])
        )
        self.fig.update_yaxes(title_text=left_label,  secondary_y=False)
        self.fig.update_yaxes(title_text=right_label, secondary_y=True)

        self.plot_indices = {}
        self.trace_secondary = {}
        display(self.fig)
        self.limits = [0, 0]
        self.current_x = 0

    def report(self, name: str, value: float, secondary_y: bool = False):
        try:
            plot_index = self.plot_indices[name]
        except KeyError:
            plot_index = len(self.fig.data)
            self.fig.add_scatter(
                y=[], x=[], name=name,
                secondary_y=secondary_y
            )
            self.plot_indices[name] = plot_index
            self.trace_secondary[name] = secondary_y
        self.fig.data[plot_index].y += (value,)
        self.fig.data[plot_index].x += (self.current_x,)

    def increment(self, n_ticks: int):
        self.limits[1] += n_ticks
        self.fig.update_layout(xaxis_range=self.limits)

    def set_limit(self, n_ticks: int):
        self.limits[1] = n_ticks
        self.fig.update_layout(xaxis_range=self.limits)

    def tick(self, n_ticks: Optional[int] = None):
        if n_ticks is None:
            n_ticks = 1
        self.current_x += n_ticks

def train(*,
          model: torch.nn.Module, 
          train_loader: DataLoader, 
          dev_loader: DataLoader, 
          optimizer: torch.optim.Optimizer, 
          criterion: torch.nn.Module, 
          max_epochs: int,
          metric: Optional[torchmetrics.metric] = None,
          device: Optional[torch.device] = None,  
          liveplot: Optional[LivePlot]=None):
    if device is None:
        device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    model.to(device)

    for epoch in range(max_epochs):
        training_loss_acc = 0
        training_examples = 0
        model.train()
        
        for i, batch in enumerate(train_loader):
            optimizer.zero_grad()
            
            x_batch, y_batch = batch
            x_batch = x_batch.to(device)  
            y_hat = model(x_batch)

            loss = criterion(y_hat, y_batch.to(device))
            loss.backward()

            optimizer.step()
            training_loss_acc += loss.item()
            training_examples += x_batch.size(0)
        
        
            if i % 100 == 0:
                model.eval()
                with torch.no_grad():
                    dev_loss_acc = 0
                    dev_examples = 0
                    dev_accuracy = 0
                    for i, batch in enumerate(dev_loader):
                        x_batch, y_batch = batch
                        x_batch = x_batch.to(device)
                        y_hat = model(x_batch)
                        dev_loss_acc += criterion(y_hat, y_batch.to(device)).item()
                        dev_examples += x_batch.size(0)
                        if metric:
                            dev_accuracy += metric(torch.argmax(y_hat, -1), y_batch.to(device))
                
                if liveplot is not None:
                    liveplot.tick() # Update the liveplot time
                    liveplot.report("Training loss", training_loss_acc / training_examples)
                    liveplot.report("Development loss", dev_loss_acc / dev_examples)
                    if metric:
                        liveplot.report("Development accuracy", dev_accuracy / (i+1), secondary_y=True)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Now let's create a model. Modify the code below to try different architectures:
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, convolutional=False):
        super(Model, self).__init__()
        self.convolutional = convolutional
        embed_size = 16
        bidir_size = 8
        window = 31

        self.embedding = nn.Embedding(21, embed_size)

        if convolutional:
            self.conv1 = nn.Conv1d(embed_size, 32, kernel_size=7)
            self.conv2 = nn.Conv1d(32, 16, kernel_size=5)
            self.conv3 = nn.Conv1d(16, 8, kernel_size=3)
            # window shrinks by (k-1) per conv: 31→25→21→19
            flat_size = (window - 12) * 8
        else:
            self.lstm = nn.LSTM(embed_size, bidir_size,
                                batch_first=True, bidirectional=True)
            flat_size = window * bidir_size * 2

        self.fc1 = nn.Linear(flat_size, 16)
        self.fc2 = nn.Linear(16, 3)

    def forward(self, x):
        x = self.embedding(x)                  # (batch, window, embed_size)

        if self.convolutional:
            x = x.permute(0, 2, 1)             # (batch, embed_size, window)
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = torch.relu(self.conv3(x))
        else:
            x, _ = self.lstm(x)                # (batch, window, bidir_size*2)

        x = x.reshape(x.size(0), -1)           # flatten
        x = self.fc1(x)
        x = self.fc2(x)                        # raw logits
        return x

model = Model(convolutional=True)
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Look at the architecture above:
* What does putting the variable "convolutional" to False mean? What happens to the `LSTM` layers when we are using a convolutional architecture?
* Which architecture would be best for this type of dataset in your opinion?
    
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}

class ProteinWindowDataset(Dataset):
    def __init__(self, windows, labels):
        self.windows = windows
        self.labels = np.asarray(labels)
        self.length = windows.shape[0]

    def __len__(self):
        return self.length

    def __getitem__(self, idx):
        window = self.windows[idx]
        label  = self.labels[idx]
        return (
            torch.tensor(window, dtype=torch.long),
            torch.tensor(label,  dtype=torch.long),
        )


def get_dataloader(X, y, target_list_data, batch_size, shuffle=False):
    datasets = []
    for target in target_list_data:
        if target in X:
            datasets.append(
                ProteinWindowDataset(X[target], y[target])
            )

    combined = ConcatDataset(datasets)

    dataloader = DataLoader(
        combined,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=False,          # keep the last (smaller) batch
    )
    return dataloader
```

Now let's load the dataset as a pickle object:

```python editable=true slideshow={"slide_type": ""}
import pickle
(X,y) = pickle.load(open("data/dataset_sseq_singleseq.pic",'rb'))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
Ok, let's start by taking the classical approach of randomly splitting the data in a trainset and a validation set (95%/5% by default, but you can change the ratio as you prefer).
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
epochs = 20
batch_size = 8

target_list_path = 'data/train_set'

target_list_file = open(target_list_path)
target_list = target_list_file.readlines()
random.shuffle(target_list)

n_targets = len(target_list)
train_list = target_list[int(n_targets/20):] #95% train
dev_list = target_list[:int(n_targets/20)] #5% validation

train_loader = get_dataloader(X, y, train_list, batch_size, shuffle=True)
dev_loader = get_dataloader(X, y, dev_list, batch_size, shuffle=False)

# define optimizer and loss function
learning_rate = 1e-3
weight_decay = 1e-5
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
criterion = nn.CrossEntropyLoss()
accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=3, top_k=1)

# Setup plot
liveplot = LivePlot()
liveplot.increment(epochs)

train(model=model, 
      train_loader=train_loader, 
      dev_loader=dev_loader, 
      optimizer=optimizer, 
      criterion=criterion,
      metric=accuracy,
      max_epochs=epochs, 
      liveplot=liveplot,
      device=torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu'))
```

<!-- #region editable=true slideshow={"slide_type": ""} -->
* What is the best validation performance that you can extract from your Model?
* What would be the best naïve classifier for this dataset? How does the validation performance of your model compare to it?
* What do you think of randomly splitting the dataset this way? Can you think of a better way of doing it? Can you think of a _worse_ day of doing it?
<!-- #endregion -->

## Splitting the dataset by sequence similarity

<!-- #region editable=true slideshow={"slide_type": ""} -->
I have used HHblits (a software to perform sequence alignments) to find out just how distant the proteins in the dataset are, evolutionarily speaking. This distance goes from 0 (sequences are identical) to 1 (no relationship between the proteins could be detected at all). The distance is basically an inverse measure of how similar the sequences are to each other.

This information is stored in a distance matrix of size NxN, where N is the number of sequences in the dataset. In the code block below I load the distance matrix from the filesystem, then we use the data to perform [linkage clustering](https://docs.scipy.org/doc/scipy/reference/generated/scipy.cluster.hierarchy.linkage.html) and plot a [dendrogram](https://en.wikipedia.org/wiki/Dendrogram) to visualize the clusters.

In the dendrogram below we can see how proteins group together at various distance thresholds.
<!-- #endregion -->

```python editable=true slideshow={"slide_type": ""}
import matplotlib.pyplot as plt
sys.setrecursionlimit(100000) #fixes issue with scipy and recursion limit
plt.rcParams['figure.figsize'] = [12, 8]
plt.rcParams['figure.dpi'] = 100
distance_matrix = pd.read_csv('data/trainset_distance_matrix.tsv', sep='\t')
dists = squareform(distance_matrix)
linkage_matrix = linkage(dists, "single")
dendrogram(linkage_matrix, color_threshold=0.8)
plt.show()
```

Below, we choose a threshold to get our cluster based on the distance threshold t. So we "cut" the dendrogram above at the threshold t, and all the proteins that fall under the same branch at that threshold will be put in the same cluster. Feel free to get a feeling of how clusters are formed/split by varying the threshold below:

```python
cluster_assignments = fcluster(linkage_matrix,criterion='distance', t=0.8)
print(len(cluster_assignments), np.max(cluster_assignments))
```

Now let's create a training and a validation set based on these clusters in such a way that a cluster of protein is EITHER in train OR in validation. Depending on the threshold we have picked, this could make sure that no proteins in the validation set are too similar to those in the trainset.

```python
target_list_file = open(target_list_path)
target_list = target_list_file.readlines()

train_list_cluster = []
validation_list_cluster = []
validation_size_limit = int(n_targets/20)

for i in range(1,np.max(cluster_assignments)+1):
    index_this_cluster = np.where(cluster_assignments == i)[0]
    if len(validation_list_cluster) < validation_size_limit: #add all elements in this cluster to either validation or train set
        validation_list_cluster += [target_list[element] for element in index_this_cluster]
    else:
        train_list_cluster += [target_list[element] for element in index_this_cluster]

random.shuffle(train_list_cluster)
validation_steps_cluster = count_steps(validation_list_cluster, batch)
print("Validation batches:", validation_steps_cluster)
```

Now, let's train a new model with the new datasets and see if we get different results:

```python


model_sseq2 = get_model(convolutional=False) #get a fresh model

hist2 = model_sseq2.fit(generate_inputs_window(X,y,train_list_cluster, batch), 
               validation_data=generate_inputs_window(X,y,validation_list_cluster, batch), 
               epochs=40, steps_per_epoch=1000, validation_steps=validation_steps_cluster)

```

Let's plot again the training curves from the first model and compare them to those from the new model. 

What are the differences, if any?

```python
plot_loss_acc(hist)
```

```python
plot_loss_acc(hist2)
```

Now let's test the two models on previously unseen data. Which performs best?

```python
test_list = open("data/test_set").readlines()
test_steps = count_steps(test_list, batch)
print("Test steps:", test_steps)
res1 = model_sseq.evaluate(generate_inputs_window(X,y,test_list, batch), verbose=1, steps=test_steps)
res2 = model_sseq2.evaluate(generate_inputs_window(X,y,test_list, batch), verbose=1, steps=test_steps)
print(f"Model 1 test acc: {res1[1]}, Model 2 test accuracy: {res2[1]}")
```

## If you have extra time and want to play more with the data


Now let's make things even worse on purpose: whenever a cluster contains more than one sample, let's put half in the training set and half in the validation set. Then let's not shuffle the trainset so that the network sees those samples first.

```python
target_list_file = open(target_list_path)
target_list = target_list_file.readlines()

train_list_bad = []
validation_list_bad = []
validation_size_limit = int(n_targets/20)

for i in range(1,np.max(cluster_assignments)+1):
    index_this_cluster = np.where(cluster_assignments == i)[0]

    if len(index_this_cluster) > 1: #add all elements in this cluster to either validation or train set
        half_elements = int(len(index_this_cluster)/2)
        validation_list_bad += [target_list[element] for element in index_this_cluster[:half_elements]]
        train_list_bad += [target_list[element] for element in index_this_cluster[half_elements:]]
    
validation_steps_bad = count_steps(validation_list_bad, batch)
print("Validation batches:", validation_steps_bad)
```

```python
model_sseq3 = get_model(convolutional=True, window=window) #get a fresh model

hist3 = model_sseq3.fit(generate_inputs_window(X,y,train_list_bad, batch), 
               validation_data=generate_inputs_window(X,y,validation_list_bad, batch), 
               epochs=40, steps_per_epoch=1000, validation_steps=validation_steps_bad)

```

```python
plot_loss_acc(hist)
plot_loss_acc(hist2)
plot_loss_acc(hist3)
```
