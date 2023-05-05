import pandas as pd

# Supervised learning algorithms are trained using labeled examples, such as an input
# where the desired output is known

# The network receives a set of inputs along with the corresponding correct outputs, 
# and the algorithm learns by comparing its actual output with correct outputs to find 
# errors; it then modifies the model accordingly


# Supervised learning is commonly used in applications where historical data predicts
# likely future events

# Data is often split into 3 sets: 
# - training - used to train model parameters
# - validation - used to determine what model hyperparameters to adjust
# - test - used to get some final performance metrics

# This means after we see the results on the final test set we don't get to 
# go back and adjust any model parameters!

# This final measure is what we label the true performance of the model to 
# be


# Overfitting and Underfitting
# Overfitting 
# - the model fits too much to the noise from the data
# - this often results in low error on training sets but high error on 
#   test/validation sets
# see notes_1 for graphic example of overfit model loss curve
# - a good indication that you've trained the model too much to the data
#   set (too many epochs)
# - in case like notes_1 you may want to drop the number of epochs down
#   to the point where the lines intersect (notes_2)

# Underfitting
# - model does not capture the underlying trend of the data and does not
#   fit the data well enough
# - low variance but high bias
# - underfitting is often a result of an excessively simple model

# When thinking about overfitting and underfitting we want to keep in mind
# the relationship of model perfomance on the training set versus the 
# test/validation set


# Model Evaluation
# The key classification metrics we need to understand are:
# - Accuracy
# - Recall
# - Precision
# - F1-Score

# Typically in any classification task your model can only achieve two results
# - Either your model was correct in it's prediction
# - Or your model was incorrect in it's prediction

# Fortunately incorrect vs correct expands to situations where you have multiple
# classes. For the purposes of explaining the metrics, let's imagine a binary
# classification situation, where we only have two available classes.

# In our example, we will attempt to predict if an image is a dog or a cat. 
# Since this is supervised learning, we will first fit/train a model on 
# training data, then test the model on testing data. Once we have the 
# model's predictions from the X_test data, we compare it to the true y
# values (the correct labels)


# Accuracy
# Accuracy in classification problems is the number of correct predictions made 
# by the model divided by the total number of predictions
# For example, if the x_test set was 100 images and our model correctly
# predicted 80 images, then the accuracy is 0.8 or 80%
# Accuracy is useful when target classes are well balanced
# In our example, we would have roughly the same amound of cat images as we 
# have dog images
# Accuracy is not a good choice with unbalanced classes. Imagine we had 99
# images of dogs and 1 image of a cat. If our model was simply a line that 
# always predicted dog we would get 99% accuracy. In this situation we'll
# want to understand recall and precision


# Recall
# Recall is the ability of a model to find all the relevant cases within a 
# dataset. The precise definition of recall is the number of true positives
# divided by the number of true positives plus the number of false negatives


# Precision
# Precision is the ability of a classification model to identify only the 
# relevant data points. Precision is defined as the number of true positives
# divided by the number of true positives plus the number of false positives


# Recall and Precision
# Often you have a trade-off between recall and precision. While recall 
# expresses the ability to find all relevant instances in a dataset, precision
# expresses the proportion of the data points our model says was relevant
# that actually were relevant


# F1-Score
# In cases where we want to find an optimal blend of precision and recall we
# can combine the two metrics using what is called the F1-Score.

# The F1 score is the harmonic mean of precision and recall taking both 
# metrics into account in the following equation:

# F_1 = 2 * ((precision * recall) / (precision + recall))

# The reason we use the harmonic mean instead of a simple average is because
# it punishes extreme values. A classifier with a precision of 1.0 and a 
# recall of 0.0 has a simple average of 0.5 but an F1-Score of 0.

# We can also view all correctly classified versus incorrectly classified 
# images in the form of a confusion matrix


#                        _________________________________________________
#                        |                                               |
#                        |           Prediction Condition                |
#           _____________|_______________________________________________|
#           |            |                       |                       |
#           | Total      |  Prediction Positive  |  Prediction Negative  |
#           | Population |                       |                       |
# __________|____________|_______________________|_______________________|
#           |            |                       |                       |
#           | Condition  |  True Positive (TP)   |  False Negative (FN)  |
#           | Positve    |                       |    (Type II Error)    |
# True      |____________|_______________________|_______________________|
# Condition |            |                       |                       |
#           | Condition  |  False Positive (FP)  |  True Negative (TN)   |
#           | Negative   |   (Type I Error)      |                       |
# __________|____________|_______________________|_______________________|

# (see notes_3 for expanded chart)

# The main point to remember with the confusion matrix and the various 
# calculated metrics is that they are all fundamentally ways of comparing
# the predicted values versus the true values. What constitutes "good" metrics
# will really depend on the specific situation.


# Often we have a precision/recall trade off. We need to decide if the model 
# will focus on fixing false positives vs false negatives. In disease 
# diagnosis, it is probably better to go in the direction of false positives, 
# so we make sure to classify as many cases of disease as possible.

# All of this is to say, machine learning is not performed in a vacuum, but 
# instead a collaborative process where we should consult with experts in the
# domain (e.g. medical doctors)


# Evaluating Performance
# Regression 
# Regression is a task when a model attempts to predict continuous values
# (unlike categorical values, which is classification). In regression 
# problems evaluation metrics like accuracy or recall are not useful. We 
# need metrics designed for continuous values. For example, attempting to 
# predict the price of a house given it's features is a regression task. 
# Attempting to predict the country a house is in given it's features would
# be a classification task.

# The most common evaluation metrics for regression are:
# - Mean Absolute Error
#   - This is the mean of the absolute value of errors
#   - The issue with MAE is that it won't punish large errors 
# - Mean Squared Error 
#   - This is the mean of the squared errors
#   - Larger errors are noted more than with MAE, making MSE more popular 
# - Root Mean Square Error
#   - This is the root of the mean of the squared errors
#   - Most popular
#   - has same units as y 

# Most common question is 'Is this RMSE good?' Context is everything. A 
# RMSE of $10 is fantastice while predicting the price of a house, but 
# horrible for predicting the price of a candy bar.

# You should compar your error metric to the average value of the label in
# your data set to try to get an intuition of it's overall performance.
# Domain knowledge also plays an important role here.


# Unsupervised Learning 
# We've covered supervised learning, where the label was known due to 
# historical labeled data, but what happens when we don't have historical
# labels? This is unsupervised learning. There are certain tasks that fall
# under the umbrella of unsupervised learning:
# - Clustering
#   - Grouping together unlabeled data points into categories/clusters
#   - Data points are assigned to a cluster based on similarity
#   - Keep in mind, since this is unsupervised data we may end up with an 
#     approach that doesn't actually correspond with the labels we had in 
#     mind
# - Anomaly Detection
#   - Attempts to detect outliers in a dataset
#     - ex: fraudulent transactions on a credit card 
# - Dimensionality Reduction
#   - Data processing techniques that reduces the number of features in a 
#     data set, either for compression, or to better understand the 
#     underlying trends within a data set. 

# It is important to note, these are situations where we don't have the  
# correct answer for historical data. This means the evaluation is much 
# harder and more nuanced.


# Understanding Artificial Neural Networks
# A large part of this section will focus on the theory behind many of the 
# ideas we will implement with code. Let's do a quick review of how we will
# gradually build an understanding of artificial neural networks.


# Perceptron Model 
# The whole idea behind deep learning is to have computers artificially 
# mimic biological natural intelligence. 

# A perceptron was a form of neural network introduced in 1958 by Frank 
# Rosenblatt. Amazingly, even back then he saw huge potential:
# "...perceptron may eventually be able to learn, make decisions, and 
# translate languages."
# However, in 1969 Marvin Minsky and Seymour Papert's published their book
# Perceptrons. It suggested that there were severe limitations to what 
# perceptrons could do. This marked the beginning of what is known as the 
# AI Winter, with little funding into AI and Neural Networks in the 1970's.
# Fortunately for us, we now know the amazing power of neural networks, 
# which all stem from the simple perceptron model, so let's head back and  
# convert our simple biological neuron model into the perceptron model
# (see notes_4). 
# If f(x) is just a sum, then y = x1 + x2. Realistically, we would want to 
# be able to adjust some parameter in order to "learn". Let's add an 
# adjustable weight we multiply against x (see notes_5). Now
# y = x1w1 + x2w2. Now we could update the weights to effect y. Let's add
# in a bias term b to the inputs (see notes_6). y = (x1w1 + b) + (x2w2 + b).
# Later we will see how we can expand this model to have x be a tensor of
# information (an n-dimensional matrix). Also we'll see we can even simplify
# the bias to be at a layer level instead of a bias per input: y = wx + b.


# Neural Networks 
# A single perceptron won't be enough to learn complicated systems. 
# Fortunately, we can expand on the idea of a single perceptron, to create
# a multi-layer perceptron model. We'll also introduce the idea of 
# activation functions.

# To build a network of perceptrons, we can connect layers of perceptrons,  
# using a multi-layer perceptron model. The outputs of one perceptron are
# directly fed into as inputs to another perceptron(neuron). This allows 
# the network as a whole to learn about interactions and relationships  
# between features.

# The first layer is the input layer. This is the layer that directly  
# receives the data. 

# The last layer is the output layer (this can be more than one neuron). 

# Layers in between the input and output layers are hidden layers. Hidden
# layers are difficult to interpret, due to their high interconnectivity 
# and distance away from known input and output values. 

# Neural networks become "deep neural networks" if they contain 2 or more
# hidden layers (see notes_7).

# Terminology: 
#  - Input Layer:
#    - First layer that directly accepts real data
# - Hidden Layer:
#   - Any layer between input and output layers
# - Output Layer: 
#   - The final estimate of the output.

# What is incredible about the neural network framework is that it can be 
# used to a approximate any function. Zhou Lu and later on Boris Hanin 
# proved mathematically that neural networks can approximate any convex
# continuous function.

# For more details on this check out:  
# Universal approximation theorem
# https://en.wikipedia.org/wiki/Universal_approximation_theorem

# Previously in our simple model we saw that the perceptron itself contained
# a very simple summation function f(x). For most cases however that won't 
# be useful, we'll want to be able to set constraints to our output values,
# especially in classification tasks.

# In classification tasks, it would be useful to have all outputs fall 
# between 0 and 1. These values can then present probability assignments
# for each class. Next we'll explore how to use activation functions to set
# boundaries to output values from the neuron. 

# Activation Functions 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 
# 










