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
# Recall that inputs x have a weight w and a bias term b attached to them in 
# the perceptron model. Clearly w implies how much weight or strength to 
# give the incoming input. We can think of b as an offset value, making
# x*w have to reach a certain threshold before having an effect.

# For example if b = -10, then the effects of x*w won't really start to 
# overcome the bias until their product surpasses 10. After that, then the 
# effect is solely based on the value of w. Thus the term bias.

# Next we want to set boundaries for the overall output value of x*w+b. We 
# can state z = xw + b and then pass z through some activation function
# to limit it's value.

# A lot of research has been done into activation functions and their  
# effectiveness. Let's explore some common activation functions. Recall
# our simple perceptron has an f(x) (see notes_5). If we had a binary 
# classification problem, we would want an output of either 0 or 1.

# To avoid confusion, let's define the total inputs as a variable z, where
# z = wx + b. In this context, we'll then refer to activation functions
# as f(z). Keep in mind, you will often see these variables capitalized  
# f(Z) or X to denote a tensor input consisting of multiple values.

# The most simple networks rely on a basic step function that outputs 0 
# or 1 (see notes_8). Regardless of the values, this always outputs
# 0 or 1. This sort of function could be useful for classification (0 or 
# 1 class). However this is a very "strong" function, since small changes
# aren't reflected. There is just an immediate cut off that splits 
# between 0 and 1.

# I would be nice if we could have a more dynamic function, for example
# the red line (see notes_9). This is the sigmoid function. Changing the 
# activation function can be beneficial depending on the task.

# This still works for classification, and will be more sensitive to small 
# changes. Using the sigmoid function allows us to receive a probability
# instead of just an output of 0 or 1.

# Some other activation functions
# Hyperbolic Tangent: tanh(z) (see notes_10)
# - Outputs between -1 and 1 instead of 0 and 1

# Rectified Linear Unit (ReLU): (see notes_11)
# - This is actually a relatively simple function max(0,z)
# - If the output of the value is less than 0, we treat it as 0. Otherwise,
#   if it's greater than 0, we go ahead and output the actual z value 
# - ReLU has been found to have very good performance, especially when 
#   dealing with the issue of vanishing gradient. 
# - We'll often default to ReLU due to it's overall good performance

# For a full list of activation functions, check out 
# https://en.wikipedia.org/wiki/Activation_function


# Multi-Class Activation Functions 
# Notice all these activation functions make sense for a single output,
# either a continuous label or trying to predict a binary classification
# (either a 0 or a 1), but what should we do if we have a multi-class 
# situation?

# There are 2 main types of multi-class situations: 
# - Non-Exclusive Classes
#   - A data point can have multiple classes/categories assigned to it
#     - Photos can have multiple tags (eg beach, family, vacation, etc)
# - Mutually Exclusive Classes
#   - A data point can only have one class/category assigned to it
#     - Photos can be categorized as being in grayscale or full color, but
#       cannot be both
#   - the more common of these two 

# Organizing multiple classes 
# The easiest way to organize multiple classes is to simply have 1 output
# node per class. Previously we thought of the last output layer as a single 
# node. That single noe could output a continuous regression value or binary
# classification (0 or 1). Let's expand this output layer to work for the 
# case of multi-classification. (see notes_12) So consider an input layer,
# your hidden layers, and then an output layer where the output layer is 
# essentially one neuron per class. This means we will need to organize 
# categories for this output layer. We're not going to be able to feed
# our neural network with a string like "red", "blue", "green", etc. Recall
# that the neural network is going to take in x values and those should be
# numbers so that we can apply weights to them and a biases (you can't
# multiply a word like "cat"). We're going to need to transform our data
# in order for it to be processed correctly by neural networks.

# Instead we use one-hot encoding (dummy variables). Let's take a look
# at what this looks like with mutually exclusive classes. So for mutually 
# exclusive classes you have data points and a class that belongs to each
# data point (ex see notes_13). Because we can't feed these string color
# codes into our neural network we have to find another way to input the
# values. What we can do is just use binary classification for each class
# and what we do is end up building a matrix (see notes_14). This is one
# hot encoding, sometimes called creating dummy variables.

# For non-exclusive classes it's going to be slightly different. Recall that
# each data point can actually have multiple classes assigned to it (see
# notes_15). 

# The term one hot encoding is in reference to the idea that 1 is on and 0 
# is off.

# Now that we've got our data correctly organized, we just need to choose 
# the correct classification activation function that the last output
# layer should have.

# Non-exclusive 
# - Sigmoid function
#   - Each neuron will output a value between 0 and 1, indicating the 
#     probability of having that class assigned to it (see notes_16)
#     - Let's say that we set the cutoff at 0.5, in the notes_16 example
#       the output would have two classes assigned to it (both class 1 and
#       class 2)
#   - Keep in mind this allows each neuron to output independent of the 
#     other classes, allowing for a single data point fed into the function
#     to have multiple classes assigned to it.

# Mutually Exclusive Classes 
# What do we do when each data point can only have a single class assigned
# to it? We can use the softmax function for this (see notes_17). K is the 
# number of categories. Softmax function calculates the probabilities 
# distribution of the event over K different events. This function will 
# calculate the probabilities of each target class over all possible target 
# classes. The range will be 0 to 1, and the sum of all probabilities will 
# be equal to 1. The model returns the probabilities of each class and the 
# target class chosen will have the highest probability. 

# The main thing to keep in mind is that if you use the softmax for
# multi-class problems you get this sort of output:
# [Red, Green, Blue]
# [0.1,  0.6,   0.3]
# The sum of all probabilities add to 1 because there is a 100% that the 
# output belongs to one of these categories


# Cost Functions and Gradient Descent
# Cost functions measure how far off we are in the predictions of our
# neural networks. Gradient Descent which is going to help us minimize 
# that cost/error. 

# We now understand that neural networks take in inputs, multiply them by 
# weights, and add biases to them. Then this result is passed through an
# activation function which at the end of all the layers leads to some
# output. This output y^ is the model's estimation of what it predicts the 
# label to be. So after the network creates it's prediction, how do we 
# evaluate it? And after the evaluation how can we update the network's
# weights and biases?

# We need to take the estimated outputs of the network and then compare 
# them to the real values of the label. Keep in mind this is using the 
# training data set during the fitting/training of the model.

# The cost function (often referred to as a loss function) must be an 
# average so it can output a single value. We can keep track of our 
# loss/cost during the training to monitor network performance.

# We'll use the following variables: 
# - y to represent the true value
# - a to represent neuron's prediction
# In terms of weights and bias:
# - wx + b = z
# Pass z into activation function o(Z) = a

# One very common cost function is the quadratic cost function (see 
# notes_18) a^L(x) is essentially the predicted output, L is 
# essentially the output layer. We simply calculate the difference between 
# the real values y(x) against our predicted values a(x.)

# Note: The notation shown here corresponds to vector inputs and outputs 
# since we will be dealing with a batch of training points and predictions. 
# Notice that squaring this does 2 useful things for us, keeps everything
# positive and punishes large errors.

# We can think of the cost function as a function of 4 main things:
# C(W, B, S^r, E^r)
# W is our neural network's weights, B is our neural network's biases, S^r
# is the input of a single training sample, and E^r is the desired output
# of that training sample.

# Notice how that information was all encoded in our simplified notation. 
# The a(x) holds information about weights and biases. a(x) is z after
# having been passed through the activation function.

# This means that if we have a huge network, we can expect C (the cost 
# function) to be quite complex, with huge vectors of weights and biases.
# Here (see notes_19) is a small network with all it's parameters labeled.
# That is a lot to calculate. How do we solve this? How do we calculate 
# this cost function and figure out how to minimize it?

# In a real case, this means we have some cost function C dependent on lots
# of weights (C(w1,w2,w3,...wn)) How do we figure out which weights lead us
# to the lowest cost? 

# For simplicity, let's imagine we only had one weight in our cost function 
# w. We want to minimize our loss/cost (overall error), which means we need
# to figure out what value of w results in the minimum of C(w). Here (see
# notes_20) is an example of our "simple" function C(w). What we want to 
# do is to figure out what value of w minimizes our cost function. Students
# of calculus know that we could just take the derivative of this function
# and solve for 0, but realize our real cost function will be very complex
# (see notes_21), it's going be n-dimensional, having as many dimensions as 
# there are weights. It would take far too long to take the derivative and 
# solve for every dimension in a real world problem. So instead we use a
# stochatric proccess and use gradient descent to solve this problem. 
# Let's consider again the simplified cost function (notes_21) to see how
# this works. Let's start off with a single point on this cost function and
# calculate the slope at this point (notes_22). Then we move downward in the  
# direction of the slope. This process is repeated until the slope converges
# to zero (the minimum value is discovered) (notes_23). Keep in mind that 
# we can change the size of the steps taken in trying to obtain the minimum.
# Smaller step sizes increase the amount of time it takes to find the 
# minimum, but should you use steps that are too large that value may be 
# overshot (notes_24). The step size is known as the learning rate.

# The learning rate just depicted was constant, each step size was equal,  
# but we can be clever and adapt our step size as we go along. We could
# start with larger steps, then go smaller as we realize the slope gets 
# closer to zero. This is known as adaptive gradient descent.

# The Adam Optimizer
# In 2015, Kingma and Ba published their paper: "Adam: A Method for 
# Stochastic Optimization". Adam is a much more efficient way of searching
# for these minimums, so we will apply it to our code. Adam vs other 
# gradient descent algorithms (see notes_25) notice how it outperforms
# all others. Realistically we're calculating this descent in an 
# n-dimensional space for all our weights (illustrated for 3 dimensions in
# notes_26). When dealing with these n-dimensional vectors(tensors), the 
# notation changes from derivative to gradient(notes_27).

# For classification problems, we often use the cross entropy loss function
# The assumption is that your model predicts a probability distribution
# p(y=i) for each class i = 1,2,...,C (notes_28).


# Backpropagation 
# Fundamentally, we want to know how the cost function results changes with 
# respect to the weights in the network, so we can update the weights to
# minimize the cost function. Let's begin with a very simple network where 
# each layer only has 1 neuron. Each input will receive a weight and 
# bias (notes_29). This means we have C(w1,b1,w2,b2,w3,b3). We've already
# seen how this process propagates forward. Let's start at the end to see 
# the backpropagation. Let's say we have L layers, then our notation
# becomes (see notes_30). Focusing on these last two layer, let's define
# z = wx + b. Then applying an activation function we'll state: a = o(z).
# This means we have z^L = w^L * a^L-1 + b^L. This means that z of the
# last layer is equal to (w of the last layer times a of the second to last
# layer) plus b of the last layer. Also a^L = o(z^L), this means the 
# activation output at the last layer is equal to the sigmoid(the activation
# function) of z at the last layer. That means then that 
# C_0(...) = (a^L - y)^2 or the cost function is equal to a^L minus y (where
# y is the actual true output) squared. 

# We want to understand how sensitive is the cost function to changes in w. 
# This is where partial derivatives come into play. We want to find out the
# relationship between the final cost function and weights at the layer L.
# So we're going to say take the partial derivative using the chain rule of 
# that cost function with respect to weights and layer L (notes_31). Recall
# that the cost function is not just a function of the weights, but it is 
# also of the biases. So we need to understand the relationship of the cost 
# function to both the weights and the biases as well. We can calculate
# the bias terms (notes_32). The main idea here is that we can use the
# gradient to go back through the network and adjust our weights and biases
# to minimize the output of the error vector on the last output layer.
# Using some calculus notation, we can expand this idea to networks with 
# multiple neurons per layer. Hadamard Product (notes_33) element by element
# product.

# Given this notation and backpropagation, we have a few main steps to 
# training neural networks. 
# Step 1: Using input x set the activation function a for the input layer
# (z = wx + b and a = o(z)). This resulting a then feeds into the next 
# layer and so on.
# Step 2: For each layer, compute:
# - z^L = (w^L * a^L-1) + b 
# - a^L = o(z^L)
# Step 3: Compute the error vector (notes_34)
# - the first term (gradient_a C) expresses the rate of change of C with  
#   respect to the output activations 
# - essentially this error vector translates to the rate of change of C 
#   equals the activation function of the last layer minus the actual 
#   output (a^L - y) (notes_35) so replacing the first term we 
#   get (notes_36)
# - Now let's write out our error term for a layer in terms of error of 
#   the next layer (since we're moving backwards)     
# Step 4: Backpropagate the error.
# - For each layer we compute (note the lower case l denoting that this is
#   dealing with layers that are not the output layer (L))
#   - (notes_37)
#     - (w^(l+1))^T is the transpose of the weight of matrix of l+1 layer
# - This is the generalized error for any layer l
# - When we apply the transpose weight matrix we can think intuitively
#   of this as moving the error backwards through the network, giving us
#   some sort of measure of the error at the output of the lth layer
# - We then take the Hadamard product (notes_38). This moves the error
#   backwards through the activation function in layer l, giving us the error
#   in l in the weighted input to layer l.

# The gradient of the cost function is given by (notes_39). This then allows
# us to adjust the weights and biases to help minimize that cost function


