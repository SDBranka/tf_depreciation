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







