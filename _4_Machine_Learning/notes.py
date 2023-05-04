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




























