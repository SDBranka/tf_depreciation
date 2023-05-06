import pandas

# The differences between TensorFlow and Keras
# TensorFlow is an open-source deep learning library developed by Google
# with TF 2.0 being officially released in 2019. TF has a large ecosystem
# of related components, including libraries like Tensorboard, Deployment
# and production APIs, and support for various programming languages.

# Keras is a high-level python library that can use a variety of deep 
# learning libraries underneath, such as TensorFlow, CNTK, or Theano.

# TF 1.x had a complex python class system for building models, and due to  
# the huge popularity of Keras, when tf2.0 was released, tf adopted keras
# as the official API for tf

# While Keras still also remains as a separate library from tf, it can  
# also now officially be imported through tf, so there is now no need to 
# additionally install it.

# The Keras API is easy to use and builds models by simply adding layers 
# on top of each other through simple calls


# Choosing an optimizer and loss 
# keep in mind what type of problem you are trying to solve
# as many optimizers can be used for any case, the thing to note in these
# examples is the loss function selected for each problem type
# for a multi-class classification problem
model.compile(optimizer="rmsprop",
            loss="categorical_crossentropy",
            metrics=["accuracy"]
            )

# for a binary classification problem 
model.compile(optimize="rmsprop",
            loss="binary_crossentropy",
            metrics=["accuracy"]
            )

# for a mean squared error regression problem  
model.compile(optimizer="rmsprop",
            loss="mse"
            )

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
