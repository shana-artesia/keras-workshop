library(keras)
# Use this to limit cpu and provide reproducible outputs
use_session_with_seed(1234) #ignore this line normally - for the workshop only to ensure enough server power for all users.
# This sets the random initialisation in Python. This will ensure the same output each time. If you use set.seed, this will 
# only set the seed in R, not the other packages.
# This cannot be used in parallel coding. Only single core. 

library(rsample) # Tidyverse equivalent of modelling - consistent language

# Use iris data set as an example
# Create training and test data splits
data_split <- initial_split(iris, strata = "Species",prop = 0.8) # 80% training, 20% test set.

# training = analysis, testing = assessment (in the materials)
full_data <- list(train = training(data_split), test=testing(data_split))

library(recipes) # now available on CRAN
# Package for transforming the data to get ready for modelling. 

empty_recipe <- recipe(Species ~ . , data=full_data$train)

# Turn named categories into dummy variables

dummy <- empty_recipe %>% 
  # Tell the recipe that we need to convert the response variable into a dummy variable. 
  step_dummy(Species, one_hot = TRUE, role = "outcome")

dummy %>% 
  prep() %>% 
  bake(full_data$train)

# centre and scale data. If the original data is on completely different scales, Keras will overweight the variables. It is VERY important to scale and centre. 
centered = dummy %>% 
  step_center(all_predictors()) %>% 
  step_scale(all_predictors())

centered %>% prep() %>% 
  bake(full_data$train) %>% View()

# Other steps that we may need in 'real' data, is handling missing values.
# Keras does not handle NAs.
# Could;
# Convert NAs to 0s,
# Remove them,
# More details in the materials

iris_recipe <- centered %>% 
  prep() # create an object to do the prep. mean and sd is stored. 

# Create matrix outputs from this.
library(purrr)
xIris <- map(full_data, bake, object=iris_recipe,
             all_predictors(), composition="matrix") # map function will apply to all elements of a list.

yIris <- map(full_data, bake, object=iris_recipe,
             all_outcomes(), composition="matrix") # map function will apply to all elements of a list.


# Bake function uses the same means and variances from the training data, apply the same values to the test set.


############################## Exercise

library(mlbench)
data("BreastCancer")

# Split data
breast_split <- initial_split(BreastCancer, strata = "Class",prop = 0.8) # 80% training, 20% test set.

# training = analysis, testing = assessment (in the materials)
full_breast <- list(train = training(breast_split), test=testing(breast_split))
# Remove ID column
full_breast <- lapply(full_breast, function(x) { x["Id"] <- NULL; x })

# Create dummy variables
empty_breast_recipe <- recipe(Class ~ . , data=full_breast$train)

# Turn named categories into dummy variables
dummy_breast <- empty_breast_recipe %>% 
  # Tell the recipe that we need to convert the response variable into a dummy variable. 
  step_dummy(Class, one_hot = TRUE, role = "outcome")

dummy_breast %>% 
  prep() %>% 
  bake(full_breast$train)

# centre and scale data. If the original data is on completely different scales, Keras will overweight the variables. It is VERY important to scale and centre. 
centered_breast = dummy_breast %>% 
  step_center(all_numeric()) %>% 
  step_scale(all_numeric())

centered_breast %>% prep() %>% 
  bake(full_breast$train) %>% View()

# Other steps that we may need in 'real' data, is handling missing values.
# Keras does not handle NAs.
# Could;
# Convert NAs to 0s,
# Remove them,
# More details in the materials

breast_recipe <- centered_breast %>% 
  prep() # create an object to do the prep. mean and sd is stored. 

# Create matrix outputs from this.
library(purrr)
xBreast <- map(full_breast, bake, object=breast_recipe,
               all_numeric(), composition="matrix") # map function will apply to all elements of a list.

yBreast <- map(full_breast, bake, object=breast_recipe,
               all_outcomes(), composition="matrix") # map function will apply to all elements of a list.

#####################################################



##################################################### WORKSHOP
library(keras)

model <- keras_model_sequential() # This model will change in place!!!! Will trip us up. BE CAREFUL
# creates empty model. Requires layers.

# Firstly, add a 'dense' layer. Think of this like new columns of data. 

model %>% layer_dense(units=10, input_shape = 4) %>% 
  # Essentially, create 10 new 'columns' from the previous 4 using weighted sums.
  # the input shape is essentially the size of the data. Need to tell it how many columns it has. Don't want to restrict rows (so 
  # we don't tell it this information, just the number of rows.)
  layer_dense(units=3, activation = "softmax")
# Add a layer for the ouput. In this example, there are 3 types of species hence, the 3 units.

model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy") 
# Compile: How should the model be fit? Optimizer - optimisation functions
# Loss function - how the algorithm should update the weights to make the difference smaller.
# Metrics - depends on the problem, standard 'accuracy'

history <- model %>% 
  fit(xIris$train,
      yIris$train,
      epochs = 100, #training numbers
      validation_data = list(xIris$test, yIris$test)) # Should use a validation set NOT the test set..
# Pass the data we want to train on (predictors and outcome separately), tell it about any validations, as well as number of times.

plot(history)

# Re-running the code MAY give different answers each time. This is because the initialisation is 'random'. The back propagation
# will also contain an element of randomness. If you don't set the seed, then this will occur. 

#### Evaluating the model

model %>% evaluate(xIris$test, yIris$test)
# Loss and accuracy for test data in the model.

# predict classes for test data

model %>% predict(xIris$test)
# 3 columns - probability for each row of data in each group. 

model %>% predict_classes(xIris$test) # since the classes come from Python, the classes now start from 0, not 1. 


##################################################### EXERCISE CONTINUED..

xBC <- readRDS("/data/BreastCancerCleanFeatures.rds")
yBC <- readRDS("/data/BreastCancerCleanTarget.rds")

model_b <- keras_model_sequential()

model_b %>% layer_dense(units=5, input_shape = 9) %>% 
  layer_dense(units=2, activation = "sigmoid")

model_b %>% compile(optimizer = "rmsprop",
                    loss = "binary_crossentropy",
                    metrics = "accuracy") 

history_b <- model_b %>% 
  fit(xBC$train,
      yBC$train,
      epochs = 20, #training numbers
      validation_data = list(xBC$test, yBC$test))

plot(history_b)

#### Evaluating the model
model_b %>% evaluate(xBC$test, yBC$test)
# Loss and accuracy for test data in the model.

##################################################### WORKSHOP CONTINUED...

# Improving the model:
# Hidden units
# More layers
# Change activation functions - allows us to apply transformation functions to the layers. SEE MATERIAL
# NB: Regression doesn't require an activation functions.
# 'relu' is the most common. If negative, set to zero, if positive, create a linear transformation. 
# Add dropout (helps prevent overfitting)
# Mostly trial and error

# tfruns package will help determine the most optimum number of layers and units, etc. 

# Dropout layer... randomly ignores some elements to remove layers which reduces overfitting.
# can have many dropout layers in different places, if need be.

library(keras)

model <- keras_model_sequential() 

model %>% layer_dense(units=10, input_shape = 4) %>% 
  layer_dropout(rate=0.3) %>% 
  layer_dense(units=3, activation = "softmax")

model %>% compile(optimizer = "rmsprop",
                  loss = "categorical_crossentropy",
                  metrics = "accuracy") 

history <- model %>% 
  fit(xIris$train,
      yIris$train,
      epochs = 100, #training numbers
      validation_data = list(xIris$test, yIris$test)) 

#### Evaluating the model
model %>% evaluate(xIris$test, yIris$test)
model %>% predict(xIris$test)
# 3 columns - probability for each row of data in each group. 
model %>% predict_classes(xIris$test) # since the classes come from Python, the classes now start from 0, not 1. 



##################################################### NETWORKS FOR SPATIAL DATA

# any 2 dimensional data is considered as 'spatial' data.

# Convolutional neural network. Adds spatial layers.
# Traditional layers do not consider relationships between layers. Spatial layers do.
# For example, in an image, neighbouring pixels will be related. In time series data, this is also true.

walking <- readRDS("/data/walking.rds")

dim(walking$x)
# 260 points in each of the 3 series, 6792 samples (people)
walking$labels
# SAMPLES SHOULD BE THE FIRST DIMENSION
# TIME/SPACE IS THE SECOND DIMENSION

walking_split=initial_split(walking, prop=0.8, strata = "labels")
# TRY AND SPLIT THESE LATER

xWalk <- readRDS("/data/xWalk.rds")
yWalk <- readRDS("/data/yWalk.rds")

# Convolution layers
# create windows called 'kernals' (size 6 in this example)
# 'strides' which says how far apart each kernal is
# 'filters' is the number of times of looking across the layers. Think of this as 'units' for dense layers.

model <- keras_model_sequential()

model %>% layer_conv_1d(filters = 40, kernel_size = 40, strides = 2,
                        activation = "relu", input_shape = c(260,3)) # don't tell how many samples, but how many values in each series, and how many series.

model

# Max pooling layer - for a particular pool size, take the maximum. Extract the strongest signal in each piece.

# max pool

model %>% layer_max_pooling_1d(pool_size=2)
model

# Flatten
model %>% layer_flatten()
model

# Finish
model %>%
  layer_dense(units = 100, activation = "sigmoid") %>%
  layer_dense(units = 15, activation = "softmax")
model


model %>% compile(loss = "categorical_crossentropy", optimizer = "adam", metrics = c("accuracy"))
history <- model %>% fit(xWalk$train, yWalk$train,
                         epochs = 15, 
                         batch_size = 128, 
                         validation_split = 0.3,
                         verbose = 1)


##################################################### NEURAL NETWORK ARCHITECTURE

# See notes

# What next?

# Pre-trained networks
# Allows us to pretrain the model to understand the general features in the data. 
# Current neural networks (?) 
# Used for forecasting.


library(tfruns)
# EXAMPLE WILL BE ADDED TO GITHUB
# Look at the tfruns website.

