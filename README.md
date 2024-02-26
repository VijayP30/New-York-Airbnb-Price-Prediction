# CSE151A-Project

[Link to Notebook](https://colab.research.google.com/drive/1w0hO2r5xkVYwURdNq6i21P-RxChRXZqR)

## Data Exploration - 
Our data exploration allowed us to explore multiple different correlations and points of information about our data. Our data contains about 456k observations, with 22 different columns and 20k rows. We then explored the distribution for each of the data's columns. Then we looked at the geographical distribution of our data, specifically what neighborhoods come up with most within the different NYC boroughs. Finally, created a pairplot to visually represent any underlying trends between the variables.

## Preprocessing - 
To preprocess our dataset, which contains information about NYC Airbnbs, we plan on examining the data for any missing values, duplicates, or outliers, and either scale them appropriately or discard those rows entirely (as they are incomplete). We will then encode categorical variables such as neighborhood and room type using techniques like one-hot encoding or label encoding. Next, we will standardize numerical features using techniques like min-max scaling to ensure consistency for the model. Finally, if we have any text data, such as the title of the listing or the names of the landlord, we may not use those columns as they may have no correlation with the price of the listing, or we may use text tokenization to include those features in our model. We will split the data into training and testing so it can be effectively create our model.

## First Model - 
Our current model is a sequential model with 6 dense layers (one of which is the input layer). Our activations are alternating tanh and relu for each of these layers. Except the last layer has a softmax activation function. We also introduced dropout to our layers to ensure even training to create a more even model. Finally we used an adam optimizer with our model.

This model has a training accuracy of 0.1974, and a testing accuracy of 0.2031. While these accuracies aren't great, we intend on creating more complex models in future iterations which will, hopefully, perform better.

Accuracy and Loss are shown below:


![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/accuracy.png)
![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/loss.png)


## The Fitting Chart - 
![fitting chart image](https://github.com/pvijay03/CSE151A-Project/blob/4b803b456519366171c458ba771a6abf689031a2/fittingchart.png)
*- Edwin Solares, CSE 151A*

To analyze where the model fits on the fitting curve, we will look at comparisons on both the y-axis, which compares model error, as well as the x-axis, which compares model complexity. As seen on the chart above, the goal is to reach a model complex enough to have good predictions, while not overfitting on the test data.

First, to understand model error, we look at the accuracy of both the training and test predictions with the model, which were 0.2031 and 0.1974 respectively. These accuracies are not great, which already means that we are on the extreme ends of the fitting curve. We can also see that the difference between our training and test accuracy is not that large, meaning that while the model fits better on the training data, it has not reached massive levels of overfitting. This points towards our model being not very complex. 

Now, to look at the x-axis, we can look into the layers of the model itself. We have many layers in the network, and specifically have six node-changing layers. Given that this is currently seen as a non-complex model, we need to either add more layers, or manipulate these layers to better fit the data we are training on. 

Overall, our current model sits on the left side of the fitting curve, and we hope to continue optimizing towards the middle of the chart.


### Next Two Models - 
1. For the first of our two upcoming models, we will play around with the number of layers, and the different activations functions of each layer. This will give us an idea of whether the model as a whole is problematic or if certain hyperparameters just need tuning.
2. For the second of our two upcoming models, we will use our most effective iteration from the first new model and change the input value. One of our beliefs for why our current model is performing poorly is because we are giving our model too much data to sift through and we don't have a complex enough model to deal with such data. Hence, our second model will involve limiting the input data so that the model isn't overloaded.

The conclusion of our first model is that there is lots of room for improvement. Our model is likely struggling because we do not have a binary classification anymore, we are dealing with data that can be classified into 5 different groups. Hence it is very likely that our model isn't able to differenciate our input data well enough to create 5 distinct groups. In order to improve it, we can enhance the model by adding more complex arrangements between layers. We can also increase the number of layers or edit the input data so that we are classifying the listing based on fewer less-vague features.

