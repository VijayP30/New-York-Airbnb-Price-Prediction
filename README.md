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
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/loss.png)

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

## Second Model - 
After reviewing the data utilized in our initial model, we identified an issue with encoding which contributed to its poor accuracy. To address this, we implemented one-hot encoding alongside bin creation, adjusting the output layer to accommodate five bins. Subsequently, employing hyperparameter tuning through RandomSearch, we optimized our model's parameters. Additionally, we experimented with varying numbers of layers to explore potential enhancements in accuracy and loss, albeit without notable success. Nevertheless, leveraging RandomSearch hyperparameter tuning significantly improved our accuracy to 0.4209 with a corresponding loss of 0.1363. While still not optimal, this marks a notable improvement from our previous results.

Accuracy and Loss for the new model are shown below:

![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2acurracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2loss.png)

### Next Model - 
While this model was significantly more accurate than our first iteration, it still leaves a lot to be desired. For our third version we are going to implement a group of changes in an attempt to increase the accuracy of our model. First, we are going to try ensemble learnings, where we utilize the predictions of multiple models to create a system that is greater than the sum of its parts. We will create three different models, each one will be trained on a different part of the dataset, so it will be the "expert" of that part. By combining the predictions of these three models we can get a potentially more accurate prediction. We can also use a method of validation such as K-fold cross validation to utilize every datapoint for testing and training. This will allow us to maximize the amount of data our model is introduced to. Finally, as a last resort, we can use data augmentation to increase the amount of data our model is trained/tested on. We believe that by combining some or all of these methods we will be able to effectively increase our model's accuracy.

## Third (Final) Model - 
After looking at what we did wrong for the first and second model, we made some key changes to our third model in an attempt to make it more accurate. First, we used data augmentation to give our model more training/testing data. We did this by taking our training/testing data, combining them and duplicating them, giving the model twice the amount of points to use. In the future we can use more fine-tuned versions of augmentation, where we are able to create new data instead of duplicating old data. We also created **insert number of models we created here** new models, **number that were used for general prediction** of these were unique and were trained on all of the data, giving slightly different predictions on the pricing. The final model used the results of the other models to come up with the final prediction. We also utilized some of the parameters we found in our hyperparameter tuning from the previous model. All of this **improved/worsened** our accuracy to **accuracy** with a loss of **loss**.

**Add loss and accuracy images**


## Conclusion - 

This project allowed us to understand better and play around with the techniques we were taught in class. Additionally, in trying to improve our model over the three different iterations, we were forced to look past the scope of this class to understand different ways of processing data and different types of models we can create; allowing us to peek into what the field of ML/AI holds. One main lesson we learned, if we were going to do this project again, is to find a dataset that does not have a large number of independent variables, as these variables (some of which are just different representations of other variables in our data), only served to confuse our model by adding unnecessary complexity to the features. This creates a less clear distinction between the different prediction classes, reducing our model accuracy. While this rule of thumb can be followed for any other machine learning project, the effects were especially prevalent with our dataset of choice. Another lesson we learned, which can be applied to any general machine learning project, is not to use every independent variable in our prediction right off the bat; we should first see how different subsets of our features can be used in the prediction (so we get a better understanding of how each feature independently affects our result), before combining multiple/all features into a feature map. Overall, our group believed that this project was a great learning experience and taught us many practical skills that we can employ for future ML models/projects.

## Collaboration - 
### Pranav Vijay - 
 - Helped with finding the data
 - Coded the preprocessing and first two models
 - Did the final write up
### Aryan Desai - 
 - Helped with finding the data
 - Coded the preprocessing and first two models
 - Did the final write up
### Nishanth Arumugam - 
 - Helped with finding the data
 - Answered all the write-up questions about milestones 1,2, and 3
 - Created the final models (iteration 3)
### Siddhant Kumar - 
 - Helped with finding the data
 - Answered the write-up questions about milestones 1,2, and 3
 - Created the final models (iteration 3)

