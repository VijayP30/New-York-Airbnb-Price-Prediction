# CSE151A-Project

[Link to Notebook](https://colab.research.google.com/drive/1w0hO2r5xkVYwURdNq6i21P-RxChRXZqR)

## Introduction -
As one of the most populated cities in the world, New York City residents frequently struggle to find housing at affordable prices. This problem persists when short term rental units are desired for those interning in the city or searching for temporary housing whilst counting the arduous search for a permanent residence. New tenants often struggle with determining whether landlord pricing is considered fair, as the high demand for housing makes it difficult to determine if a rental unit is overpriced. To help address this issue, our project aims too develop a machine learning model trained on AirBNB rental data in order to determine if a property is priced fairly. Using features such as the neighborhood, asking price, and size of the unit, we plan to train the model to identify whether a potential tenant is getting a fair deal, represented by a property that is priced fairly given its features and location compared to other properties in the area. To accomplish this, a model framework based on a neural network will be used, with experimentation focused on number and features of different layers. A good predictive model will not only enable those searching for rental properties to determine if they are finding fair rental rates, but will also guide AirBNB hosts with regards to the pricing schema for their rental properties. 

## Methods - 
### Data Exploration - 
Our data exploration allowed us to explore multiple different correlations and points of information about our data. Our data contains about 456k observations, with 22 different columns and 20k rows. We then explored the distribution for each of the data's columns. Then we looked at the geographical distribution of our data, specifically what neighborhoods come up with most within the different NYC boroughs. Finally, created a pairplot to visually represent any underlying trends between the variables.

### Preprocessing - 
To preprocess our dataset, which contains information about NYC Airbnbs, we plan on examining the data for any missing values, duplicates, or outliers, and either scale them appropriately or discard those rows entirely (as they are incomplete). We will then encode categorical variables such as neighborhood and room type using techniques like one-hot encoding or label encoding. Next, we will standardize numerical features using techniques like min-max scaling to ensure consistency for the model. Finally, if we have any text data, such as the title of the listing or the names of the landlord, we may not use those columns as they may have no correlation with the price of the listing, or we may use text tokenization to include those features in our model. We will split the data into training and testing so it can be effectively create our model.

### Model 1 - 
Our current model is a sequential model with 6 dense layers (one of which is the input layer). Our activations are alternating tanh and relu for each of these layers. Except the last layer has a softmax activation function. We also introduced dropout to our layers to ensure even training to create a more even model. Finally we used an adam optimizer with our model.

This model has a training accuracy of 0.1974, and a testing accuracy of 0.2031. While these accuracies aren't great, we intend on creating more complex models in future iterations which will, hopefully, perform better.

Accuracy and Loss are shown below:


![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/accuracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/loss.png)

### The Fitting Chart for Model 1 - 
![fitting chart image](https://github.com/pvijay03/CSE151A-Project/blob/4b803b456519366171c458ba771a6abf689031a2/fittingchart.png)
*- Edwin Solares, CSE 151A*

To analyze where the model fits on the fitting curve, we will look at comparisons on both the y-axis, which compares model error, as well as the x-axis, which compares model complexity. As seen on the chart above, the goal is to reach a model complex enough to have good predictions, while not overfitting on the test data.

First, to understand model error, we look at the accuracy of both the training and test predictions with the model, which were 0.2031 and 0.1974 respectively. These accuracies are not great, which already means that we are on the extreme ends of the fitting curve. We can also see that the difference between our training and test accuracy is not that large, meaning that while the model fits better on the training data, it has not reached massive levels of overfitting. This points towards our model being not very complex. 

Now, to look at the x-axis, we can look into the layers of the model itself. We have many layers in the network, and specifically have six node-changing layers. Given that this is currently seen as a non-complex model, we need to either add more layers, or manipulate these layers to better fit the data we are training on. 

Overall, our current model sits on the left side of the fitting curve, and we hope to continue optimizing towards the middle of the chart.


#### Next Two Models - 
1. For the first of our two upcoming models, we will play around with the number of layers, and the different activations functions of each layer. This will give us an idea of whether the model as a whole is problematic or if certain hyperparameters just need tuning.
2. For the second of our two upcoming models, we will use our most effective iteration from the first new model and change the input value. One of our beliefs for why our current model is performing poorly is because we are giving our model too much data to sift through and we don't have a complex enough model to deal with such data. Hence, our second model will involve limiting the input data so that the model isn't overloaded.

The conclusion of our first model is that there is lots of room for improvement. Our model is likely struggling because we do not have a binary classification anymore, we are dealing with data that can be classified into 5 different groups. Hence it is very likely that our model isn't able to differenciate our input data well enough to create 5 distinct groups. In order to improve it, we can enhance the model by adding more complex arrangements between layers. We can also increase the number of layers or edit the input data so that we are classifying the listing based on fewer less-vague features.

### Model 2 - 
After reviewing the data utilized in our initial model, we identified an issue with encoding which contributed to its poor accuracy. To address this, we implemented one-hot encoding alongside bin creation, adjusting the output layer to accommodate five bins. Subsequently, employing hyperparameter tuning through RandomSearch, we optimized our model's parameters. Additionally, we experimented with varying numbers of layers to explore potential enhancements in accuracy and loss, albeit without notable success. Nevertheless, leveraging RandomSearch hyperparameter tuning significantly improved our accuracy to 0.4209 with a corresponding loss of 0.1363. While still not optimal, this marks a notable improvement from our previous results.

Accuracy and Loss for the new model are shown below:

![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2acurracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2loss.png)

#### Next Model - 
While this model was significantly more accurate than our first iteration, it still leaves a lot to be desired. For our third version we are going to implement a group of changes in an attempt to increase the accuracy of our model. First, we are going to try ensemble learnings, where we utilize the predictions of multiple models to create a system that is greater than the sum of its parts. We will create three different models, each one will be trained on a different part of the dataset, so it will be the "expert" of that part. By combining the predictions of these three models we can get a potentially more accurate prediction. We can also use a method of validation such as K-fold cross validation to utilize every datapoint for testing and training. This will allow us to maximize the amount of data our model is introduced to. Finally, as a last resort, we can use data augmentation to increase the amount of data our model is trained/tested on. We believe that by combining some or all of these methods we will be able to effectively increase our model's accuracy.

### Model 3 (Final) - 
After looking at what we did wrong for the first and second model, we made some key changes to our third model in an attempt to make it more accurate. First, we used data augmentation to give our model more training/testing data. We did this by taking our training/testing data, combining them and duplicating them, giving the model twice the amount of points to use. In the future we can use more fine-tuned versions of augmentation, where we are able to create new data instead of duplicating old data. We also created three new models, two of these were unique and were trained on all of the data, giving slightly different predictions on the pricing. The final model used the results of the other models to come up with the final prediction. We also utilized some of the parameters we found in our hyperparameter tuning from the previous model. All of this made little to no change in our accuracy. Our final model accuracy was to 0.419 with a loss of 0.158.

![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/main/model3accuracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/main/model3loss.png)

## Discussion - 
Throughout the process of developing a model to predict New York City rental rates, several key decision points were reached with regards to the best means to attempt this task. The first important stage of this process began with identifying how the predictions should be approached. Since specific price predictions would be challenging to produce given the data and needed accuracy for such a predictor, it made sense to approach the problem as a classification problem based on price tiers. Thus, based on the distribution of the data, price tiers were selected in order determine what outputs the model would provide. This concept was also believed to be best suited for the task of validating such costs, as specific prices can be hard to interpret when it comes to finding what a fair rate would be whereas a price tier categorization can be more generally applied for such verdicts.

Once the categorization criteria was set forth, the data exploration results were used to gauge the best first model to attempt. Given the pair plots of the large array of features present in the dataset, no obvious trends were seen, discouraging any efforts linked to logistic, linear, or polynomial regression models. Thus, a neural network approach was established in order to best suit the dataset. Since this was the first model, a thorough assessment and analysis of the infrastructure for the model was not completed, and model structure was chosen based on some arbitrary thoughts with regards to the data layout. The first model comprised of 6 layers including the input layer, with alternating Tanh and ReLU activation functions for all the layers except the last layer, which featured a SoftMax activation function. An Adam optimizer was also used for the model. The data was subject to one-hot encoding, polynomial feature expansion, and standardization to enhance model effectiveness. Ultimately, this model was found to have incredibly poor accuracy for both training and testing models at 0.1974 and 0.2031 respectively. After assessing the results from this model, two primary causes attributed to the incredibly low results. The first was the arbitrary nature of choosing the neural network infrastructure, as no hyperparameter tuning was performed to produce a thoughtfully develop neural network. The second cause was an issue identified with the one-hour encoding of the categories, which resulted in some price tiers consistently being overlooked by the model.  These takeaways were acknowledged and taken into consideration when developing our second model.

For our second model, the issue with the data encoding was fixed in order to better enhance model effectiveness. Furthermore, hyperparameter tuning with RandomSearch was performed in order to better curate the neural network infrastructure. The number of layers were also experimented with in order to further enhance any issues with accuracy and loss for the model. With these optimizations, the new model almost doubled in accuracy, with a new accuracy was 0.4209. Whilst this was a considerable improvement compared to the previous model, it was still far below what might be a desired accuracy should this model be used for its intended purpose. 

After the considerable efforts already performed to enhance the model, we found it difficult to consider additions that could further improve the accuracy of the model. We decided to once again revisit the data being used for model training, choosing to combine our training and testing data and duplicating them, providing the model with more data points to use. Ideally, this would have been possible through the addition of new data but the lack thereof resulted in this being the best course of action in order to enhance the data used to build the model. Additionally, we built three new models, two of which took in our new augmented data in order to gain slightly different predictions on pricing. The final model used the results of these other models in order to generate a prediction, alongside results from prior hyperparameter training. Ultimately, this did not yield any significant improvements from the previous model.

Given the limitations of the data and the several efforts to improve the model for our specific use case, it is plausible that our approach and chosen dataset are not suited for the desired target of accurately gauging the fairness of New York City rental rates. Through numerous efforts and manipulations, our model accuracy remained below 0.5, which would not be considered reliable for a prediction given a target user for this application. It is possible that different model structures beyond a neural network could be beneficial, but it is also possible that the dataset and features used are not usable for effective predictions regarding rental rate pricing. It is salient that much more effort and research would be required in order to develop a model that could be used to gauge the fairness of rental rates in New York City, as the models we developed, whilst demonstrating promise, are not directly usable for effectively performing this task.

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

