# CSE151A-Project

[Link to Notebook](https://colab.research.google.com/drive/1w0hO2r5xkVYwURdNq6i21P-RxChRXZqR)

## Introduction -
As one of the most populated cities in the world, New York City residents frequently struggle to find housing at affordable prices. This problem persists when short term rental units are desired for those interning in the city or searching for temporary housing whilst counting the arduous search for a permanent residence. New tenants often struggle with determining whether landlord pricing is considered fair, as the high demand for housing makes it difficult to determine if a rental unit is overpriced. To help address this issue, our project aims too develop a machine learning model trained on AirBNB rental data in order to determine if a property is priced fairly. Using features such as the neighborhood, asking price, and size of the unit, we plan to train the model to identify whether a potential tenant is getting a fair deal, represented by a property that is priced fairly given its features and location compared to other properties in the area. To accomplish this, a model framework based on a neural network will be used, with experimentation focused on number and features of different layers. A good predictive model will not only enable those searching for rental properties to determine if they are finding fair rental rates, but will also guide AirBNB hosts with regards to the pricing schema for their rental properties. 

## Methods - 
### Data Exploration - 
Our data exploration allowed us to explore multiple different correlations and points of information about our data. Our data contains about 456k observations, with 22 different columns and 20k rows. We then explored the distribution for each of the data's columns. Then we looked at the geographical distribution of our data, specifically what neighborhoods come up with most within the different NYC boroughs. Finally, created a pairplot to visually represent any underlying trends between the variables.

Data Distribution:
```
print("Summary Statistics for Numerical Features in Dataset:")
print(rawDataFrame.describe())
```

Neighborhood Exploration:
```
unique_neighborhood_groups = rawDataFrame['neighbourhood_group'].unique()

for neighborhood_group in unique_neighborhood_groups:
    plt.figure(figsize=(12, 6))
    neighborhood_data = rawDataFrame[rawDataFrame['neighbourhood_group'] == neighborhood_group]
    unique_neighborhoods = neighborhood_data['neighbourhood'].unique()
    x_ticks = range(len(unique_neighborhoods))
    plt.bar(x_ticks, neighborhood_data['neighbourhood'].value_counts(), width=0.8)
    plt.title(f'Neighborhoods in {neighborhood_group}')
    plt.xlabel('Neighborhood')
    plt.ylabel('Frequency')
    plt.xticks(x_ticks, unique_neighborhoods, rotation=90, ha='right')
    plt.tight_layout()
    plt.show()
```

Pairplot:
```
sns.pairplot(rawDataFrame, hue='neighbourhood_group')
plt.show()
```

### Preprocessing - 
To preprocess our dataset, which contains information about NYC Airbnbs, we plan on examining the data for any missing values, duplicates, or outliers, and either scale them appropriately or discard those rows entirely (as they are incomplete). We will then encode categorical variables such as neighborhood and room type using techniques like one-hot encoding or label encoding. Next, we will standardize numerical features using techniques like min-max scaling to ensure consistency for the model. Finally, if we have any text data, such as the title of the listing or the names of the landlord, we may not use those columns as they may have no correlation with the price of the listing, or we may use text tokenization to include those features in our model. We will split the data into training and testing so it can be effectively create our model.

Cleaning Data:
```
leanedDataFrame = rawDataFrame[rawDataFrame['rating'] != 'No rating']
cleanedDataFrame = cleanedDataFrame[cleanedDataFrame['rating'] != 'New ']

# Replace Studio in bedrooms column with 1
cleanedDataFrame['bedrooms'] = cleanedDataFrame['bedrooms'].replace('Studio', 1)

# Replace Not specified in baths column with 0
cleanedDataFrame['baths'] = cleanedDataFrame['baths'].replace('Not specified', 0)

# Separate target variable and features
X = cleanedDataFrame.drop(columns=['price'])
y = cleanedDataFrame['price']


# Set numerical and categorical features
numerical_cols = ['rating', 'bedrooms', 'beds', 'baths']
categorical_cols = ['neighbourhood_group', 'room_type']
```

Scaling, Encoding, and Imputing (Transformations):
```
# Define the transformation pipelines

# Simple Imputer, Polynomial Feature Expansion, and Standard Scaler
numerical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='mean')),
    ('poly', PolynomialFeatures(degree=2, include_bias=False)),
    ('scaler', StandardScaler())
])

# Simple Imputer and One-Hot Encoding
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder())
])

# Apply the pipelines to the preprocessor and transform the data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])
```


### Example Fitting Chart for Models - 
![fitting chart image](https://github.com/pvijay03/CSE151A-Project/blob/4b803b456519366171c458ba771a6abf689031a2/fittingchart.png)
*- Edwin Solares, CSE 151A*

### Model 1 - 
Our current model is a sequential model with 6 dense layers (one of which is the input layer). Our activations are alternating tanh and relu for each of these layers. Except the last layer has a softmax activation function. We also introduced dropout to our layers to ensure even training to create a more even model. Finally we used an adam optimizer with our model.
```
model = Sequential()
model.add(Dense(1024, activation='tanh', kernel_regularizer=regularizers.l2(0.01), input_shape=(X_train.shape[1],)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(512, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(256, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))
model.add(Dense(64, activation='tanh', kernel_regularizer=regularizers.l2(0.01)))
model.add(BatchNormalization())
model.add(Dropout(0.5))

model.add(Dense(1, activation='softmax'))

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
```

#### Next Two Models - 
1. For the first of our two upcoming models, we will play around with the number of layers, and the different activations functions of each layer. This will give us an idea of whether the model as a whole is problematic or if certain hyperparameters just need tuning.
2. For the second of our two upcoming models, we will use our most effective iteration from the first new model and change the input value. One of our beliefs for why our current model is performing poorly is because we are giving our model too much data to sift through and we don't have a complex enough model to deal with such data. Hence, our second model will involve limiting the input data so that the model isn't overloaded.

The conclusion of our first model is that there is lots of room for improvement. Our model is likely struggling because we do not have a binary classification anymore, we are dealing with data that can be classified into 5 different groups. Hence it is very likely that our model isn't able to differenciate our input data well enough to create 5 distinct groups. In order to improve it, we can enhance the model by adding more complex arrangements between layers. We can also increase the number of layers or edit the input data so that we are classifying the listing based on fewer less-vague features.

### Model 2 - 
After reviewing the data utilized in our initial model, we identified an issue with encoding which contributed to its poor accuracy. To address this, we implemented one-hot encoding alongside bin creation, adjusting the output layer to accommodate five bins. Subsequently, employing hyperparameter tuning through RandomSearch, we optimized our model's parameters. Additionally, we experimented with varying numbers of layers to explore potential enhancements in accuracy and loss, albeit without notable success.

New One-Hot Encoding:
```
one_hot_encoded_train = pd.get_dummies(y_train)
one_hot_encoded_test = pd.get_dummies(y_test)
```
Hyperparameter Tuning:
```
def build_model(hp):
  model = Sequential([
      Dense(units=hp.Int('units_1', min_value=0, max_value=1024, step=1), activation=hp.Choice('activation_1', values=['relu', 'tanh', 'sigmoid']), input_shape=(X_train.shape[1],)),
      BatchNormalization(),
      Dropout(rate=hp.Float('dropout_1', min_value=0.2, max_value=0.8, step=0.1)),
      Dense(units=hp.Int('units_2', min_value=0, max_value=512, step=1), activation=hp.Choice('activation_2', values=['relu', 'tanh', 'sigmoid'])),
      BatchNormalization(),
      Dropout(rate=hp.Float('dropout_2', min_value=0.2, max_value=0.8, step=0.1)),
      Dense(units=hp.Int('units_3', min_value=0, max_value=256, step=1), activation=hp.Choice('activation_3', values=['relu', 'tanh', 'sigmoid'])),
      BatchNormalization(),
      Dropout(rate=hp.Float('dropout_3', min_value=0.2, max_value=0.8, step=0.1)),
      Dense(units=hp.Int('units_4', min_value=0, max_value=64, step=1), activation=hp.Choice('activation_4', values=['relu', 'tanh', 'sigmoid'])),
      BatchNormalization(),
      Dropout(rate=hp.Float('dropout_4', min_value=0.2, max_value=0.8, step=0.1)),
      Dense(units=one_hot_encoded_train.shape[1], activation=hp.Choice('activation_5', values=['softmax', 'relu', 'tanh', 'sigmoid']))
  ])
  # Choice of loss function and learning rate
  loss_function = hp.Choice('loss', values=['mean_squared_error', 'binary_crossentropy', 'categorical_crossentropy'])
  learning_rate = hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")

  # Compile the model
  model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
                loss=loss_function,
                metrics=['accuracy'])
  return model

tuner = keras_tuner.RandomSearch(
    hypermodel=build_model,
    objective="val_accuracy",
    max_trials=3,
    executions_per_trial=2,
    overwrite=True,
    directory="hw2",
    project_name="hyperparamter_tuning_results"
)
```

#### Next Model - 
While this model was significantly more accurate than our first iteration, it still leaves a lot to be desired. For our third version we are going to implement a group of changes in an attempt to increase the accuracy of our model. First, we are going to try ensemble learnings, where we utilize the predictions of multiple models to create a system that is greater than the sum of its parts. We will create three different models, each one will be trained on a different part of the dataset, so it will be the "expert" of that part. By combining the predictions of these three models we can get a potentially more accurate prediction. We can also use a method of validation such as K-fold cross validation to utilize every datapoint for testing and training. This will allow us to maximize the amount of data our model is introduced to. Finally, as a last resort, we can use data augmentation to increase the amount of data our model is trained/tested on. We believe that by combining some or all of these methods we will be able to effectively increase our model's accuracy.

### Model 3 (Final) - 
After looking at what we did wrong for the first and second model, we made some key changes to our third model in an attempt to make it more accurate. First, we used data augmentation to give our model more training/testing data. We did this by taking our training/testing data, combining them and duplicating them, giving the model twice the amount of points to use. In the future we can use more fine-tuned versions of augmentation, where we are able to create new data instead of duplicating old data. We also created three new models, two of these were unique and were trained on all of the data, giving slightly different predictions on the pricing.

Setup for Cross Validation:
```
X = np.concatenate((X_train, X_test), axis=0)
Y_one_hot = np.concatenate((one_hot_encoded_train, one_hot_encoded_test), axis=0)

X_augmented = np.repeat(X, 2, axis=0)
Y_one_hot_augmented = np.repeat(Y_one_hot, 2, axis=0)

split_index = len(X_train)*2

x_train_augmented = X_augmented[:split_index]
x_test_augmented = X_augmented[split_index:]
y_train_one_hot_augmented = Y_one_hot_augmented[:split_index]
y_test_one_hot_augmented = Y_one_hot_augmented[split_index:]
```

First Model:
```
first_model = Sequential([
      Dense(units=best_hyperparameters.get('units_1'), activation=best_hyperparameters.get('activation_1'), input_shape=(X_train.shape[1],)),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_1')),
      Dense(units=best_hyperparameters.get('units_2'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_3')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_3')),
      Dense(units=best_hyperparameters.get('units_4'), activation=best_hyperparameters.get('activation_4')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_4')),
      Dense(units=one_hot_encoded_train.shape[1], activation=best_hyperparameters.get('activation_5'))
    ])
```

Second Model:
```
second_model = Sequential([
      Dense(units=best_hyperparameters.get('units_1'), activation=best_hyperparameters.get('activation_1'), input_shape=(X_train.shape[1],)),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_1')),
      Dense(units=best_hyperparameters.get('units_2'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=best_hyperparameters.get('units_3'), activation=best_hyperparameters.get('activation_3')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_4')),
      Dense(units=one_hot_encoded_train.shape[1], activation=best_hyperparameters.get('activation_5'))
    ])
```

Third Model:
```
final_model = Sequential([
      Dense(units=best_hyperparameters.get('units_1'), activation=best_hyperparameters.get('activation_1'), input_shape=(concatenated_x_train.shape[1],)),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_1')),
      Dense(units=best_hyperparameters.get('units_2'), activation=best_hyperparameters.get('activation_2')),
      BatchNormalization(),
      Dropout(best_hyperparameters.get('dropout_2')),
      Dense(units=one_hot_encoded_train.shape[1], activation=best_hyperparameters.get('activation_5'))
    ])
```

## Results - 
### Data Exploration - 
Data Distribution:
```
Frequency Counts for Unique Categorical Feature Values:
name:
Rental unit in New York · 1 bedroom · 1 bed · 1 bath                    409
Rental unit in New York · ★5.0 · 1 bedroom · 1 bed · 1 bath             235
Rental unit in Brooklyn · 1 bedroom · 1 bed · 1 bath                    201
Rental unit in Brooklyn · ★5.0 · 1 bedroom · 1 bed · 1 bath             173
Rental unit in Brooklyn · 1 bedroom · 1 bed · 1 shared bath             158
                                                                       ... 
Rental unit in Brooklyn · ★4.52 · 1 bedroom · 3 beds · 1 bath             1
Rental unit in Brooklyn · ★4.83 · 2 bedrooms · 1 bed · 1 shared bath      1
Townhouse in Queens · ★4.33 · 2 bedrooms · 3 beds · 1 bath                1
Rental unit in New York · ★4.50 · 1 bedroom · 1 bed · 1 private bath      1
Rental unit in New York · ★4.46 · 1 bedroom · 2 beds · 1 bath             1
Name: name, Length: 9836, dtype: int64

host_name:
Jeniffer           184
Urban Furnished    176
Michael            148
David              144
Stay With Vibe     113
                  ... 
Maud                 1
Raych                1
Alessio              1
Adrien               1
Glenroy              1
Name: host_name, Length: 5815, dtype: int64

neighbourhood_group:
Manhattan        8038
Brooklyn         7719
Queens           3761
Bronx             949
Staten Island     291
Name: neighbourhood_group, dtype: int64

neighbourhood:
Bedford-Stuyvesant            1586
Harlem                        1063
Williamsburg                   969
Midtown                        942
Hell's Kitchen                 867
                              ... 
Lighthouse Hill                  1
Bay Terrace, Staten Island       1
Navy Yard                        1
Neponsit                         1
Chelsea, Staten Island           1
Name: neighbourhood, Length: 221, dtype: int64

room_type:
Entire home/apt    11549
Private room        8804
Shared room          293
Hotel room           112
Name: room_type, dtype: int64

last_review:
2023-09-04    326
2023-12-03    255
2023-12-17    244
2023-09-05    223
2023-11-30    212
             ... 
2021-11-03      1
2016-07-31      1
2021-05-16      1
2021-02-03      1
2019-07-11      1
Name: last_review, Length: 1878, dtype: int64

license:
No License            17569
Exempt                 2135
OSE-STRREG-0000068       14
OSE-STRREG-0008664        6
OSE-STRREG-0000437        6
                      ...  
OSE-STRREG-0000131        1
OSE-STRREG-0000288        1
OSE-STRREG-0000215        1
OSE-STRREG-0000985        1
OSE-STRREG-0000513        1
Name: license, Length: 879, dtype: int64

rating:
No rating    3593
5.0          2882
4.67          517
4.75          469
4.80          439
             ... 
4.02            1
3.42            1
3.73            1
3.26            1
2.75            1
Name: rating, Length: 162, dtype: int64

bedrooms:
1         13439
2          3602
Studio     1815
3          1394
4           345
5           112
6            29
7             9
9             6
8             5
15            1
14            1
Name: bedrooms, dtype: int64

baths:
1                17015
2                 2082
1.5               1001
2.5                240
3                  171
0                   82
4                   69
3.5                 50
4.5                 18
Not specified       13
5                    6
6                    4
5.5                  2
6.5                  2
11.5                 1
7                    1
15.5                 1
Name: baths, dtype: int64
```

[Neighborhood Exploration Example](https://github.com/pvijay03/CSE151A-Project/blob/main/neighborhooddist.png)

[Pairplot](https://github.com/pvijay03/CSE151A-Project/blob/main/pairplot.png)

### Preprocessing -
Cleaning Data: We were able to successfully remove all of the invalid data from our dataframe
```
Column 'id': 0 rows with 'Not specified' rawDataFrame
Column 'name': 0 rows with 'Not specified' rawDataFrame
Column 'host_id': 0 rows with 'Not specified' rawDataFrame
Column 'host_name': 0 rows with 'Not specified' rawDataFrame
Column 'neighbourhood_group': 0 rows with 'Not specified' rawDataFrame
Column 'neighbourhood': 0 rows with 'Not specified' rawDataFrame
Column 'latitude': 0 rows with 'Not specified' rawDataFrame
Column 'longitude': 0 rows with 'Not specified' rawDataFrame
Column 'room_type': 0 rows with 'Not specified' rawDataFrame
Column 'price': 0 rows with 'Not specified' rawDataFrame
Column 'minimum_nights': 0 rows with 'Not specified' rawDataFrame
Column 'number_of_reviews': 0 rows with 'Not specified' rawDataFrame
Column 'last_review': 0 rows with 'Not specified' rawDataFrame
Column 'reviews_per_month': 0 rows with 'Not specified' rawDataFrame
Column 'calculated_host_listings_count': 0 rows with 'Not specified' rawDataFrame
Column 'availability_365': 0 rows with 'Not specified' rawDataFrame
Column 'number_of_reviews_ltm': 0 rows with 'Not specified' rawDataFrame
Column 'license': 0 rows with 'Not specified' rawDataFrame
Column 'rating': 0 rows with 'Not specified' rawDataFrame
Column 'bedrooms': 0 rows with 'Not specified' rawDataFrame
Column 'beds': 0 rows with 'Not specified' rawDataFrame
Column 'baths': 13 rows with 'Not specified' rawDataFrame
Column 'log_price': 0 rows with 'Not specified' rawDataFrame
```

Scaling, Encoding, and Imputing (Transformations):

[Example of New Scaled/Imputed Data](https://github.com/pvijay03/CSE151A-Project/blob/main/dataScaled.png)

One-Hot Encoding:
```
ONE HOT ENCODED TEST
       1  3  5  7  9
7417   1  0  0  0  0
4253   0  0  0  1  0
12741  0  0  0  1  0
14213  0  0  0  1  0
6554   1  0  0  0  0
...   .. .. .. .. ..
3471   0  0  0  1  0
15923  1  0  0  0  0
6795   0  0  0  1  0
4393   0  0  0  1  0
8163   1  0  0  0  0

[3402 rows x 5 columns]
```

### Model 1 - 
This model has a training accuracy of 0.1974, and a testing accuracy of 0.2031. While these accuracies aren't great, we intend on creating more complex models in future iterations which will, hopefully, perform better.

Accuracy and Loss are shown below:


![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/accuracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/af6042741cdc1a28df0813f154a91280ea3d7fcd/loss.png)

To analyze where the model fits on the fitting curve, we will look at comparisons on both the y-axis, which compares model error, as well as the x-axis, which compares model complexity. As seen on the chart above, the goal is to reach a model complex enough to have good predictions, while not overfitting on the test data.

First, to understand model error, we look at the accuracy of both the training and test predictions with the model, which were 0.2031 and 0.1974 respectively. These accuracies are not great, which already means that we are on the extreme ends of the fitting curve. We can also see that the difference between our training and test accuracy is not that large, meaning that while the model fits better on the training data, it has not reached massive levels of overfitting. This points towards our model being not very complex. 

Now, to look at the x-axis, we can look into the layers of the model itself. We have many layers in the network, and specifically have six node-changing layers. Given that this is currently seen as a non-complex model, we need to either add more layers, or manipulate these layers to better fit the data we are training on. 

Overall, our current model sits on the left side of the fitting curve, and we hope to continue optimizing towards the middle of the chart.

### Model 2 -
Leveraging RandomSearch hyperparameter tuning significantly improved our accuracy to 0.4209 with a corresponding loss of 0.1363. While still not optimal, this marks a notable improvement from our previous results.

Accuracy and Loss for the new model are shown below:

![accuracy image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2acurracy.png)
![loss image](https://github.com/pvijay03/CSE151A-Project/blob/main/model2loss.png)

### Model 3 (Final) -
The final model used the results of the other models to come up with the final prediction. We also utilized some of the parameters we found in our hyperparameter tuning from the previous model. All of this made little to no change in our accuracy. Our final model accuracy was to 0.419 with a loss of 0.158.

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

