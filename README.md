# Credit Card Fraud Detecton.

**Using Tensorflow Neural Network and implementing Decision Tree , KNN and SVM to find accuracy and F1 score.**




**About the Dataset.**


The dataset contains transactions made by credit cards in September 2013 by European cardholders. This dataset presents transactions that occurred in two days, where we have 492 frauds out of 284,807 transactions. The dataset is highly unbalanced, the positive class (frauds) account for 0.172% of all transactions.

It contains only numerical input variables which are the result of a PCA transformation. Unfortunately, due to confidentiality issues, we cannot provide the original features and more background information about the data. Features V1, V2, … V28 are the principal components obtained with PCA, the only features which have not been transformed with PCA are 'Time' and 'Amount'. Feature 'Time' contains the seconds elapsed between each transaction and the first transaction in the dataset. The feature 'Amount' is the transaction Amount, this feature can be used for example-dependant cost-sensitive learning. Feature 'Class' is the response variable and it takes value 1 in case of fraud and 0 otherwise.


After importing all necessary modules and packages the data is read using pandas and is shown as df. There are 28 features in it shown as v1,v2,v3...,hence we are reducing the features to 10 and renaming the columns to feature 1, feature 2,....feature10. Thus cleaning and organizing the data.


Next we go for transaction distribution :

        Total number of transactions: 284807
        Number of normal transactions: 284315
        Number of fraudulent transactions: 492
        Percentage of fraudulent transactions: 0.17
        
The above result shows the number of transactions that took place , the number of transactions that were found to be normal and the number of fraudulent transactions and its percentage resulting to 0.17% . Here we see that we have a highly imbalanced dataset with only 0.17% fraudulent transactions, which is expected since we don't expect our credit card to be involved in a scam/fraud every single day (otherwise many would worry about the financial security provided by financial institutions!)


The next thing is we check for NUllVALUES.


We display a plot which represents at what time transactions took place which includes Normal and Fraudulent transactions.

<img width="448" alt="Screenshot 2022-09-19 at 7 18 25 PM" src="https://user-images.githubusercontent.com/83020129/191032675-0656a409-e735-4f88-a4ad-44d421853808.png">



We discard the duplicates and diplay the distribution in histograms of each single features which compares Normal and Fraudulent transactions.
The below image shows the histographical representation of feature 1,feature 2,feature 3, feature 4.

<img width="788" alt="Screenshot 2022-09-19 at 7 19 41 PM" src="https://user-images.githubusercontent.com/83020129/191033179-69469728-1279-4f14-a735-d3691fa6693c.png">

<img width="788" alt="Screenshot 2022-09-19 at 7 19 54 PM" src="https://user-images.githubusercontent.com/83020129/191033231-10c1e82a-ef94-444d-942a-6ae03c96447e.png">

We clean,reshape and organize the data to move on to the training of the dataset.we apply the trained model to determine how well our model fare in terms of successfully predicting fraudulent transactions.The dataset is to split up the train-test set into 80-20 percent. There is no hard rule in splitting the dataset and one is free to slightly alter the fractions.


Tensotflow Neural Network is used using neural network with stacks of logistic regressions, and each layer consists of linear function followed by RELU activation function (except the last layer, where we will use sigmoid activation function). We will choose to use Adam Optimizers for regularization.

Set number of hidden nodes in each layer with a constant ratio. 
We will use 4 layers with 
**LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> RELU -> LINEAR -> SIGMOID.**

Next we build the weights and biases for each layer, where we write operations between tensors and initialize variables.
The weights and the bias are declared the stddev is set to 0.25.

After implementing all layers we get W1, b1, W2, b2, W3, b3, W4, b4. And they are defined under forward propagation to get y4.
The hyperparameters such as number of epochs, learning rate and batch size are set where learning_rate = 0.0005,batch_size = 2048 and epoch = 10.

The next thing is we compute cost function with the formula **cost = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=labels))**

We implement backpropagation and get the results.

Now we use few algorithms to find accuracy and f1 score of our model

The first one is **Decision Tree**

which gives the result as 


<img width="332" alt="Screenshot 2022-09-19 at 7 37 15 PM" src="https://user-images.githubusercontent.com/83020129/191036793-89928f76-4fcb-4767-8166-9e7faabc79a2.png">


The next is K-Nearest Neighbor Algorithm

Which gives the result as

<img width="325" alt="Screenshot 2022-09-19 at 7 39 13 PM" src="https://user-images.githubusercontent.com/83020129/191037321-e61db4bf-c622-4db2-9dc0-e7753aa5ddf6.png">

The last is the Support Vector Machine
Which gives the result as 

<img width="321" alt="Screenshot 2022-09-19 at 7 40 36 PM" src="https://user-images.githubusercontent.com/83020129/191037623-470ad4ad-d953-4cea-ace9-9f3c4009f81e.png">


This documentation is a personal learning documentation created by referring various sources and self knowlegde.
