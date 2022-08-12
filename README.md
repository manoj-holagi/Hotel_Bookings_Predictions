# NextGrowth Labs

 
### Problem Statement : Given the historical data of customer booking in a hotel, train a machine learning model that predicts the customer who is going to be checked in.




 ### 1. Write about any difficult problem that you solved.(According to us difficult - is something which 90% of people would have only 10% probability in getting a similarly good solution). 
 One of the difficult problem I solved was Coustomer Conversion prediction that was given to me in a Datathon by Guvi Geek. I this proble we had to check which customers are most likely to be converted to purchasers of the insurance.
there were about 11 features and 45000 records. The Target feature was categorical Yes/No where it implied weather the customer buys the insurance or not. But the problem arose whne the data was imbalanced where it contained more than 70% was No. Hence I had to use the various over and under sampling techniques. Finally I could balance the data with SMOTEEN methos of over sampling. The imbalaced data skews the output to bias and also tends to give incorrect result even when the answers are obvious.



### 2. Explain backpropagation and tell us how you handle a dataset if 4 out of 30 parameters have null values more than 40 percentage
The usual method of dealing with the null or missing values is if it's less than 25% of the total then Central measures are considered. If it exceeds the minimum percentage of the data that's missing or null then we should check the importance of the feature on the target and if is it necessary to impute if it plays a role in affecting the target feature. The imputation techniques are filling the null values with mean if the outliers are less and if the outliers are greater then we go with median to fill the null values. If the feature turns out to not have much impact on the target feature then we consider deleting the entire column containing more than 40% of the values with null as it might reduce the accuracy of the model and also might cause overfitting or underfitting.
