# Learning to Rank
In this project I created a Learning to Rank machine learning model. Many different libraries and algorithms were researched before creating my model including libraries such as LightGBM, XGBoost and Catboost along with many algorithms using pairwise and listwise regression. For my final model the Catboost library was selected using a ranking algorithm titled 'YetiRankPairwise'. 
## Model
My model uses the Catboost library to implement a ranker function that is able to be trained using training data, along with other helpful features for ranking problems. This ranking function takes many parameters which can alter the way the model will fit data with the three important ones being objective, iterations and learning rate. In my final model the parameters are as follows
```
{Objective = 'YetiRankPairwise', iterations = 1000, learning_rate = 0.1}
```
Every other parameter was left to default.
## Run
Before running the model, please ensure that your data files (test and training) are in the same folder as the python model file. Ensure you have also loaded in the requirements text so that you have the libraries needed to run the model.  
