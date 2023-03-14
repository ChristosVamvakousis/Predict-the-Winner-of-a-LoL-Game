This is a dataset from 2017, I have trained 3 different models using GradientBoostingClassifier algorithm in order to predict the winner
team based on champions picked and objectives. 

First model is before the game starts, its when all players have locked their champion and make the first prediction with accuracy of
~52%. If the dataset included players rank and skills on the champions, the accuracy could be higher.
Early Game model is the second one which includes players champions and which team took first blood and tower with an accuracy of 70%.
Late Game model is the last one which includes all the above and which team took first dragon, baron and inhibitor. The accuracy is 89%.

The dataset is using a JSON file which includes the champions ID.

First, the program trains the models and then collects the data we are about to predict. Later on makes the first predict and requires more info
for the early game model and so on for the late game model.

Note that the code is not clean/good written, it may be confusing.