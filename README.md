# Titanic: Machine Learning from Disaster

This is my submission for the Kaggle Titanic competition. My job was to predict if a passenger survived the sinking of the Titanic or not. The score of the competition is the percentage of passengers correctly predicted. 

A score of 0.80383 was achieved using Support Vector Machine Algorithm and similarly for KNN Aglorithm, while a score of 0.78468 was achieved using Random Forest Algorithm.

One of the reasons why Random Forest achieved a lower score is that it is intrinsically suited for multiclass problems, KNN can work for any given number of classes, while SVM is intrinsically two-class. 

One of the advantages of KNN over SVM is that it directly classifies points thanks to a given distance metric, whereas SVMs needs a proper training phase.

Since our training data is small and for the purpose of this competition, the differences in using either KNN or SVM are minor and would not impact the results significantly.
