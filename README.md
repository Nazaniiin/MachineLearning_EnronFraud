# MachineLearning
This project is about analysing a dataset from Enron Corporation.

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 

In this project, I build a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. Using Python and Machine Learning techniques, I select my features of interest, train and fit them on my model and predict results on a testing dataset for predictive analysis.

This project is sort of an exploratory work, and these are the steps I take to complete it:
- Audit and clean the data from inconsistencies (i.e. removing outliers, unuseful features or observations, creating new features)
- Select the number of features I want to include in my `features_list` based on the previous step
- Create new features (if applicable)
- Try a variety of classifiers to train my data, such as Gaussian Naive Bayes, Support Vector Machines (SVM), and Decision Trees
- Predict results for the testing set
- Perform Cross Validation to calculate the percision and recall scores for my algorithms
- Select the best algorithm among the ones I work with


