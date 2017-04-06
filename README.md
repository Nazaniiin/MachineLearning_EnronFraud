# Machine Learning and Data Analysis on Enron Scandal Dataset
This project is about analysing a dataset from Enron Corporation.

In 2000, Enron was one of the largest companies in the United States. By 2002, it had collapsed into bankruptcy due to widespread corporate fraud. In the resulting Federal investigation, a significant amount of typically confidential information entered into the public record, including tens of thousands of emails and detailed financial data for top executives. 

In this project, I build a person of interest (POI) identifier based on financial and email data made public as a result of the Enron scandal. Using Python and Machine Learning techniques, I select my features of interest, train and fit them on my model and predict results on a testing dataset for predictive analysis.

This project is sort of an exploratory work, and these are the steps I take to complete it:
- Audit and clean the data from inconsistencies (i.e. removing outliers, unuseful features or observations, creating new features)
- Select the number of features I want to include in my `features_list` based on the previous step
- Create new features (if applicable)
- Try a variety of classifiers to train my data, such as Gaussian Naive Bayes, Decision Trees, and Logistic Regression
- Perform Cross Validation to calculate the percision and recall scores for my algorithms
- Select the best algorithm among the ones I work with

## Files

- `Enron_Final_Project.ipynb`
This file contains the complete walk-through of the analysis on this dataset. Reasoning is provided for each action and an in-depth description is given for the code written in `poi_id.py` file
  
- `poi_id.py`
This file contains the code for cleaning the data, adding new features, and the best classifier that best fit my dataset. I have also included, but commented out, all other classifier I used during my analysis

- `enron61702insiderpay.pdf`
Showing the enteries (people) in Enron with any financial information related to them, such as salaries, bonuses, stocks, etc.

- `final_project_dataset.pkl`
The dataset I import and work with for my analysis

