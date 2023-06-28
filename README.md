### Example unbalanced classification problem with scikit-learn

Predicting [Telco customer churn](https://www.kaggle.com/datasets/blastchar/telco-customer-churn) with logistic regression, using a combination of under- and oversampling (SMOTEEN) to address class imbalance

Results:

Without SMOTEEN:
| Class   | Precision | Recall | F1-score | Support | 
| ------- | --------- | ------ | -------- | ------- |
| 0       |   0.84    |  0.90  |  0.87    |  1549   |
| 1       |   0.66    |  0.53  |  0.59    |   561   |


After SMOTEEN:
| Class   | Precision | Recall | F1-score | Support | 
| ------- | --------- | ------ | -------- | ------- |
| 0       |   0.96    |  0.97  |  0.97    |  867    |
| 1       |   0.90    |  0.88  |  0.88    | 260     |
