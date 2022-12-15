# Optimizing an ML Pipeline in Azure

## Overview
This project is part of the Udacity Azure ML Nanodegree.
In this project, we build and optimize an Azure ML pipeline using the Python SDK and a provided Scikit-learn model.
This model is then compared to an Azure AutoML run.

## Useful Resources
- [ScriptRunConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-core/azureml.core.scriptrunconfig?view=azure-ml-py)
- [Configure and submit training runs](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-set-up-training-targets)
- [HyperDriveConfig Class](https://docs.microsoft.com/en-us/python/api/azureml-train-core/azureml.train.hyperdrive.hyperdriveconfig?view=azure-ml-py)
- [How to tune hyperparamters](https://docs.microsoft.com/en-us/azure/machine-learning/how-to-tune-hyperparameters)


## Summary
This dataset contains data from the UCI repository ["Bank Marketing Data Set"](https://archive.ics.uci.edu/ml/datasets/bank+marketing) which contains data from a Portugeuse banking institution and predicts on whether a client will subscribe a term deposit to their bank using 20 attributes such as age, employment variation rate, and days of the week the client was contacted. 

Two methods of model training were investigated: the first was a Scikit-learn pipeline that ingested the dataset and trained a logistic regression model with its parameters tuned via Azure Hyperdrive. The second was to simply feed the dataset into Azure AutoML and identify the best model created after 30 minutes of model iterations.

The models were only evaluated in terms of the model accuracy, to keep comparisons between methods simple. 

The best performing model was found to be a XGBoost classifier with a 91.8% successful prediction rate, identified via Azure's AutoML SDK implementation. 

## Scikit-learn Pipeline

Once data is imported, it goes through a cleaning function which essentially encodes all strings in the dataset to improve on performance. Data is split 70/30 for training and testing, which is standard for most ML processes. 

The chosen classifier was a logistic regression model, which is relatively simple model to tune with only the maximum number of iterations for convergence ('max_iter') and an inverse regularisation strength parameter ('C'), where higher values of 'C' would lead to weaker regulatization.

Hyperdrive tuning was controlled through a random sampler algorithm and a Bandit stopping policy. The random sampler ensured fast computational performance and (from experience) typically has comparable performance to more computationally hungry sampling methods like grid-search. The Bandit stopping policy checked every 3 iterations whether performance of the model had degraded by more than 10%, stopping the model if that was the case. This helps to ensure the tuning process will not carry on for an unnecessary amount of time. 

For both parameters, discrete ranges of values were chosen based on past experiences. 

## AutoML
AutoML configuration is relatively simple in comparison to the Scikit Pipeline / Hyperdrive method. Aside from the target dataset (the cleaned dataset) and the target label, the only other parameter to consider are the number of cross validations performed per AutoML iteration. This was chosen as 10 as this is the standard method of cross-validation seen in most ML research papers. 

## Pipeline comparison

The best performing model was found to be a XGBoost classifier with a 91.8% successful prediction rate, identified via Azure's AutoML SDK implementation. However, it only marginally outperformed the logistic regression classifier at 91.2% prediction rate which was simply finetuned via Hyperdrive. 

_Fig 1: Performance of Hyperdrive model measured as accuracy over model iteration. The model's highest accuracy is highlighted._ 
![Proj1_Hyperdrive_performance](https://user-images.githubusercontent.com/24628312/207839162-04f0197e-c752-4306-a5fb-1c0745301c84.jpg)

_Fig 2: Output metrics of the AutoML Model. The accuracy of the best performing model (XGBoost) is highlighted._
![Proj1_AutoML_performance](https://user-images.githubusercontent.com/24628312/207839214-c702efdd-685f-4ab0-9ae8-974a7f098920.jpg)

Logistic Regression and XGBoost are naturally very different classifiers and so their performance on a particular ML problem will be entirely context-dependent. However, XGBoost models are more architecturally complex and typically tend to perform quite well as classifiers in comparison to the more simplisitic logisitc regression model (based on past research and experience). In this particular instance, XGBoost appears to have found a good combination of weak learning configurations to give it the edge over the logistic regression model. 

## Future work
The Hyperdrive pipeline could be improved through experimenting with different types of search algorithms like stochastic and grid-search as well as widening the range of the logistic regression variables. Some consideration would need to be given on the computational cost of these additions. 

In terms of improving the AutoML configuration, limiting the types of classifiers considered for AutoML and reducing the number of cross-validation checks would allow the AutoML to cycle through more iterations within a given timeframe of 30 minutes and perhaps lead to identifying a more optimal solution. 

Another possible solution would be to combine the learning of the AutoML run (where XGBoost was found to be the best classifier) and then perform hyperdrive tuning on XGBoost models to determine whether a more optimal configuration of XGBoost exists, ignorning all other classifiers to reduce computational cost. 

## Proof of cluster clean up

![Proj1_Cluster_cleanup_proof](https://user-images.githubusercontent.com/24628312/207839253-1b8f46f0-a24b-4f4f-a7cd-7a837a7343f7.jpg)

