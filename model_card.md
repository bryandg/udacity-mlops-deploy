# Model Card
For additional information see the Model Card paper: https://arxiv.org/pdf/1810.03993.pdf

## Model Details
Bryan created this model. It is a logistic regression model using the default hyperparameters in scikit-learn 1.0.2.

## Intended Use
Predict income (>50K or <=50K) based on a variety of demographic data points.

## Training Data
data/train.csv -- w/ the proper credentials this can be pulled from s3 using DVC.

## Evaluation Data
data/test.csv -- w/ the proper credentials this can be pulled from s3 using DVC.

## Metrics
precision: 0.72, recall: 0.25, fbeta: 0.37

## Ethical Considerations
Model performance varies significantly by native country of the individual -- there is very little data for people born outside of the United States.

## Caveats and Recommendations
See data/slice_metrics.csv to better understand the limitations of the model on different populations.
