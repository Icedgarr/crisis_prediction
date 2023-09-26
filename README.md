# Combining clinical notes with structured electronic health records enhances the prediction of mental health crises


This library contains the code used to process the data, build the algorithms and generate the results for the study presented in the manuscript: link.
There are three main steps to generate the results:
- Feature creation: The classes in the `crisis_prediction/features` folder are used to preprocess the raw data and create the features.
- Model development: The classes in the `crisis_prediction/models` folder are used to tune the hyperparameters and build the models presented in the manuscript. For each of the models, some preprocessing steps are applied to the features. The preprocessors used are stored in the folder `crisis_prediction/preprocessors`.
- Model evaluation: The classes and functions in the `crisis_prediction/validators` are used to produce the metrics reported in the paper. The code to generate each of the metrics is stored in `crisis_prediction/metrics`.

Specific details to replicate the results can be found in the manuscript: DOI link.