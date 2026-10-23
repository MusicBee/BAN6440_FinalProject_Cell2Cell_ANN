BAN6440 Final Project: Cell2Cell Customer Churn Prediction
Project Overview
This project investigates customer churn prediction using the Cell2Cell telecommunications customer dataset. It compares a baseline Artificial Neural Network (ANN) with a Logistic Regression benchmark and examines how selecting a classification threshold using validation data affects churn detection.
The business objective is to help a telecommunications provider identify customers who may be at risk of leaving so retention teams can better prioritise interventions. Model outputs should support—not replace—business judgment and campaign testing.
Dataset
Dataset file expected: `cell2celltrain.csv`
Rows: 51,047
Original columns: 58
Target: `Churn` (Yes/No)
Target distribution in the analysed source file: 36,336 No (71.18%) and 14,711 Yes (28.82%)
Place the dataset in the project directory or pass its path to the scripts. Confirm the dataset's original source, licence, feature definitions, and citation before final submission. The dataset itself is not included in this README.
Workflow
Inspect the dataset, missing values, duplicates, and target distribution.
Exclude `CustomerID` from model predictors because it is an identifier.
Exclude `RetentionCalls`, `RetentionOffersAccepted`, and `MadeCallToRetentionTeam` from the primary feature set to reduce potential target leakage.
Split the data using stratified training, validation, and test partitions (70% / 15% / 15%; random seed 42).
Fit preprocessing on the training data only:
Numerical features: median imputation and standardisation.
Categorical features: most-frequent imputation and one-hot encoding.
Train and evaluate the baseline ANN.
Train and evaluate Logistic Regression as a benchmark.
Select a classification threshold on validation data by maximising F1-score, then evaluate that selected threshold on the held-out test set.
Compare model performance and document limitations and potential business implications.
Baseline ANN
The baseline ANN architecture used in the experiment:
Input layer based on the transformed feature matrix
Dense layer: 32 neurons, ReLU activation
Dense layer: 16 neurons, ReLU activation
Output layer: 1 neuron, sigmoid activation
Optimiser: Adam, learning rate 0.001
Loss: binary cross-entropy
Batch size: 32
Maximum epochs: 100
Early stopping: validation-loss patience of 10, restoring best weights
The reported run trained for 12 epochs and restored the best weights from epoch 2.
Confirmed Results
Baseline ANN at threshold 0.50
Metric	Test result
Accuracy	0.7068
Precision	0.4296
Recall	0.0526
F1-score	0.0937
ROC-AUC	0.6314
Average precision	0.3872
Test-set size	7,658
Confusion matrix (rows = actual class; columns = predicted class):
```text
[[5297, 154],
 [2091, 116]]
```
At the default threshold, recall for churn is low, meaning the model misses most actual churners. Accuracy alone is therefore not a sufficient measure of usefulness for this imbalanced classification task.
Baseline ANN with validation-selected threshold 0.23
Metric	Test result
Accuracy	0.4619
Precision	0.3319
Recall	0.8559
F1-score	0.4783
ROC-AUC	0.6314
Average precision	0.3872
Test-set size	7,658
Confusion matrix:
```text
[[1648, 3803],
 [ 318, 1889]]
```
The threshold was selected using validation data by maximising F1-score over the tested threshold grid. On the test set, the model identified 1,889 of 2,207 actual churners, while also flagging 3,803 non-churners as potential churners. Lowering the threshold increases churn detection but also increases false positives. ROC-AUC and average precision remain unchanged because changing the threshold does not change the model's underlying score ranking.
Logistic Regression with validation-selected threshold 0.22
Metric	Test result
Accuracy	0.4510
Precision	0.3247
Recall	0.8378
F1-score	0.4680
ROC-AUC	0.6128
Average precision	0.3674
Test-set size	7,658
Confusion matrix:
```text
[[1605, 3846],
 [ 358, 1849]]
```
For reference, Logistic Regression at threshold 0.50 achieved accuracy 0.7080, precision 0.4383, recall 0.0467, F1-score 0.0844, ROC-AUC 0.6128, and average precision 0.3674.
Interpretation
At their respective validation-selected thresholds, the baseline ANN performed somewhat better than Logistic Regression on the reported test metrics. This is a comparison of the current experiments, not evidence of statistical significance or guaranteed business impact.
Important terminology: The ANN result at threshold 0.23 is the baseline ANN with an adjusted classification threshold. It is not a separately trained or hyperparameter-optimised ANN. Results for any future Optimized ANN must remain pending until that experiment has been run and evaluated on the same held-out test records.
Expected Project Structure
Your local project may be organised as follows; update filenames to match the scripts actually present in your project:
```text
Ann_FinalProject/
├── cell2celltrain.csv
├── cell2cell_data_prep.py
├── cell2cell_baseline_ann.py
├── cell2cell_ann_threshold_analysis_with_plots.py
├── cell2cell_logistic_benchmark_with_plots.py
├── outputs/
│   ├── cell2cell_baseline/
│   ├── cell2cell_ann_threshold_analysis/
│   └── cell2cell_logistic_benchmark/
├── README.md
└── requirements.txt
```
Running the Scripts
Run commands from the project directory after installing the required packages and confirming that each script exists under the filename shown.
```bash
python cell2cell_data_prep.py
python cell2cell_baseline_ann.py
python cell2cell_ann_threshold_analysis_with_plots.py --data cell2celltrain.csv --output-dir outputs/cell2cell_ann_threshold_analysis
python cell2cell_logistic_benchmark_with_plots.py --data cell2celltrain.csv --output-dir outputs/cell2cell_logistic_benchmark
```
If a script uses different command-line arguments or filenames, check its `--help` output or source code and adjust the command accordingly. The exact commands depend on the version of each script in your local project.
Dependencies
The workflow uses Python and commonly used data-science / machine-learning packages, including:
pandas
NumPy
scikit-learn
TensorFlow / Keras
Matplotlib
joblib
Install the packages in your Python environment. If the project has a verified `requirements.txt`, use it to reproduce the environment. Record the actual Python and package versions used for the final submission; do not claim that an unverified environment is fully reproducible.
Outputs and Artifacts
Depending on the script version, output folders may contain:
Test predictions
Model and preprocessing artifacts
JSON metric summaries
Confusion matrices
ROC and precision-recall curves
Threshold-analysis plots
Text interpretation notes
Check the generated files after each run and retain the actual output used to support the report. Do not treat a planned output or an illustrative result as an executed experiment.
Limitations and Next Steps
The baseline ANN has modest ranking performance and a low churn recall at the default threshold.
Lower thresholds improve recall but generate many false positives.
Compare Precision@10% and Lift@10% before recommending the model for a constrained retention campaign.
Consider additional ANN optimisation only after the baseline and benchmark comparisons are documented.
Evaluate campaign cost, customer value, and the effectiveness of retention offers before operational deployment.
Validate feature timing and the original dataset documentation; exclusion of identified retention-related fields reduces potential leakage risk but does not prove that every remaining feature is free of leakage.
The current experiments are offline model evaluations. Do not claim production deployment or realised retention uplift unless these have actually been implemented and measured.
References and Disclosure
Include the verified original Cell2Cell dataset citation and cite technical or academic sources actually consulted. The project should also include the required Nexford University AI disclosure form. Describe AI assistance accurately and verify all reported results against executed code outputs.
Suggested methodological references:
Kaufman, S., Rosset, S., Perlich, C., & Stitelman, O. (2012). Leakage in data mining: Formulation, detection, and avoidance. ACM Transactions on Knowledge Discovery from Data, 6(4), Article 15. https://doi.org/10.1145/2382577.2382579
Saito, T., & Rehmsmeier, M. (2015). The precision-recall plot is more informative than the ROC plot when evaluating binary classifiers on imbalanced datasets. PLOS ONE, 10(3), e0118432. https://doi.org/10.1371/journal.pone.0118432
Academic Integrity
This README documents the current project workflow and the results provided from executed experiments. Review all paths, filenames, dependency versions, data-source details, and statements against the actual project before submission. Report only experiments that were run and results that can be reproduced or verified.
