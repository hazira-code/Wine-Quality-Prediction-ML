# Wine-Quality-Prediction-ML
🍇 Built an ML system that converts wine’s chemical composition into quality insights, classifying it as Good or Bad. 📈 Performed EDA to identify key taste factors, applied feature scaling, trained multiple models, and optimized performance using hyperparameter tuning for reliable predictions. 🍷
From Chemistry to Cheers
Wine Quality Intelligence using Machine Learning

An end-to-end Machine Learning project that transforms laboratory chemical measurements into meaningful wine quality predictions. This system classifies wine as Good or Bad, simulating how data-driven models can support quality assessment in real-world production and resale scenarios.

🎯 Project Goal

To design a robust binary classification system that predicts wine quality while demonstrating the complete Machine Learning workflow—from raw data exploration to optimized, deployment-ready models.

🧠 Problem Definition

Learning Approach: Supervised Learning

ML Task: Binary Classification

Target Variable: quality_label

Class Mapping
Label	Description
1	Good Wine (quality ≥ 7)
0	Bad Wine (quality < 7)
📊 Dataset Snapshot

Physicochemical attributes such as acidity, pH, alcohol content, etc.

Fully numerical and well-structured dataset

No missing values

Naturally imbalanced target classes

Ideal for statistical analysis and ML modeling

🔍 Exploratory Data Analysis

EDA was performed to understand:

Distribution of wine quality scores

Degree of class imbalance

Feature scaling differences

Relationships between chemical properties

Visual insights directly influenced preprocessing choices and model selection.

⚙️ Data Preparation Pipeline

Feature and target separation

80/20 train-test split

Feature standardization using StandardScaler

Scikit-learn Pipelines to ensure reproducibility and prevent data leakage

🤖 Models Evaluated

Multiple algorithms were trained and compared to identify the most reliable classifier:

Logistic Regression

K-Nearest Neighbors (KNN)

Decision Tree

Random Forest

Support Vector Machine (SVM)

All models were evaluated using identical metrics for fair comparison.

📈 Performance Evaluation

Accuracy Score

Confusion Matrix

Precision, Recall, F1-Score

Cross-validation performance

These metrics ensured both correctness and stability across unseen data.

🧪 Model Optimization

Hyperparameter tuning using GridSearchCV

Integrated with Pipelines

5-fold cross-validation

Reduced overfitting and improved generalization

The final model was selected based on validation performance, not just test accuracy.

🏆 Final Outcome

Best-performing model identified through systematic benchmarking

Tuned classifier achieved stronger and more consistent predictions

Final pipeline is ready for deployment or extension (API / Web App)

🛠️ Tools & Technologies

Language: Python

Data Handling: Pandas, NumPy

Visualization: Matplotlib, Seaborn

Machine Learning: Scikit-learn

Optimization: Pipelines, GridSearchCV

📁 Repository Structure
├── wine_quality_prediction.py
├── winequality.csv
├── model_comparison.png
├── quality_distribution.png
├── binary_classification.png
└── README.md

🌱 Future Enhancements

Handle class imbalance using SMOTE

Feature importance analysis for interpretability

Deploy using Flask or Streamlit

Extend to multi-class quality prediction
