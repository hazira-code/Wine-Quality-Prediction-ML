"""
Wine Quality Prediction - End-to-End Machine Learning Project
Complete implementation with all tasks
"""

# ============================================================================
# TASK 1: Load and Understand the Dataset
# ============================================================================

# Import required libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline

# Set display options
pd.set_option('display.max_columns', None)
plt.style.use('seaborn-v0_8-darkgrid')

# Load the dataset
df = pd.read_csv('winequality.csv')

print("=" * 80)
print("TASK 1: LOAD AND UNDERSTAND THE DATASET")
print("=" * 80)

# Display first 5 rows
print("\nFirst 5 rows:")
print(df.head())

# Display last 5 rows
print("\nLast 5 rows:")
print(df.tail())

# Display random 5 rows
print("\nRandom 5 rows:")
print(df.sample(5))

print("\n--- Explanation ---")
print("This dataset contains chemical properties of wine samples.")
print("Each row represents one wine sample with its chemical measurements.")
print("The 'quality' column is our target variable representing wine quality score.")

# ============================================================================
# TASK 2: Basic Data Inspection
# ============================================================================

print("\n" + "=" * 80)
print("TASK 2: BASIC DATA INSPECTION")
print("=" * 80)

# Column names
print("\nColumn names:")
print(df.columns.tolist())

# Shape of dataset
print(f"\nNumber of rows: {df.shape[0]}")
print(f"Number of columns: {df.shape[1]}")

# Data types
print("\nData types:")
print(df.dtypes)

# Summary statistics
print("\nSummary Statistics:")
print(df.describe())

print("\n--- Explanation ---")
print("Data inspection is crucial before training ML models because:")
print("- It helps understand the scale and distribution of features")
print("- Identifies potential issues like missing values or outliers")
print("- Reveals if features need preprocessing like scaling")
print("- Helps decide which algorithms might work best")

# ============================================================================
# TASK 3: Missing Values Analysis
# ============================================================================

print("\n" + "=" * 80)
print("TASK 3: MISSING VALUES ANALYSIS")
print("=" * 80)

# Check for missing values
print("\nMissing values per column:")
print(df.isnull().sum())

print(f"\nTotal missing values: {df.isnull().sum().sum()}")

print("\n--- Explanation ---")
if df.isnull().sum().sum() == 0:
    print("No missing values found in this dataset.")
else:
    print("Missing values found!")

print("\nHow to handle missing values in real-world projects:")
print("1. Drop rows/columns if missing data is very small")
print("2. Fill with mean/median for numerical columns")
print("3. Fill with mode for categorical columns")
print("4. Use advanced imputation techniques like KNN imputer")
print("5. Investigate why data is missing before deciding")

# ============================================================================
# TASK 4: Exploratory Data Analysis (EDA)
# ============================================================================

print("\n" + "=" * 80)
print("TASK 4: EXPLORATORY DATA ANALYSIS (EDA)")
print("=" * 80)

# Value counts of quality
print("\nQuality distribution:")
print(df['quality'].value_counts().sort_index())

# Plot count plot
plt.figure(figsize=(10, 6))
sns.countplot(data=df, x='quality', palette='viridis')
plt.title('Distribution of Wine Quality Scores', fontsize=14, fontweight='bold')
plt.xlabel('Quality Score', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.tight_layout()
plt.savefig('quality_distribution.png', dpi=300)
plt.show()

print("\n--- Observations ---")
print("1. Most wines have quality scores of 5 and 6 (middle range)")
print("2. Very few wines have extremely low (3) or high (8-9) quality")
print("3. The data is imbalanced - more medium quality wines than extreme ones")
print("\n---Dataset Observations---")
print("Most wines have quality scores of 5 and 6 (middle range)")
print("The data is imbalanced - more medium quality wines than extreme ones")
print("\n--- How EDA helps ---")
print("EDA helps us understand:")
print("- Data distribution and potential imbalances")
print("- Whether we need to handle class imbalance")
print("- Which features might be important")
print("- If we need to transform the target variable")

# ============================================================================
# TASK 5: Convert to Binary Classification
# ============================================================================

print("\n" + "=" * 80)
print("TASK 5: CONVERT TO BINARY CLASSIFICATION")
print("=" * 80)

# Create binary quality label
df['quality_label'] = df['quality'].apply(lambda x: 1 if x >= 7 else 0)

print("\nNew quality_label distribution:")
print(df['quality_label'].value_counts())
print(f"\nGood Wine (1): {df['quality_label'].sum()}")
print(f"Bad Wine (0): {(df['quality_label'] == 0).sum()}")

# Visualize binary classification
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='quality_label', palette='coolwarm')
plt.title('Binary Classification: Good vs Bad Wine', fontsize=14, fontweight='bold')
plt.xlabel('Quality Label (0=Bad, 1=Good)', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks([0, 1], ['Bad Wine (0)', 'Good Wine (1)'])
plt.tight_layout()
plt.savefig('binary_classification.png', dpi=300)
plt.show()

print("\n--- Explanation ---")
print("Binary classification is more practical because:")
print("1. Real-world decisions are often binary (buy/don't buy, accept/reject)")
print("2. Easier to interpret and explain to stakeholders")
print("3. More reliable predictions with limited data")
print("4. Simpler deployment in production systems")

# ============================================================================
# TASK 6: Feature and Target Separation
# ============================================================================

print("\n" + "=" * 80)
print("TASK 6: FEATURE AND TARGET SEPARATION")
print("=" * 80)

# Separate features and target
X = df.drop(['quality', 'quality_label'], axis=1)
y = df['quality_label']

print(f"\nFeatures (X) shape: {X.shape}")
print(f"Target (y) shape: {y.shape}")

print("\nFeature columns:")
print(X.columns.tolist())

print("\n--- Explanation ---")
print("We must NOT use the original 'quality' column as a feature because:")
print("1. It directly reveals the answer we're trying to predict")
print("2. This would cause data leakage - using future information")
print("3. The model would learn to cheat instead of learning patterns")
print("4. In real-world deployment, we won't have 'quality' available")

# ============================================================================
# TASK 7: Train-Test Split
# ============================================================================

print("\n" + "=" * 80)
print("TASK 7: TRAIN-TEST SPLIT")
print("=" * 80)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

print(f"\nTraining set size: {X_train.shape[0]} samples ({X_train.shape[0]/len(df)*100:.1f}%)")
print(f"Testing set size: {X_test.shape[0]} samples ({X_test.shape[0]/len(df)*100:.1f}%)")

print("\n--- Explanation ---")
print("We split data into training and testing sets because:")
print("1. Training set: Used to teach the model patterns")
print("2. Testing set: Used to evaluate model on unseen data")
print("3. This simulates real-world scenario where model sees new data")
print("\nProblem with using same data for training and testing:")
print("- Model memorizes the data instead of learning patterns")
print("- This is called 'overfitting'")
print("- Performance looks great but fails on new real-world data")

# ============================================================================
# TASK 8: Feature Scaling
# ============================================================================

print("\n" + "=" * 80)
print("TASK 8: FEATURE SCALING")
print("=" * 80)

# Initialize scaler
scaler = StandardScaler()

# Fit on training data and transform both
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print("\nBefore scaling (first 3 samples):")
print(X_train.head(3))

print("\nAfter scaling (first 3 samples):")
print(pd.DataFrame(X_train_scaled, columns=X.columns).head(3))

print("\n--- Explanation ---")
print("Feature scaling is important because:")
print("1. Features have different ranges (e.g., pH: 2-4, alcohol: 8-15)")
print("2. Some algorithms are sensitive to feature magnitudes")
print("\nModels that NEED scaling:")
print("- KNN, SVM, Logistic Regression (distance/gradient based)")
print("- Neural Networks (gradient descent optimization)")
print("\nModels that DON'T need scaling:")
print("- Decision Trees, Random Forests (tree-based, use splits)")

# ============================================================================
# TASK 9: Model Training
# ============================================================================

print("\n" + "=" * 80)
print("TASK 9: MODEL TRAINING")
print("=" * 80)

# Dictionary to store models and predictions
models = {}
predictions = {}

# 1. Logistic Regression
print("\n1. Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_scaled, y_train)
models['Logistic Regression'] = lr
predictions['Logistic Regression'] = lr.predict(X_test_scaled)
print("   → Finds a linear decision boundary using probability")

# 2. K-Nearest Neighbors
print("\n2. Training K-Nearest Neighbors...")
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(X_train_scaled, y_train)
models['KNN'] = knn
predictions['KNN'] = knn.predict(X_test_scaled)
print("   → Classifies based on majority vote of 5 nearest neighbors")

# 3. Decision Tree
print("\n3. Training Decision Tree...")
dt = DecisionTreeClassifier(random_state=42)
dt.fit(X_train, y_train)
models['Decision Tree'] = dt
predictions['Decision Tree'] = dt.predict(X_test)
print("   → Makes decisions using if-else rules learned from data")

# 4. Random Forest
print("\n4. Training Random Forest...")
rf = RandomForestClassifier(random_state=42, n_estimators=100)
rf.fit(X_train, y_train)
models['Random Forest'] = rf
predictions['Random Forest'] = rf.predict(X_test)
print("   → Combines predictions from 100 decision trees for better accuracy")

# 5. Support Vector Machine
print("\n5. Training Support Vector Machine...")
svm = SVC(random_state=42)
svm.fit(X_train_scaled, y_train)
models['SVM'] = svm
predictions['SVM'] = svm.predict(X_test_scaled)
print("   → Finds the best hyperplane to separate classes with maximum margin")

# ============================================================================
# TASK 10: Model Evaluation and Comparison
# ============================================================================

print("\n" + "=" * 80)
print("TASK 10: MODEL EVALUATION AND COMPARISON")
print("=" * 80)

# Calculate accuracies
results = []
for model_name, y_pred in predictions.items():
    accuracy = accuracy_score(y_test, y_pred)
    results.append({'Model': model_name, 'Accuracy': accuracy})
    print(f"\n{model_name}:")
    print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")

# Create comparison DataFrame
results_df = pd.DataFrame(results)
results_df = results_df.sort_values('Accuracy', ascending=False)

print("\n" + "-" * 50)
print("MODEL COMPARISON TABLE")
print("-" * 50)
print(results_df.to_string(index=False))

# Plot comparison
plt.figure(figsize=(10, 6))
sns.barplot(data=results_df, x='Accuracy', y='Model', palette='rocket')
plt.title('Model Performance Comparison', fontsize=14, fontweight='bold')
plt.xlabel('Accuracy', fontsize=12)
plt.ylabel('Model', fontsize=12)
plt.xlim(0, 1)
for i, row in enumerate(results_df.itertuples()):
    plt.text(row.Accuracy + 0.01, i, f'{row.Accuracy:.4f}', va='center')
plt.tight_layout()
plt.savefig('model_comparison.png', dpi=300)
plt.show()

best_model = results_df.iloc[0]['Model']
best_accuracy = results_df.iloc[0]['Accuracy']

print(f"\n--- Best Model ---")
print(f"Model: {best_model}")
print(f"Accuracy: {best_accuracy:.4f}")

print("\n--- Why this model performed best ---")
print(f"{best_model} likely performed best because:")
if 'Random Forest' in best_model:
    print("- It combines multiple decision trees reducing overfitting")
    print("- Handles non-linear relationships well")
    print("- Robust to outliers and doesn't need feature scaling")
elif 'SVM' in best_model:
    print("- It finds optimal decision boundary with maximum margin")
    print("- Effective in high-dimensional spaces")
    print("- Good at handling non-linear patterns with kernel trick")
else:
    print("- It captures the underlying patterns in the data effectively")
    print("- The dataset characteristics align well with this algorithm")

# ============================================================================
# TASK 11: Pipeline and Hyperparameter Tuning
# ============================================================================

print("\n" + "=" * 80)
print("TASK 11: PIPELINE AND HYPERPARAMETER TUNING")
print("=" * 80)

# Create pipeline
pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('classifier', LogisticRegression(random_state=42, max_iter=1000))
])

# Define parameter grid
param_grid = {
    'classifier__C': [0.01, 0.1, 1, 10, 100],
    'classifier__solver': ['liblinear', 'lbfgs']
}

print("\nPerforming Grid Search with Cross-Validation...")
print(f"Testing {len(param_grid['classifier__C']) * len(param_grid['classifier__solver'])} combinations")

# Perform grid search
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

grid_search.fit(X_train, y_train)

print("\n--- Results ---")
print(f"Best Parameters: {grid_search.best_params_}")
print(f"Best Cross-Validation Score: {grid_search.best_score_:.4f}")

# Test the best model
best_pipeline = grid_search.best_estimator_
y_pred_tuned = best_pipeline.predict(X_test)
tuned_accuracy = accuracy_score(y_test, y_pred_tuned)
print(f"Test Set Accuracy: {tuned_accuracy:.4f}")

print("\n--- Explanation ---")
print("Pipelines are used in real-world ML systems because:")
print("1. Ensure consistent preprocessing steps in training and deployment")
print("2. Prevent data leakage by proper fit/transform sequence")
print("3. Make code cleaner and easier to maintain")
print("4. Enable easy experimentation with different preprocessing steps")

print("\nHyperparameter tuning improves performance because:")
print("1. Default parameters are rarely optimal for specific datasets")
print("2. Finds the best configuration through systematic search")
print("3. Uses cross-validation to prevent overfitting")
print("4. Balances model complexity and generalization")

# ============================================================================
# TASK 12: Final Conclusion
# ============================================================================

print("\n" + "=" * 80)
print("TASK 12: FINAL CONCLUSION")
print("=" * 80)

print("\n--- Dataset Understanding ---")
print("The wine quality dataset contains chemical properties of wine samples")
print(f"with {df.shape[0]} samples and {df.shape[1]-2} features.")
print("We converted it into a binary classification problem (good vs bad wine).")

print("\n--- Important EDA Observations ---")
print("1. Data has no missing values - good quality dataset")
print("2. Quality scores are imbalanced - mostly medium quality wines")
print("3. After binary conversion, we have class imbalance")
print("4. Features have different scales requiring standardization")

print("\n--- Best Performing Model ---")
print(f"The {best_model} achieved the highest accuracy of {best_accuracy:.4f}")
print("After hyperparameter tuning, we achieved:", tuned_accuracy)

print("\n--- Key Learnings ---")
print("1. Importance of data preprocessing and EDA before modeling")
print("2. Different algorithms have different requirements (scaling)")
print("3. Model comparison helps select the best algorithm")
print("4. Hyperparameter tuning can significantly improve performance")
print("5. Pipelines ensure reproducibility and prevent data leakage")

print("\n--- Real-World Application ---")
print("This project mirrors real-world ML applications:")
print("1. Start with business problem (predict wine quality)")
print("2. Explore and understand data thoroughly")
print("3. Preprocess and engineer features")
print("4. Train multiple models and compare")
print("5. Optimize best model with hyperparameter tuning")
print("6. Evaluate on test set to estimate real-world performance")
print("7. Use pipelines for production deployment")

print("\n" + "=" * 80)
print("PROJECT COMPLETED SUCCESSFULLY!")
print("=" * 80)

# Display final summary
print("\n--- Summary Statistics ---")
print(f"Total samples: {len(df)}")
print(f"Features used: {len(X.columns)}")
print(f"Training samples: {len(X_train)}")
print(f"Testing samples: {len(X_test)}")
print(f"Number of models trained: {len(models)}")
print(f"Best model: {best_model}")
print(f"Best accuracy: {best_accuracy:.4f}")
print(f"Tuned accuracy: {tuned_accuracy:.4f}")
