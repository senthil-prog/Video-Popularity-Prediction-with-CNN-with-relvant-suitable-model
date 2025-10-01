#!/usr/bin/env python
# coding: utf-8

# # Video Popularity Prediction - Enhanced Version

# Installing dependencies
import pandas as pd
import seaborn as sns
import numpy as np
from numpy import savetxt
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Function to plot confusion matrix
def plot_confusion_matrix(cm, classes, normalize=False, title='Confusion Matrix', cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    plt.figure(figsize=(10, 8))
    
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')
        
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)
    
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in np.ndindex(cm.shape):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")
    
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()

# Importing training file
print("Loading training data...")
# Update the path to match your environment
data = pd.read_csv("c:/Users/srija/Downloads/Video-Popularity-Prediction-main/Video-Popularity-Prediction-main/Testing and training files/train_meta_df.csv")

# Checking the data
print("Data shape:", data.shape)
print("\nFirst 5 rows:")
print(data.head())

# Check for null values
print("\nChecking for null values:")
print(data.isnull().sum())

# Descriptive statistics
print("\nDescriptive statistics:")
print(data.describe())

# Correlation matrix
print("\nGenerating correlation matrix...")
plt.figure(figsize=(12, 10))
correlation_matrix = data.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Matrix')
plt.tight_layout()
plt.savefig('correlation_matrix.png')
plt.close()

# Feature distributions
print("\nPlotting feature distributions...")
plt.figure(figsize=(15, 10))
for i, column in enumerate(['ratio', 'duration', 'n_likes', 'views', 'n_tags', 'n_formats']):
    plt.subplot(2, 3, i+1)
    sns.histplot(data[column], kde=True)
    plt.title(f'Distribution of {column}')
plt.tight_layout()
plt.savefig('feature_distributions.png')
plt.close()

# Relationship between features and target variable
print("\nPlotting relationships between features and views...")
plt.figure(figsize=(15, 10))
for i, column in enumerate(['ratio', 'duration', 'n_likes', 'n_tags', 'n_formats']):
    plt.subplot(2, 3, i+1)
    sns.scatterplot(x=data[column], y=data['views'])
    plt.title(f'{column} vs views')
plt.tight_layout()
plt.savefig('feature_vs_views.png')
plt.close()

# Removing outliers
print("\nRemoving outliers...")
Q1 = data['views'].quantile(0.25)
Q3 = data['views'].quantile(0.75)
IQR = Q3 - Q1
lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

data_no_outliers = data[(data['views'] >= lower_bound) & (data['views'] <= upper_bound)]
print(f"Original data shape: {data.shape}")
print(f"Data after outlier removal: {data_no_outliers.shape}")

# Prepare data for modeling
X = data_no_outliers.drop(['comp_id', 'views'], axis=1)
y = data_no_outliers['views']

# Split data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)

# Define models for comparison
models = {
    'Linear Regression': LinearRegression(),
    'Ridge Regression': Ridge(alpha=1.0),
    'Lasso Regression': Lasso(alpha=0.1),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42)
}

# Train and evaluate models
print("\nTraining and evaluating models...")
results = {}
predictions = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    
    # Make predictions
    y_pred_train = model.predict(X_train_scaled)
    y_pred_val = model.predict(X_val_scaled)
    
    # Store predictions
    predictions[name] = y_pred_val
    
    # Calculate metrics
    train_rmse = np.sqrt(mean_squared_error(y_train, y_pred_train))
    val_rmse = np.sqrt(mean_squared_error(y_val, y_pred_val))
    train_mae = mean_absolute_error(y_train, y_pred_train)
    val_mae = mean_absolute_error(y_val, y_pred_val)
    train_r2 = r2_score(y_train, y_pred_train)
    val_r2 = r2_score(y_val, y_pred_val)
    
    # Store results
    results[name] = {
        'Train RMSE': train_rmse,
        'Validation RMSE': val_rmse,
        'Train MAE': train_mae,
        'Validation MAE': val_mae,
        'Train R²': train_r2,
        'Validation R²': val_r2
    }
    
    print(f"{name} - Validation RMSE: {val_rmse:.2f}, Validation MAE: {val_mae:.2f}, Validation R²: {val_r2:.4f}")

# Create performance comparison dataframe
performance_df = pd.DataFrame(results).T
print("\nModel Performance Comparison:")
print(performance_df)

# Plot model performance comparison
plt.figure(figsize=(12, 8))
performance_df[['Validation RMSE', 'Validation MAE']].plot(kind='bar')
plt.title('Model Performance Comparison (Lower is Better)')
plt.ylabel('Error')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_performance_rmse_mae.png')
plt.close()

plt.figure(figsize=(10, 6))
performance_df[['Validation R²']].plot(kind='bar', color='green')
plt.title('Model Performance Comparison (Higher is Better)')
plt.ylabel('R² Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('model_performance_r2.png')
plt.close()

# Create confusion matrices for regression by binning the values
print("\nCreating confusion matrices for regression models...")

# Define bins for views
bins = [0, 100, 500, 1000, 5000, float('inf')]
bin_labels = ['0-100', '101-500', '501-1000', '1001-5000', '5000+']

# Convert actual values to bins
y_val_binned = pd.cut(y_val, bins=bins, labels=bin_labels)

# Plot confusion matrices for each model
for name, y_pred in predictions.items():
    # Convert predicted values to bins
    y_pred_binned = pd.cut(y_pred, bins=bins, labels=bin_labels)
    
    # Create confusion matrix
    cm = pd.crosstab(y_val_binned, y_pred_binned).values
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=bin_labels, yticklabels=bin_labels)
    plt.title(f'Confusion Matrix - {name}')
    plt.ylabel('Actual Views')
    plt.xlabel('Predicted Views')
    plt.tight_layout()
    plt.savefig(f'confusion_matrix_{name.replace(" ", "_")}.png')
    plt.close()

# Feature importance for tree-based models
if 'Random Forest' in models:
    print("\nFeature importance from Random Forest:")
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': models['Random Forest'].feature_importances_
    }).sort_values('Importance', ascending=False)
    
    print(feature_importance)
    
    plt.figure(figsize=(12, 8))
    sns.barplot(x='Importance', y='Feature', data=feature_importance)
    plt.title('Feature Importance (Random Forest)')
    plt.tight_layout()
    plt.savefig('feature_importance.png')
    plt.close()

# Select the best model based on validation RMSE
best_model_name = performance_df['Validation RMSE'].idxmin()
best_model = models[best_model_name]
print(f"\nBest model: {best_model_name}")

# Load test data
print("\nLoading test data...")
test = pd.read_csv("c:/Users/srija/Downloads/Video-Popularity-Prediction-main/Video-Popularity-Prediction-main/Testing and training files/public_meta_df.csv")
print(f"Test data shape: {test.shape}")

# Prepare test data
X_test = test.drop(['comp_id'], axis=1)
X_test_scaled = scaler.transform(X_test)

# Make predictions with the best model
print(f"\nMaking predictions with {best_model_name}...")
pred = best_model.predict(X_test_scaled)

# Post-process predictions
pred = np.maximum(pred, 0)  # Ensure non-negative values
pred = np.maximum(pred, 70)  # Minimum view count of 70

# Display sample predictions
print("\nSample predictions:")
print(pred[:10])
print(f"Total number of predictions: {len(pred)}")

# Create submission file
solution = pd.DataFrame({
    'comp_id': test['comp_id'],
    'views': pred
})

solution.to_csv('solution_enhanced.csv', index=False)
print("\nSolution saved to solution_enhanced.csv")

print("\nAnalysis complete! Check the generated visualization files for detailed comparisons.")