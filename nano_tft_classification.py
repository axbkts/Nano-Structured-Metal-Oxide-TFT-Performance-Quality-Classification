import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier

# Suppress joblib warning by setting environment variable
os.environ['LOKY_MAX_CPU_COUNT'] = '1'

def load_data(file_path):
    """Loads the dataset and displays basic information."""
    # Check if file exists to prevent crash
    if not os.path.exists(file_path):
        print(f"Error: The file '{file_path}' was not found.")
        return None

    df = pd.read_csv(file_path)
    
    print("First 5 Rows of the Dataset:")
    print(df.head())
    print("\n" + "="*50 + "\n")
    
    print("Dataset Information:")
    df.info()
    print("\n" + "="*50 + "\n")
    
    print("Missing Values per Column:")
    print(df.isnull().sum())
    print("\n" + "="*50 + "\n")
    
    print("Distribution of Target Variable (Device_Quality):")
    print(df['Device_Quality'].value_counts())
    print("\n" + "="*50 + "\n")
    
    print("Statistical Summary of Numerical Columns:")
    print(df.describe().T)
    
    return df

def preprocess_data(df):
    """Preprocesses and scales the data."""
    # Separate features (X) and target (y)
    X = df.drop('Device_Quality', axis=1)
    y = df['Device_Quality']
    
    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object']).columns
    numerical_cols = X.select_dtypes(include=np.number).columns
    
    print("\nCategorical Columns:")
    print(categorical_cols)
    print("\nNumerical Columns:")
    print(numerical_cols)
    print("\n" + "="*50 + "\n")
    
    # One-Hot Encoding for Categorical Data
    X = pd.get_dummies(X, columns=categorical_cols, drop_first=True)
    
    print("First 5 Rows after One-Hot Encoding:")
    print(X.head())
    print("\n" + "="*50 + "\n")
    
    # Label Encoding for Target Variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    print("Encoded Target Variable (First 10 values):")
    print(y_encoded[:10])
    print("Label Classes and Corresponding Integers:")
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    print("\n" + "="*50 + "\n")
    
    # Feature Scaling (StandardScaler)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    X_processed = pd.DataFrame(X_scaled, columns=X.columns)
    
    print("First 5 Rows after Scaling:")
    print(X_processed.head())
    print("\n" + "="*50 + "\n")
    print("Data preprocessing completed. Data ready for modeling.")
    
    return X_processed, y_encoded, le

def split_data(X, y):
    """Splits data into training and testing sets."""
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    print("\nDataset Dimensions:")
    print("Total X shape: ", X.shape)
    print("Training set (X_train) shape: ", X_train.shape)
    print("Test set (X_test) shape: ", X_test.shape)
    print("Training set (y_train) shape: ", y_train.shape)
    print("Test set (y_test) shape: ", y_test.shape)
    print("\n" + "="*50 + "\n")
    
    # Check class distribution in train and test sets
    train_counts = np.bincount(y_train)
    test_counts = np.bincount(y_test)
    print("Class distribution in Training Set:", train_counts)
    print("Class distribution in Test Set:", test_counts)
    print("\n" + "="*50 + "\n")
    print("Data splitting completed.")
    
    return X_train, X_test, y_train, y_test

def initialize_models():
    """Initializes all classification models."""
    models = {
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=5),
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Naive Bayes": GaussianNB(),
        "Decision Trees": DecisionTreeClassifier(random_state=42),
        "Support Vector Machines": SVC(random_state=42),
        "Random Forest": RandomForestClassifier(random_state=42, n_jobs=1)
    }
    
    algorithm_formulas = {
        "K-Nearest Neighbors": r"$\hat{y} = \arg\max_c \sum_{i \in K-NN} \mathbb{1}(y_i = c)$" + "\n" + "Decision based on vote of K nearest neighbors.",
        "Logistic Regression": r"$P(y=1|x) = \frac{1}{1 + e^{-w^T x - b}}$" + "\n" + "Classification using Sigmoid function.",
        "Naive Bayes": r"$P(y|x) = \frac{P(x|y) \cdot P(y)}{P(x)}$" + "\n" + "Conditional probability calculation using Bayes' theorem.",
        "Decision Trees": r"$\text{Gain} = H(D) - \sum_v \frac{|D_v|}{|D|} H(D_v)$" + "\n" + "Best split selection using Information Gain.",
        "Support Vector Machines": r"$f(x) = \text{sign}\left(\sum_i \alpha_i y_i K(x_i, x) + b\right)$" + "\n" + "Maximal margin classification using Kernel function.",
        "Random Forest": r"$\hat{y} = \arg\max_c \sum_{t=1}^{T} \mathbb{1}(T_t(x) = c)$" + "\n" + "Decision via majority voting of T decision trees."
    }
    
    return models, algorithm_formulas

def train_and_evaluate_models(models, X_train, X_test, y_train, y_test, label_names):
    """Trains and evaluates models."""
    results_list = []
    report_data_list = []
    train_test_results = []
    
    # Plotting Confusion Matrices
    fig, axes = plt.subplots(3, 2, figsize=(14, 20), num='Confusion Matrices')
    fig.suptitle('Confusion Matrices of All Models', fontsize=16, fontweight='bold', color='#1f77b4')
    axes = axes.flatten()
    
    # Iterate over models
    for i, (model_name, model) in enumerate(models.items()):
        print("\n" + "="*20 + f" {model_name} " + "="*20)
        
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        
        # Test set accuracy
        accuracy = accuracy_score(y_test, y_pred)
        error_rate = 1 - accuracy
        
        # Training set accuracy (to check for overfitting)
        y_train_pred = model.predict(X_train)
        train_accuracy = accuracy_score(y_train, y_train_pred)
        
        # Store results
        train_test_results.append({
            'Model': model_name,
            'Train Accuracy': train_accuracy,
            'Test Accuracy': accuracy,
            'Train Error Rate': 1 - train_accuracy,
            'Test Error Rate': error_rate
        })
        
        results_list.append({'Model': model_name, 'Accuracy': accuracy, 'Error Rate': error_rate})
        
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Error Rate: {error_rate:.4f}\n")
        print("Classification Report:")
        print(classification_report(y_test, y_pred, target_names=label_names))
        
        # Collect report data for plotting
        report_dict = classification_report(y_test, y_pred, target_names=label_names, output_dict=True)
        for class_name, metrics in report_dict.items():
            if class_name in label_names:
                report_data_list.append({
                    'Model': model_name, 'Class': class_name,
                    'precision': metrics['precision'], 'recall': metrics['recall'], 'f1-score': metrics['f1-score']
                })
                
        cm = confusion_matrix(y_test, y_pred)
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=label_names, yticklabels=label_names, ax=axes[i])
        axes[i].set_title(model_name)
        axes[i].set_ylabel('True Label')
        axes[i].set_xlabel('Predicted Label')
    
    fig.subplots_adjust(top=0.92, hspace=0.5, wspace=0.3)
    plt.show()
    
    return results_list, report_data_list, train_test_results

def create_comparison_plots(results_list, report_data_list, train_test_results, label_names):
    """Generates comparison plots."""
    
    # 1. Accuracy Comparison Plot
    results_df = pd.DataFrame(results_list).sort_values(by='Accuracy', ascending=False).reset_index(drop=True)
    print("\n" + "="*20 + " Model Performance Summary " + "="*20)
    print(results_df)
    
    plt.figure(figsize=(12, 7), num='Model Accuracy Comparison')
    barplot = sns.barplot(x='Accuracy', y='Model', data=results_df, hue='Model', palette='RdYlGn', saturation=0.85, legend=False)
    plt.title('Accuracy Comparison of Classification Models', fontsize=14, fontweight='bold', pad=20)
    plt.xlabel('Accuracy Score', fontsize=12, fontweight='bold')
    plt.ylabel('Classification Model', fontsize=12, fontweight='bold')
    plt.xlim(0, 1.0)
    barplot.grid(axis='x', alpha=0.3, linestyle='--')
    for index, row in results_df.iterrows():
        barplot.text(row.Accuracy + 0.01, index, f'{row.Accuracy:.4f}', color='black', ha="left", fontweight='bold')
    plt.show()
    
    # 2. Detailed Class-wise Metrics Plot
    report_df = pd.DataFrame(report_data_list)
    report_df_long = report_df.melt(id_vars=['Model', 'Class'], var_name='Metric', value_name='Score')
    
    g = sns.catplot(
        data=report_df_long, x='Model', y='Score', hue='Metric',
        col='Class', kind='bar', height=5, aspect=1.1, palette='muted',
        col_order=['High', 'Medium', 'Low'] 
    )
    g.fig.canvas.manager.set_window_title('Class-wise Metric Comparison')
    g.fig.suptitle('Class-wise Metric Comparison (Precision, Recall, F1-Score)', fontsize=14, fontweight='bold', y=1.00)
    g.set_xticklabels(rotation=45, ha='right', fontsize=10)
    g.set_ylabels('Score', fontsize=11, fontweight='bold')
    g.set_xlabels('Model', fontsize=11, fontweight='bold')
    g.set_titles("Class: {col_name}", fontsize=12, fontweight='bold', color='#1f77b4')
    for ax in g.axes.flat:
        ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    sns.move_legend(g, "upper left")
    g.fig.tight_layout(rect=[0, 0, 1, 0.97])
    plt.show()
    
    # 3. Train vs Test Accuracy Comparison Plot
    train_test_df = pd.DataFrame(train_test_results).sort_values(by='Test Accuracy', ascending=False).reset_index(drop=True)
    
    plt.figure(figsize=(14, 7), num='Train vs Test Accuracy')
    x = np.arange(len(train_test_df))
    width = 0.2
    
    colors = ['#2ecc71', '#27ae60', "#d14e3f", '#c0392b']
    plt.bar(x - 1.5*width, train_test_df['Train Accuracy'], width, label='Train Accuracy', alpha=0.85, color=colors[0])
    plt.bar(x - 0.5*width, train_test_df['Test Accuracy'], width, label='Test Accuracy', alpha=0.85, color=colors[1])
    plt.bar(x + 0.5*width, train_test_df['Train Error Rate'], width, label='Train Error Rate', alpha=0.85, color=colors[2])
    plt.bar(x + 1.5*width, train_test_df['Test Error Rate'], width, label='Test Error Rate', alpha=0.85, color=colors[3])
    
    plt.xlabel('Classification Model', fontsize=12, fontweight='bold')
    plt.ylabel('Score / Rate', fontsize=12, fontweight='bold')
    plt.title('Train vs Test: Accuracy & Error Rate Comparison', fontsize=14, fontweight='bold', pad=20)
    plt.xticks(x, train_test_df['Model'], rotation=45, ha='right', fontsize=10)
    plt.ylim(0, 1.0)
    plt.legend(loc='upper right', fontsize=10, framealpha=0.95)
    plt.grid(axis='y', alpha=0.3, linestyle='--')
    plt.tight_layout()
    plt.show()

def main():
    """Main program execution."""
    # Load dataset
    file_name = 'Nano_Structured_Metal_Oxide_TFT_Dataset.csv'   # Update with your actual file path!
    df = load_data(file_name)
    
    if df is not None:
        # Preprocess data
        X_processed, y_encoded, label_encoder = preprocess_data(df)
        
        # Split data
        X_train, X_test, y_train, y_test = split_data(X_processed, y_encoded)
        
        # Initialize models
        models, algorithm_formulas = initialize_models()
        
        # Train and evaluate
        label_names = label_encoder.classes_
        results_list, report_data_list, train_test_results = train_and_evaluate_models(
            models, X_train, X_test, y_train, y_test, label_names
        )
        
        # Create comparison plots
        create_comparison_plots(results_list, report_data_list, train_test_results, label_names)

if __name__ == "__main__":
    main()