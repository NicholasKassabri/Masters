
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score, roc_auc_score
import pickle
import category_encoders as ce
from keras.models import Sequential
from keras.layers import Dense
from xgboost import XGBClassifier

#get_ipython().run_line_magic('matplotlib', 'notebook')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

import numpy as np



# Function for Data Preprocessing
def preprocess_data(filepath):
    df = pd.read_csv(filepath)
    print('Rows: ', df.shape[0], ' Columns: ', df.shape[1])

    # Remove duplicates
    df.drop_duplicates(inplace=True)

    # Mapping categorical data to numeric
    mappings = {
        'Married/Single': {'single': 0, 'married': 1},
        'House_Ownership': {'norent_noown': 0, 'rented': 1, 'owned': 2},
        'Car_Ownership': {'no': 0, 'yes': 1}
    }
    df.replace(mappings, inplace=True)

    # Count encoding for high cardinality features
    high_card_features = ['Profession', 'CITY', 'STATE']
    count_encoder = ce.CountEncoder()
    count_encoded = count_encoder.fit_transform(df[high_card_features])
    df = df.join(count_encoded.add_suffix("_count"))
    df.drop(labels=high_card_features + ['Id'], axis=1, inplace=True)
    
    return df

# Function to Evaluate Model Performance and Store Results
def evaluate_model(model_name, y_true, y_pred, results_df):
    accuracy = accuracy_score(y_true, y_pred)
    recall = recall_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)  # Set zero_division to 0
    f1 = f1_score(y_true, y_pred, zero_division=0)  # Set zero_division to 0
    auc = roc_auc_score(y_true, y_pred)

    new_row = pd.DataFrame({
        'Model': [model_name],
        'Accuracy': [accuracy],
        'Recall': [recall],
        'Precision': [precision],
        'F1 Score': [f1],
        'AUC Score': [auc]
    })

    results_df = pd.concat([results_df, new_row], ignore_index=True)
    return results_df


# Function for Logistic Regression with Grid Search
def logistic_regression_grid_search(X_train, y_train, X_test, y_test, results_df):
    parameters = {'C': [0.001, 0.01, 0.1, 1, 10, 100]}
    logreg = LogisticRegression()
    clf = GridSearchCV(logreg, parameters)
    clf.fit(X_train, y_train)

    logreg_clf = clf.best_estimator_

    y_pred = logreg_clf.predict(X_test)
    y_pred_prob = logreg_clf.predict_proba(X_test)[:, 1]

    results_df = evaluate_model("Logistic Regression (Grid Search)", y_test, y_pred, results_df)
    return results_df, clf.best_params_, y_pred_prob, logreg_clf


# Function for Random Forest with Grid Search
def random_forest_grid_search(X_train, y_train, X_test, y_test, results_df):
    parameters = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20, None],
        'min_samples_split': [2, 5]
    }
    rfc = RandomForestClassifier(random_state=42)
    clf = GridSearchCV(rfc, parameters)
    clf.fit(X_train, y_train)

    # Get the best trained model
    rfc_clf = clf.best_estimator_

    y_pred = rfc_clf.predict(X_test)
    y_pred_prob = rfc_clf.predict_proba(X_test)[:, 1]

    results_df = evaluate_model("Random Forest (Grid Search)", y_test, y_pred, results_df)
    return results_df, clf.best_params_, y_pred_prob, rfc_clf


# Function for Artificial Neural Network
def artificial_neural_network(X_train, y_train, X_test, y_test, results_df):
    model = Sequential()
    model.add(Dense(12, input_dim=X_train.shape[1], activation='relu'))
    model.add(Dense(8, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))
    
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    model.fit(X_train, y_train, epochs=100, batch_size=10, verbose=0)

    y_pred_prob = model.predict(X_test).ravel()
    y_pred = np.round(y_pred_prob)

    results_df = evaluate_model("Artificial Neural Network", y_test, y_pred, results_df)
    return results_df, y_pred_prob, model

# XGBoost with GridSearchCV
def xgboost_grid_search(X_train, y_train, X_test, y_test, results_df):
    parameters = {
        'max_depth': [3, 5, 7],
        'n_estimators': [100, 200],
        'learning_rate': [0.01, 0.1, 0.3]
    }
    xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
    clf = GridSearchCV(xgb, parameters, scoring='roc_auc')
    clf.fit(X_train, y_train)

    best_xgb = clf.best_estimator_  # Get the best trained model
    y_pred = best_xgb.predict(X_test)
    y_pred_prob = best_xgb.predict_proba(X_test)[:, 1]

    results_df = evaluate_model("XGBoost (Grid Search)", y_test, y_pred, results_df)
    return results_df, clf.best_params_, y_pred_prob, best_xgb


# Function to plot ROC AUC for each model
def plot_roc_auc(roc_data, y_test, X_test_scaled, X_test):
    plt.figure(figsize=(12, 8))

    for i, (model_name, (model, y_pred_prob)) in enumerate(roc_data.items()):
        # Determine the right test set
        print(model_name)
        test_set = X_test_scaled if 'scaled' in model_name else X_test
        
        if model is not None:
            # Use predict_proba if the model has it
            fpr, tpr, _ = roc_curve(y_test, model.predict_proba(test_set)[:, 1])
        else:
            # Directly use y_pred_prob if it's the ANN model
            fpr, tpr, _ = roc_curve(y_test, y_pred_prob)

        roc_auc = auc(fpr, tpr)
        plt.subplot(2, 2, i+1)
        plt.plot(fpr, tpr, label=f'{model_name} (AUC = {roc_auc:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title(f'ROC Curve - {model_name}')
        plt.legend(loc='lower right')

    plt.tight_layout()
    plt.show()
    
def plot_roc_curve_ann(y_test, y_pred_prob):
    # Compute ROC curve and ROC area for the ANN model
    fpr, tpr, _ = roc_curve(y_test, y_pred_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='ROC curve (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - Artificial Neural Network')
    plt.legend(loc="lower right")
    plt.show()
    
def plot_combined_metrics_histogram(results_df):
    # Define the metrics and number of models
    metrics = ['Recall', 'Precision', 'F1 Score']
    n_models = len(results_df)
    
    # Create bar width, positions and colors
    barWidth = 0.25
    r1 = np.arange(n_models)
    r2 = [x + barWidth for x in r1]
    r3 = [x + barWidth for x in r2]
    colors = ['#ff9999','#66b3ff','#c299ff']  # Adjusted color for F1 Score to light purple

    # Create subplots
    plt.figure(figsize=(10, 6))

    # Create bars for each metric
    plt.bar(r1, results_df['Recall'], color=colors[0], width=barWidth, edgecolor='gray', label='Recall')
    plt.bar(r2, results_df['Precision'], color=colors[1], width=barWidth, edgecolor='gray', label='Precision')
    plt.bar(r3, results_df['F1 Score'], color=colors[2], width=barWidth, edgecolor='gray', label='F1 Score')

    # Add xticks on the middle of the group bars
    plt.xlabel('Model', fontweight='bold')
    plt.ylabel('Scores')
    plt.xticks([r + barWidth for r in range(n_models)], results_df['Model'])

    plt.title('Model Comparison - Recall, Precision, and F1 Score')
    plt.legend()

    plt.tight_layout()
    plt.show()
    
def format_results(df):
    return df.style.background_gradient(cmap='coolwarm').format("{:.2%}", subset=['Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC Score'])


# Preprocessing
file_path = "/Users/omarsyed/Dropbox/DATA 5000/Project/Training Data.csv"

df = preprocess_data(file_path)

X = df.drop("Risk_Flag", axis=1)
y = df["Risk_Flag"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=7)

# Scaling - Important for Logistic Regression and ANN
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# DataFrame to store results
results_df = pd.DataFrame(columns=['Model', 'Accuracy', 'Recall', 'Precision', 'F1 Score', 'AUC Score'])



# Logistic Regression with Grid Search
results_df, logreg_params, logreg_pred_prob, logreg_clf = logistic_regression_grid_search(X_train_scaled, y_train, X_test_scaled, y_test, results_df)
print("Logistic Regression has Completed")

# Random Forest with Grid Search
results_df, rf_params, rf_pred_prob, rf_clf = random_forest_grid_search(X_train, y_train, X_test, y_test, results_df)
print("Random Forest has Completed")

# Artificial Neural Network
results_df, ann_pred_prob,ann_model = artificial_neural_network(X_train_scaled, y_train, X_test_scaled, y_test, results_df)
print("ANN has Completed")

# XGBoost
results_df, xgb_params, xgb_pred_prob, xgb_clf = xgboost_grid_search(X_train, y_train, X_test, y_test, results_df)
print("XGBoost has Completed")

display(format_results(results_df))



roc_data = {
    'Logistic Regression': (logreg_clf, logreg_pred_prob),
    'Random Forest': (rf_clf, rf_pred_prob),
    'Artificial Neural Network': (ann_model, ann_pred_prob),
    'XGBoost': (xgb_clf, xgb_pred_prob)
}
# print(roc_data)
# print(y_test)
# print(X_train_scaled)

plot_roc_auc(roc_data, y_test, X_train_scaled, X_test)




plot_combined_metrics_histogram(results_df)




plot_roc_curve_ann(y_test, ann_pred_prob)




# Random Forest with Grid Search
results_df, rf_params, trained_models['Random Forest'] = random_forest_grid_search(X_train, y_train, X_test, y_test, results_df)
#print("Best Parameters for Random Forest:", rf_params)
#print(results_df)



# Artificial Neural Network
results_df, ann_model = artificial_neural_network(X_train_scaled, y_train, X_test_scaled, y_test, results_df)
trained_models['ANN'] = ann_model
#print(results_df)



# XGBoost
results_df, xgb_params, trained_models['XGBoost'] = xgboost_grid_search(X_train, y_train, X_test, y_test, results_df)



# Display Results
print(results_df)
display(format_results(results_df))
#plot_roc_auc(trained_models, X_test_scaled, y_test)




plt.plot([1, 2, 3], [4, 5, 6])
plt.title('Test Plot')
plt.show()



