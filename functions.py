import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import StandardScaler
from sklearn.feature_selection import RFECV
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, confusion_matrix, f1_score, recall_score, precision_score, roc_curve, auc, precision_recall_curve
import numpy as np
import matplotlib.pyplot as plt
import numpy as np

def load_dataset(file_path, target='task1'):
    df = pd.read_csv(file_path)
    if target == 'task1':
        df.drop(['target'], axis=1, inplace=True)
        df.rename(columns={'readmitted': 'target'}, inplace=True)
        return df
    else:
        raise ValueError("Invalid target specified. Use 'task1'.")
    

def load_subset(target):
    if target == 'task1':
        # Load subset for task 1
        return pd.read_csv('subset_task1.csv')
    else:
        raise ValueError("Invalid target specified. Use 'task1'.")
    



def plot_pca(df):
    X = df.drop(['target'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA()
    pca.fit(X_scaled)
    explained = np.cumsum(pca.explained_variance_ratio_)

    plt.figure(figsize=(8, 5))
    plt.plot(explained, linestyle='-', color='blue')
    plt.xlabel('Number of components')
    plt.ylabel('Cumulative explained variance')
    plt.grid(True)
    plt.axhline(y=0.99, color='black', linestyle='--', label='99% Variance')
    plt.axhline(y=0.95, color='r', linestyle='--', label='95% Variance')
    plt.axhline(y=0.90, color='b', linestyle='--', label='90% Variance')
    plt.axhline(y=0.85, color='g', linestyle='--', label='85% Variance')
    plt.axhline(y=0.80, color='y', linestyle='--', label='80% Variance')
    plt.legend()
    plt.title('Choose the number of PCA components')
    plt.show()

    thresholds = [0.80, 0.85, 0.90, 0.95, 0.99]

    for t in thresholds:
        n_components = np.argmax(explained >= t) + 1
        print(f"{int(t*100)}% variance: {n_components} components")



def pca_df(df, n_components):
    X = df.drop(['target'], axis=1)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    pca = PCA(n_components=n_components)
    X_pca = pca.fit_transform(X_scaled)
    pca_df = pd.DataFrame(data=X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    pca_df['target'] = df['target'].values
    return pca_df


def feature_selection(X, y, rf_model):
    
    selector = RFECV(estimator=rf_model, step=1, cv=StratifiedKFold(3), scoring='accuracy', verbose=1)
    selector.fit(X, y)
    print("Optimal number of features:", selector.n_features_)
    selected_features = X.columns[selector.support_]
    print("Selected features:", selected_features.tolist())
    X_selected = X[selected_features]
    
    return X_selected, y



##########################

def fine_tune_and_cross_validate(X, y, model, param_grid, cv_splits=5):
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc
    import numpy as np
    import matplotlib.pyplot as plt

    # Fine tuning con GridSearchCV
    grid_search = GridSearchCV(model, param_grid, cv=cv_splits, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X, y)
    print("Best hyperparameters:", grid_search.best_params_)

    # Cross validation con best hyperparameters
    best_model = grid_search.best_estimator_
    kfold = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)

    # metriche base
    accuracy_list, recall_list, precision_list, f1_list = [], [], [], []

    # ROC curve
    mean_fpr = np.linspace(0, 1, 100)
    tpr_list, roc_auc_list = [], []

    print("Cross-validation results for each fold:\n")
    for i, (train_idx, test_idx) in enumerate(kfold.split(X, y), 1):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
        best_model.fit(X_train, y_train)
        y_pred = best_model.predict(X_test)
        y_prob = best_model.predict_proba(X_test)[:, 1]

        # metriche scalari
        accuracy = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        precision = precision_score(y_test, y_pred, average='weighted')
        print(f"Fold {i} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")

        accuracy_list.append(accuracy)
        f1_list.append(f1)
        recall_list.append(recall)
        precision_list.append(precision)

        # ROC
        fpr, tpr, _ = roc_curve(y_test, y_prob)
        tpr_interp = np.interp(mean_fpr, fpr, tpr)
        tpr_list.append(tpr_interp)
        roc_auc_list.append(auc(fpr, tpr))

    # statistiche globali (mean, std)
    accuracy_mean, f1_mean = np.mean(accuracy_list), np.mean(f1_list)
    recall_mean, precision_mean = np.mean(recall_list), np.mean(precision_list)
    accuracy_std, f1_std = np.std(accuracy_list), np.std(f1_list)
    recall_std, precision_std = np.std(recall_list), np.std(precision_list)
    roc_mean = np.mean(roc_auc_list)
    roc_std = np.std(roc_auc_list)

    print("\nOverall Cross-validation results:\n")
    print(f"\t- Accuracy: {accuracy_mean:.4f} ± {accuracy_std:.4f}")
    print(f"\t- F1 Score: {f1_mean:.4f} ± {f1_std:.4f}")
    print(f"\t- Recall: {recall_mean:.4f} ± {recall_std:.4f}")
    print(f"\t- Precision: {precision_mean:.4f} ± {precision_std:.4f}")
    print(f"\t- ROC AUC: {roc_mean:.4f} ± {roc_std:.4f}")

    # Plot ROC
    mean_tpr = np.mean(tpr_list, axis=0)
    std_tpr = np.std(tpr_list, axis=0)
    plt.figure(figsize=(6,5))
    plt.plot(mean_fpr, mean_tpr, color='b', label=f'Mean ROC (score={roc_mean:.2f})')
    plt.fill_between(mean_fpr, np.clip(mean_tpr-std_tpr,0,1), np.clip(mean_tpr+std_tpr,0,1), color='b', alpha=0.2)
    plt.plot([0,1],[0,1],'k--')
    plt.title("ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.show()

    return (accuracy_mean, f1_mean, recall_mean, precision_mean,
            accuracy_std, f1_std, recall_std, precision_std,
            best_model, grid_search.best_params_)


def test_model(model, X_train, y_train, X_test, y_test, model_name="Model"):
    from sklearn.metrics import accuracy_score, f1_score, recall_score, precision_score, roc_curve, auc, confusion_matrix, ConfusionMatrixDisplay
    import matplotlib.pyplot as plt

    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    # metriche scalari
    accuracy = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    precision = precision_score(y_test, y_pred, average='weighted')

    # ROC curve + AUC
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    print(f"{model_name} - Accuracy: {accuracy:.4f}, F1: {f1:.4f}, Recall: {recall:.4f}, Precision: {precision:.4f}")
    print(f"{model_name} - ROC AUC: {roc_auc:.4f}")

    # Plot ROC curve
    plt.figure(figsize=(6,5))
    plt.plot(fpr, tpr, color='b', label=f'ROC curve (AUC={roc_auc:.2f})')
    plt.plot([0,1], [0,1], 'k--')
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title(f"ROC Curve - {model_name}")
    plt.legend()
    plt.show()

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(cmap='Blues')
    plt.title(f"Confusion Matrix - {model_name}")
    plt.show()

    return {
        "accuracy": accuracy,
        "f1": f1,
        "recall": recall,
        "precision": precision,
        "roc_auc": roc_auc
    }
