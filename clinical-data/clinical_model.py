import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import RobustScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, precision_score, recall_score, cohen_kappa_score
from sklearn.feature_selection import SelectKBest, f_classif, VarianceThreshold
from imblearn.over_sampling import SMOTE
import optuna
import seaborn as sns
import matplotlib.pyplot as plt
from collections import Counter
from torch.utils.data import DataLoader, TensorDataset
import os
from google.colab import drive

# Mount Google Drive
drive.mount('/content/drive')

# Define the model
class AdvancedMLP(nn.Module):
    def __init__(self, input_size, hidden_sizes, num_classes, dropout_rate=0.3):
        super(AdvancedMLP, self).__init__()
        layers = []
        prev_size = input_size
        for hidden_size in hidden_sizes:
            layers.extend([
                nn.Linear(prev_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.ReLU(),
                nn.Dropout(dropout_rate)
            ])
            prev_size = hidden_size
        layers.append(nn.Linear(prev_size, num_classes))
        self.network = nn.Sequential(*layers)

    def forward(self, x):
        return self.network(x)

# Focal Loss
class FocalLoss(nn.Module):
    def __init__(self, gamma=2.0, alpha=None):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.alpha = alpha if alpha is not None else torch.FloatTensor([1.7, 2.0, 1.0, 1.0, 1.0])

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = ((1 - pt) ** self.gamma) * ce_loss
        return focal_loss.mean()

# Function to align columns
def align_columns(df_small, df_large, target_col):
    large_cols = [col for col in df_large.columns if col != target_col]
    small_cols = [col for col in df_small.columns if col != target_col]
    missing_cols = [col for col in large_cols if col not in small_cols]

    missing_data = {}
    for col in missing_cols:
        if df_large[col].dtype in ['float64', 'int64']:
            missing_data[col] = [df_large[col].median()] * len(df_small)
        else:
            missing_data[col] = [df_large[col].mode()[0]] * len(df_small)

    missing_df = pd.DataFrame(missing_data, index=df_small.index)
    df_small = pd.concat([df_small, missing_df], axis=1)
    df_small = df_small[[col for col in df_large.columns if col != target_col] + [target_col]]
    return df_small.copy()

# Function to add synthetic class 4 samples
def add_synthetic_class_4(df_small, df_large, target_col, n_samples=100):
    class_4_samples = df_large[df_large[target_col] == 4].sample(n_samples, replace=True, random_state=42)
    return pd.concat([df_small, class_4_samples], ignore_index=True)

# Function to preprocess clinical data
def preprocess_clinical_data(df, target_col, preprocessor=None, apply_smote=False, selector=None, expected_features=None, dataset_name=""):
    for col in df.columns:
        if col != target_col and (df[col].dtype == 'object' or df[col].dtype.name == 'category'):
            df[col] = df[col].astype(str)

    df[target_col] = pd.to_numeric(df[target_col], errors='coerce').fillna(-1).astype(int)

    X = df.drop(columns=[target_col])
    y = df[target_col].values

    if preprocessor is None:
        numeric_cols = X.select_dtypes(include=['float64', 'int64']).columns
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='mean')),
            ('scaler', RobustScaler())
        ])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(
            transformers=[
                ('num', numeric_transformer, numeric_cols),
                ('cat', categorical_transformer, categorical_cols)
            ])
        X_processed = preprocessor.fit_transform(X)
    else:
        X_processed = preprocessor.transform(X)

    variance_threshold = VarianceThreshold(threshold=0.0)
    X_processed = variance_threshold.fit_transform(X_processed)

    if selector is None or expected_features is not None:
        selector = SelectKBest(f_classif, k=min(20, X_processed.shape[1]))
        X_processed = selector.fit_transform(X_processed, y)
    else:
        current_features = X_processed.shape[1]
        if current_features != expected_features:
            if current_features < expected_features:
                padding = np.zeros((X_processed.shape[0], expected_features - current_features))
                X_processed = np.hstack((X_processed, padding))
            elif current_features > expected_features:
                X_processed = X_processed[:, :expected_features]
        X_processed = selector.transform(X_processed)

    print(f"Number of features for {dataset_name}: {X_processed.shape[1]}")

    if apply_smote and len(np.unique(y)) > 1 and min(Counter(y).values()) > 1:
        smote = SMOTE(random_state=42, k_neighbors=1)
        X_processed, y = smote.fit_resample(X_processed, y)
        print(f"Class distribution after SMOTE for {dataset_name}: {Counter(y)}")

    X_train, X_test, y_train, y_test = train_test_split(X_processed, y, test_size=0.2, random_state=42)
    X_train = torch.FloatTensor(X_train)
    X_test = torch.FloatTensor(X_test)
    y_train = torch.LongTensor(y_train)
    y_test = torch.LongTensor(y_test)

    # Save test set
    np.save(f'/content/drive/MyDrive/X_test_{dataset_name}.npy', X_test.numpy())
    np.save(f'/content/drive/MyDrive/y_test_{dataset_name}.npy', y_test.numpy())
    print(f"Test set for {dataset_name} saved to /content/drive/MyDrive/X_test_{dataset_name}.npy and /content/drive/MyDrive/y_test_{dataset_name}.npy")

    return X_train, X_test, y_train, y_test, X_processed.shape[1], 5, preprocessor, selector

# Training function
def train_model(model, X_train, y_train, X_val, y_val, num_epochs=10, learning_rate=0.001, patience=10, checkpoint_path='checkpoint.pth', load_checkpoint=False):
    dataset = TensorDataset(X_train, y_train)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)
    best_qwk = -float('inf')
    epochs_no_improve = 0
    best_model_state = None

    initial_alpha = torch.FloatTensor([1.7, 2.0, 1.0, 1.0, 1.0])
    focal_loss = FocalLoss(gamma=2.0, alpha=initial_alpha)

    start_epoch = 0
    if load_checkpoint and os.path.exists(checkpoint_path):
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        try:
            model.load_state_dict(checkpoint['model_state_dict'], strict=False)
            optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            start_epoch = checkpoint['epoch'] + 1
            best_qwk = checkpoint['best_qwk']
            epochs_no_improve = checkpoint['epochs_no_improve']
            focal_loss.alpha = checkpoint['focal_alpha']
            print(f"Resuming from epoch {start_epoch}")
        except RuntimeError as e:
            print(f"Checkpoint mismatch: {e}")
            start_epoch = 0

    for epoch in range(start_epoch, num_epochs):
        model.train()
        train_loss = 0.0
        train_correct = 0
        total = 0
        train_predictions = []
        train_true_labels = []

        for X_batch, y_batch in loader:
            outputs = model(X_batch)
            loss = focal_loss(outputs, y_batch)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            train_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += y_batch.size(0)
            train_correct += (predicted == y_batch).sum().item()
            train_predictions.extend(predicted.cpu().numpy())
            train_true_labels.extend(y_batch.cpu().numpy())

        # Calculate evaluation metrics on training set
        train_qwk = cohen_kappa_score(train_true_labels, train_predictions, labels=[0, 1, 2, 3, 4], weights='quadratic')
        train_accuracy = accuracy_score(train_true_labels, train_predictions)
        train_f1 = f1_score(train_true_labels, train_predictions, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
        train_sensitivity = recall_score(train_true_labels, train_predictions, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
        train_cm = confusion_matrix(train_true_labels, train_predictions, labels=[0, 1, 2, 3, 4])
        train_specificity = []
        for cls in range(5):
            tn = np.sum(train_cm) - np.sum(train_cm[cls, :]) - np.sum(train_cm[:, cls]) + train_cm[cls, cls]
            fp = np.sum(train_cm[:, cls]) - train_cm[cls, cls]
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
            train_specificity.append(specificity)
        train_precision = precision_score(train_true_labels, train_predictions, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

        # Print training metrics
        print(f"Epoch {epoch+1}/{num_epochs}:")
        print(f"  Train QWK: {train_qwk:.4f}")
        print(f"  Train Accuracy: {train_accuracy:.4f}")
        print(f"  Train F1-score: {[f'{f1:.4f}' for f1 in train_f1]}")
        print(f"  Train Sensitivity: {[f'{sens:.4f}' for sens in train_sensitivity]}")
        print(f"  Train Specificity: {[f'{spec:.4f}' for spec in train_specificity]}")
        print(f"  Train Precision: {[f'{prec:.4f}' for prec in train_precision]}")

        # Evaluate on validation set
        model.eval()
        with torch.no_grad():
            val_outputs = model(X_val)
            val_loss = focal_loss(val_outputs, y_val)
            _, val_predicted = torch.max(val_outputs, 1)
            val_qwk = cohen_kappa_score(y_val.numpy(), val_predicted.numpy(), labels=[0, 1, 2, 3, 4], weights='quadratic')
            val_accuracy = accuracy_score(y_val.numpy(), val_predicted.numpy())

            cm = confusion_matrix(y_val.numpy(), val_predicted.numpy(), labels=[0, 1, 2, 3, 4])
            recalls = [cm[i, i] / cm[i].sum() if cm[i].sum() > 0 else 0.01 for i in range(5)]
            class_weights = [1.0 / max(r, 0.01) for r in recalls]
            class_weights = torch.FloatTensor(class_weights) / sum(class_weights) * 5.0
            focal_loss.alpha = class_weights * initial_alpha
            print(f"  Val Accuracy: {val_accuracy:.4f}, Val QWK: {val_qwk:.4f}, Class Weights: {class_weights.tolist()}")

        scheduler.step(val_qwk)
        if val_qwk > best_qwk:
            best_qwk = val_qwk
            best_model_state = model.state_dict()
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        if (epoch + 1) % 10 == 0 or epoch == num_epochs - 1:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'best_qwk': best_qwk,
                'epochs_no_improve': epochs_no_improve,
                'focal_alpha': focal_loss.alpha
            }, checkpoint_path)
            print(f"Checkpoint saved at epoch {epoch+1}")

        if epochs_no_improve >= patience:
            print("Early stopping triggered")
            break

    model.load_state_dict(best_model_state)
    return model, focal_loss, best_qwk

# Fine-tuning function
def fine_tune_model(model, X_fine, y_fine, hidden_sizes, input_size, num_epochs=20, learning_rate=0.00005, n_splits=3, checkpoint_path='fine_tune_checkpoint.pth'):
    dataset = TensorDataset(X_fine, y_fine)
    loader = DataLoader(dataset, batch_size=32, shuffle=True)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
    fine_tuned_models = []
    for fold, (train_idx, val_idx) in enumerate(kf.split(X_fine)):
        print(f"\nFine-tuning fold {fold+1}/{n_splits}...")
        X_train_fold = X_fine[train_idx]
        y_train_fold = y_fine[train_idx]
        X_val_fold = X_fine[val_idx]
        y_val_fold = y_fine[val_idx]
        model_fold = AdvancedMLP(input_size, hidden_sizes, 5, dropout_rate=model.network[3].p)
        model_fold.load_state_dict(model.state_dict())
        model_fold, _, _ = train_model(
            model_fold, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            num_epochs=num_epochs, learning_rate=learning_rate, patience=10,
            checkpoint_path=f'{checkpoint_path}_fold{fold+1}', load_checkpoint=False
        )
        fine_tuned_models.append(model_fold)
    return fine_tuned_models

# Function to fine-tune on multiple small datasets
def fine_tune_multiple_small_datasets(model, small_data_paths, target_col, hidden_sizes, input_size, preprocessor, selector):
    fine_tuned_models = []
    df_large = pd.read_csv('/content/drive/MyDrive/TrainingWiDS2021_filled.csv')
    expected_features = input_size
    for i, small_data_path in enumerate(small_data_paths):
        dataset_name = f"small_dataset_{i+1}"
        print(f"\nProcessing small dataset {i+1}: {small_data_path}")
        df_small = pd.read_csv(small_data_path)
        df_small = align_columns(df_small, df_large, target_col)
        df_small = add_synthetic_class_4(df_small, df_large, target_col, n_samples=100)
        X_fine, X_test_fine, y_fine, y_test_fine, fine_input_size, _, _, _ = preprocess_clinical_data(
            df_small, target_col, preprocessor=preprocessor, selector=None, apply_smote=True,
            expected_features=expected_features, dataset_name=dataset_name
        )
        print(f"Fine-tuning on small dataset {i+1} with input size {fine_input_size}...")
        models = fine_tune_model(
            model, X_fine, y_fine, hidden_sizes, input_size,
            num_epochs=20, learning_rate=0.00005, n_splits=3,
            checkpoint_path=f'fine_tune_checkpoint_small{i+1}'
        )
        fine_tuned_models.extend(models)
        for j, model_fold in enumerate(models):
            torch.save({
                'model_state_dict': model_fold.state_dict(),
                'input_size': input_size,
                'hidden_sizes': hidden_sizes,
                'num_classes': 5,
                'dropout_rate': model_fold.network[3].p,
                'focal_gamma': 2.0,
                'focal_alpha': None
            }, f'/content/drive/MyDrive/fine_tuned_model_small{i+1}_fold{j+1}.pth')
    return fine_tuned_models

# Function to calculate Quadratic Weighted Kappa
def quadratic_weighted_kappa(y_true, y_pred, num_classes=5):
    conf_mat = np.zeros((num_classes, num_classes))
    for t, p in zip(y_true, y_pred):
        conf_mat[t, p] += 1
    w = np.zeros((num_classes, num_classes))
    for i in range(num_classes):
        for j in range(num_classes):
            w[i, j] = ((i - j) ** 2) / ((num_classes - 1) ** 2)
    act_hist = np.histogram(y_true, bins=num_classes, range=(0, num_classes))[0]
    pred_hist = np.histogram(y_pred, bins=num_classes, range=(0, num_classes))[0]
    E = np.outer(act_hist, pred_hist) / np.sum(act_hist)
    conf_mat = conf_mat / conf_mat.sum()
    num = np.sum(w * conf_mat)
    den = np.sum(w * E)
    return 1 - num / den if den != 0 else 0

# Evaluation function with ensemble weights
def evaluate_model(models, X_test, y_test, accuracy_threshold=0.85, dataset_name="", model_set_name=""):
    model_weights = []
    predictions = []
    for model in models:
        model.eval()
        with torch.no_grad():
            outputs = model(X_test)
            _, predicted = torch.max(outputs, 1)
            acc = accuracy_score(y_test.numpy(), predicted.numpy())
            cm = confusion_matrix(y_test.numpy(), predicted.numpy(), labels=[0, 1, 2, 3, 4])
            weight = acc * (0.3 * cm[0, 0] / cm[0].sum() + 0.7 * cm[1, 1] / cm[1].sum() if cm[0].sum() > 0 and cm[1].sum() > 0 else 0.01)
            model_weights.append(weight)
            predictions.append(predicted.numpy())

    model_weights = np.array(model_weights) / np.sum(model_weights)
    predictions = np.array(predictions)
    ensemble_preds = np.apply_along_axis(
        lambda x: np.bincount(x, weights=model_weights).argmax(),
        axis=0,
        arr=predictions
    )

    # Calculate evaluation metrics on test set
    y_true_test = y_test.numpy()
    y_pred_classes_test = ensemble_preds

    # Quadratic Weighted Kappa (QWK)
    qwk_test = cohen_kappa_score(y_true_test, y_pred_classes_test, labels=[0, 1, 2, 3, 4], weights='quadratic')

    # Overall accuracy
    accuracy = accuracy_score(y_true_test, y_pred_classes_test)

    # F1-score and Sensitivity (Recall) for each class
    f1_test = f1_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)
    sensitivity_test = recall_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

    # Specificity for each class
    cm = confusion_matrix(y_true_test, y_pred_classes_test, labels=[0, 1, 2, 3, 4])
    specificity_test = []
    for cls in range(5):
        tn = np.sum(cm) - np.sum(cm[cls, :]) - np.sum(cm[:, cls]) + cm[cls, cls]
        fp = np.sum(cm[:, cls]) - cm[cls, cls]
        specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
        specificity_test.append(specificity)

    # Precision for each class
    precision_test = precision_score(y_true_test, y_pred_classes_test, average=None, labels=[0, 1, 2, 3, 4], zero_division=0)

    # Create dictionary for metrics logging
    log_metrics = {
        'Dataset': dataset_name,
        'Model_Set': model_set_name,
        'QWK': qwk_test,
        'Accuracy': accuracy,
        'F1_Class_0': f1_test[0],
        'F1_Class_1': f1_test[1],
        'F1_Class_2': f1_test[2],
        'F1_Class_3': f1_test[3],
        'F1_Class_4': f1_test[4],
        'Sensitivity_Class_0': sensitivity_test[0],
        'Sensitivity_Class_1': sensitivity_test[1],
        'Sensitivity_Class_2': sensitivity_test[2],
        'Sensitivity_Class_3': sensitivity_test[3],
        'Sensitivity_Class_4': sensitivity_test[4],
        'Specificity_Class_0': specificity_test[0],
        'Specificity_Class_1': specificity_test[1],
        'Specificity_Class_2': specificity_test[2],
        'Specificity_Class_3': specificity_test[3],
        'Specificity_Class_4': specificity_test[4],
        'Precision_Class_0': precision_test[0],
        'Precision_Class_1': precision_test[1],
        'Precision_Class_2': precision_test[2],
        'Precision_Class_3': precision_test[3],
        'Precision_Class_4': precision_test[4]
    }

    # Print metrics
    print(f"\nMetrics for {dataset_name} evaluated on {model_set_name}:")
    print(f"QWK on test set: {qwk_test:.4f}")
    print(f"Accuracy on test set: {accuracy:.4f}")
    print(f"F1-score per class on test set: {[f'{f1:.4f}' for f1 in f1_test]}")
    print(f"Sensitivity per class on test set: {[f'{sens:.4f}' for sens in sensitivity_test]}")
    print(f"Specificity per class on test set: {[f'{spec:.4f}' for spec in specificity_test]}")
    print(f"Precision per class on test set: {[f'{prec:.4f}' for prec in precision_test]}")

    # Plot confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0, 1, 2, 3, 4], yticklabels=[0, 1, 2, 3, 4])
    plt.title(f'Confusion Matrix (Test) - {dataset_name} - {model_set_name}')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.savefig(f'/content/drive/MyDrive/confusion_matrix_test_{dataset_name}_{model_set_name}.png')
    plt.close()

    if accuracy < accuracy_threshold:
        print(f"\nPer-class Metrics (Test) for {dataset_name} on {model_set_name}:")
        for i in range(5):
            print(f"Class {i}: Precision={precision_test[i]:.4f}, Recall={sensitivity_test[i]:.4f}, F1={f1_test[i]:.4f}")

        # Plot F1-score per class
        plt.figure(figsize=(8, 6))
        plt.bar(range(5), f1_test, tick_label=[0, 1, 2, 3, 4])
        plt.title(f'F1-Score per Class (Test) - {dataset_name} - {model_set_name}')
        plt.xlabel('Class')
        plt.ylabel('F1-Score')
        plt.savefig(f'/content/drive/MyDrive/f1_score_per_class_test_{dataset_name}_{model_set_name}.png')
        plt.close()

    return ensemble_preds, log_metrics

def log_metrics_to_file(metrics, log_file_path='/content/drive/MyDrive/evaluation_metrics.csv'):
    import pandas as pd
    import os
    # Convert metrics dictionary to DataFrame
    metrics_df = pd.DataFrame([metrics])
    # If file exists, read and append; otherwise, create new file
    if os.path.exists(log_file_path):
        existing_df = pd.read_csv(log_file_path)
        metrics_df = pd.concat([existing_df, metrics_df], ignore_index=True)
    # Save to CSV file
    metrics_df.to_csv(log_file_path, index=False)
    print(f"Metrics for {metrics['Dataset']} on {metrics['Model_Set']} logged to {log_file_path}")

# Hyperparameter optimization function
def objective(trial):
    learning_rate = trial.suggest_float('learning_rate', 1e-5, 1e-2, log=True)
    hidden_size1 = trial.suggest_int('hidden_size1', 64, 256)
    hidden_size2 = trial.suggest_int('hidden_size2', 32, 128)
    dropout_rate = trial.suggest_float('dropout_rate', 0.2, 0.5)
    model = AdvancedMLP(input_size_global, [hidden_size1, hidden_size2], 5, dropout_rate)
    X_train_split, X_val_split, y_train_split, y_val_split = train_test_split(
        X_train_global.numpy(), y_train_global.numpy(), test_size=0.2, random_state=42
    )
    X_train_split = torch.FloatTensor(X_train_split)
    X_val_split = torch.FloatTensor(X_val_split)
    y_train_split = torch.LongTensor(y_train_split)
    y_val_split = torch.LongTensor(y_val_split)
    model, _, _ = train_model(
        model, X_train_split, y_train_split, X_val_split, y_val_split,
        num_epochs=10, learning_rate=learning_rate, patience=5,
        checkpoint_path=f'optuna_checkpoint_trial{trial.number}.pth',
        load_checkpoint=False
    )
    model.eval()
    with torch.no_grad():
        outputs = model(X_val_split)
        _, predicted = torch.max(outputs, 1)
        qwk = cohen_kappa_score(y_val_split.numpy(), predicted.numpy(), labels=[0, 1, 2, 3, 4], weights='quadratic')
    return qwk

# Main function
def main():
    global X_train_global, y_train_global, input_size_global, preprocessor, selector
    large_data_path = '/content/drive/MyDrive/TrainingWiDS2021_filled.csv'
    small_data_paths = ['/content/drive/MyDrive/diabetes_filled.csv', '/content/drive/MyDrive/diabetes_prediction_dataset_filled.csv']
    target_col = 'severity_label'
    log_file_path = '/content/drive/MyDrive/evaluation_metrics.csv'

    # Process large dataset
    df_large = pd.read_csv(large_data_path)
    X_train, X_test, y_train, y_test, input_size, num_classes, preprocessor, selector = preprocess_clinical_data(
        df_large, target_col, apply_smote=True, dataset_name="large_dataset"
    )
    X_train_global = X_train
    y_train_global = y_train
    input_size_global = input_size

    print("Optimizing hyperparameters...")
    study = optuna.create_study(direction='maximize')
    study.optimize(objective, n_trials=10)
    best_params = study.best_params
    print(f"Best hyperparameters: {best_params}")

    kf = KFold(n_splits=3, shuffle=True, random_state=42)
    main_models = []
    best_qwk_overall = -float('inf')
    best_model_state_overall = None

    for fold, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        print(f"\nTraining fold {fold+1}/3 on large dataset...")
        X_train_fold = X_train[train_idx]
        y_train_fold = y_train[train_idx]
        X_val_fold = X_train[val_idx]
        y_val_fold = y_train[val_idx]
        model_main = AdvancedMLP(
            input_size=input_size,
            hidden_sizes=[best_params['hidden_size1'], best_params['hidden_size2']],
            num_classes=5,
            dropout_rate=best_params['dropout_rate']
        )
        model_main, _, fold_best_qwk = train_model(
            model_main, X_train_fold, y_train_fold, X_val_fold, y_val_fold,
            num_epochs=10, learning_rate=best_params['learning_rate'], patience=10,
            checkpoint_path=f'/content/drive/MyDrive/main_checkpoint_fold{fold+1}.pth'
        )
        main_models.append(model_main)
        torch.save({
            'model_state_dict': model_main.state_dict(),
            'input_size': input_size,
            'hidden_sizes': [best_params['hidden_size1'], best_params['hidden_size2']],
            'num_classes': 5,
            'dropout_rate': best_params['dropout_rate'],
            'focal_gamma': 2.0,
            'focal_alpha': torch.FloatTensor([1.7, 2.0, 1.0, 1.0, 1.0])
        }, f'/content/drive/MyDrive/main_model_fold{fold+1}.pth')

        if fold_best_qwk > best_qwk_overall:
            best_qwk_overall = fold_best_qwk
            best_model_state_overall = model_main.state_dict()

    if best_model_state_overall is not None:
        torch.save({
            'model_state_dict': best_model_state_overall,
            'input_size': input_size,
            'hidden_sizes': [best_params['hidden_size1'], best_params['hidden_size2']],
            'num_classes': 5,
            'dropout_rate': best_params['dropout_rate'],
            'focal_gamma': 2.0,
            'focal_alpha': torch.FloatTensor([1.7, 2.0, 1.0, 1.0, 1.0]),
            'best_qwk': best_qwk_overall
        }, '/content/drive/MyDrive/main_clinical_model.pth')
        print(f"Main clinical model saved to /content/drive/MyDrive/main_clinical_model.pth with QWK: {best_qwk_overall:.4f}")

    print("\nFine-tuning on multiple small datasets...")
    fine_tuned_models = fine_tune_multiple_small_datasets(
        main_models[0], small_data_paths, target_col,
        hidden_sizes=[best_params['hidden_size1'], best_params['hidden_size2']],
        input_size=input_size,
        preprocessor=preprocessor,
        selector=selector
    )

    print("\nEvaluating ensemble on each test set...")
    all_models = main_models + fine_tuned_models

    # Evaluate test set of large dataset
    # 1. On main_models
    _, metrics_large_main = evaluate_model(main_models, X_test, y_test, dataset_name="large_dataset", model_set_name="main_models")
    log_metrics_to_file(metrics_large_main, log_file_path)
    # 2. On fine_tuned_models
    _, metrics_large_fine = evaluate_model(fine_tuned_models, X_test, y_test, dataset_name="large_dataset", model_set_name="fine_tuned_models")
    log_metrics_to_file(metrics_large_fine, log_file_path)
    # 3. On all_models
    _, metrics_large_all = evaluate_model(all_models, X_test, y_test, dataset_name="large_dataset", model_set_name="all_models")
    log_metrics_to_file(metrics_large_all, log_file_path)

    # Evaluate test sets of small datasets
    for i in range(len(small_data_paths)):
        dataset_name = f"small_dataset_{i+1}"
        X_test_small = torch.FloatTensor(np.load(f'/content/drive/MyDrive/X_test_{dataset_name}.npy'))
        y_test_small = torch.LongTensor(np.load(f'/content/drive/MyDrive/y_test_{dataset_name}.npy'))
        # 1. On main_models
        _, metrics_small_main = evaluate_model(main_models, X_test_small, y_test_small, dataset_name=dataset_name, model_set_name="main_models")
        log_metrics_to_file(metrics_small_main, log_file_path)
        # 2. On fine_tuned_models
        _, metrics_small_fine = evaluate_model(fine_tuned_models, X_test_small, y_test_small, dataset_name=dataset_name, model_set_name="fine_tuned_models")
        log_metrics_to_file(metrics_small_fine, log_file_path)
        # 3. On all_models
        _, metrics_small_all = evaluate_model(all_models, X_test_small, y_test_small, dataset_name=dataset_name, model_set_name="all_models")
        log_metrics_to_file(metrics_small_all, log_file_path)

if __name__ == "__main__":
    main()
