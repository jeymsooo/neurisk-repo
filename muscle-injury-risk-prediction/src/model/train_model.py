from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import classification_report, precision_recall_fscore_support
from imblearn.over_sampling import SMOTE
import pandas as pd
import joblib
import os

MUSCLES = ['calves', 'hamstrings', 'quadriceps']
DEMOGRAPHIC_COLS = [
    'age', 'height', 'weight', 'bmi', 'training_frequency', 'previous_injury',
    'contraction_type', 'rms_time_corr', 'mnf_time_corr'
    # 'fatigue_level' removed, will add muscle-specific below
]

def load_data(file_path):
    return pd.read_csv(file_path)

def preprocess_data(data, muscle):
    muscle_fatigue_col = f'fatigue_level_{muscle}'
    muscle_feats = [f'{feat}_{muscle}' for feat in ['rms', 'mav', 'zc', 'ssc', 'wl', 'mdf', 'mnf']]
    # Use muscle-specific fatigue if present, else fallback to generic
    if muscle_fatigue_col in data.columns:
        fatigue_cols = [muscle_fatigue_col]
    elif 'fatigue_level' in data.columns:
        fatigue_cols = ['fatigue_level']
    else:
        raise KeyError(f"No fatigue column found for {muscle}")
    X = data[DEMOGRAPHIC_COLS + fatigue_cols + muscle_feats]
    X = pd.get_dummies(X, columns=['previous_injury', 'contraction_type'])
    y = data['injury_risk']
    return X, y

def train_and_evaluate_per_muscle(data_dir='.'):
    os.makedirs('models', exist_ok=True)
    for muscle in MUSCLES:
        file_path = os.path.join(data_dir, f'synthetic_emg_{muscle}.csv')
        data = load_data(file_path)
        X, y = preprocess_data(data, muscle)
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        smote = SMOTE(random_state=42)
        X_res, y_res = smote.fit_resample(X_train, y_train)
        # Regularized Gradient Boosting
        model = GradientBoostingClassifier(
            max_depth=4,
            learning_rate=0.08,
            subsample=0.8,
            min_samples_split=8,
            n_estimators=120,
            random_state=42
        )
        # Cross-validation on train set
        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_scores = cross_val_score(model, X_res, y_res, cv=cv, scoring='f1_weighted')
        print(f"{muscle.capitalize()} CV F1 (train, weighted): {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        # Fit and evaluate on test set
        model.fit(X_res, y_res)
        y_pred = model.predict(X_test)
        print(f"\n{muscle.capitalize()} Test Set Classification Report:")
        print(classification_report(y_test, y_pred, digits=3))
        # Save model
        joblib.dump(model, os.path.join('models', f'model_{muscle}.pkl'))

if __name__ == "__main__":
    train_and_evaluate_per_muscle(data_dir='.')