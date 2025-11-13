from __future__ import annotations

"""
Machine Learning module for Venture-Scope.

Predicts startup success (acquisition/IPO) vs failure (closed) 
based on KPIs and features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import pickle

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, 
    precision_score, 
    recall_score, 
    f1_score,
    confusion_matrix,
    classification_report,
    roc_auc_score,
    roc_curve
)

import matplotlib.pyplot as plt
import seaborn as sns

print("=== ML MODULE LOADED ===")


# ==================== DATA PREPARATION ====================

def prepare_ml_dataset(
    df: pd.DataFrame,
    include_operating: bool = False,
    verbose: bool = True
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Prepare dataset for ML by creating labels and filtering.
    
    Labels:
    - 1 = Success (acquired or ipo)
    - 0 = Failure (closed)
    
    Args:
        df: Input dataframe with 'status' column
        include_operating: If True, include 'operating' as unknown (not recommended)
        verbose: Print progress
    
    Returns:
        (X, y) tuple with features and labels
    """
    if verbose:
        print("\nPreparing ML Dataset...")
        print(f"  Original size: {len(df):,} startups")
    
    # Create labels
    df = df.copy()
    
    # Success = acquired or ipo
    df['success'] = df['status'].apply(lambda x: 
        1 if x in ['acquired', 'ipo'] else (0 if x == 'closed' else -1)
    )
    
    if verbose:
        print(f"\n  Label distribution:")
        print(f"    Success (acquired/ipo): {(df['success'] == 1).sum():,}")
        print(f"    Failure (closed):       {(df['success'] == 0).sum():,}")
        print(f"    Unknown (operating):    {(df['success'] == -1).sum():,}")
    
    # Filter to known outcomes only (unless include_operating=True)
    if not include_operating:
        df = df[df['success'] != -1].copy()
        if verbose:
            print(f"\n  Filtered to known outcomes: {len(df):,} startups")
            success_rate = (df['success'] == 1).sum() / len(df) * 100
            print(f"  Success rate: {success_rate:.1f}%")
    
    return df


def engineer_features(
    df: pd.DataFrame,
    verbose: bool = True
) -> pd.DataFrame:
    """
    Feature engineering: encode categoricals, create interactions.
    
    Args:
        df: Input dataframe
        verbose: Print progress
    
    Returns:
        Dataframe with engineered features
    """
    if verbose:
        print("\nEngineering Features...")
    
    result = df.copy()
    
    # 1. One-Hot Encode categorical features
    categorical_cols = ['stage', 'sector', 'country']
    
    for col in categorical_cols:
        if col in result.columns:
            if verbose:
                print(f"  Encoding {col}...")
            # Get dummies (one-hot encoding)
            dummies = pd.get_dummies(result[col], prefix=col, drop_first=True)
            result = pd.concat([result, dummies], axis=1)
    
    if verbose:
        print(f"  Total features after encoding: {len(result.columns)}")
    
    return result


def select_features(df: pd.DataFrame) -> list:
    """
    Select features for ML model.
    
    Returns:
        List of feature column names
    """
    # Numeric KPIs
    numeric_features = [
        'rule_of_40',
        'traction_index',
        'capital_efficiency',
        'burn_multiple',
        'runway_months',
        'funding_amount',
        'investors_count',
        'investment_score'
    ]
    
    # Categorical (one-hot encoded) - all columns starting with stage_, sector_, country_
    categorical_features = [
        col for col in df.columns 
        if col.startswith(('stage_', 'sector_', 'country_'))
    ]
    
    # Combine
    all_features = numeric_features + categorical_features
    
    # Filter to existing columns
    features = [f for f in all_features if f in df.columns]
    
    return features


# ==================== MODEL TRAINING ====================

def train_model(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_estimators: int = 100,
    random_state: int = 42,
    verbose: bool = True
) -> RandomForestClassifier:
    """
    Train Random Forest classifier.
    
    Args:
        X_train: Training features
        y_train: Training labels
        n_estimators: Number of trees
        random_state: Random seed
        verbose: Print progress
    
    Returns:
        Trained model
    """
    if verbose:
        print(f"\nTraining Random Forest ({n_estimators} trees)...")
    
    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=10,
        min_samples_split=20,
        min_samples_leaf=10,
        random_state=random_state,
        n_jobs=-1  # Use all CPU cores
    )
    
    model.fit(X_train, y_train)
    
    if verbose:
        print(f"  Model trained on {len(X_train):,} samples")
    
    return model


# ==================== EVALUATION ====================

def evaluate_model(
    model: RandomForestClassifier,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    verbose: bool = True
) -> Dict:
    """
    Evaluate model performance.
    
    Returns:
        Dictionary with metrics
    """
    if verbose:
        print(f"\nEvaluating Model on {len(X_test):,} test samples...")
    
    # Predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of success
    
    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'f1': f1_score(y_test, y_pred),
        'roc_auc': roc_auc_score(y_test, y_pred_proba)
    }
    
    if verbose:
        print(f"\n{'='*70}")
        print("MODEL PERFORMANCE")
        print(f"{'='*70}")
        print(f"  Accuracy:  {metrics['accuracy']:.2%}  (Overall correctness)")
        print(f"  Precision: {metrics['precision']:.2%}  (Of predicted successes, % truly successful)")
        print(f"  Recall:    {metrics['recall']:.2%}  (Of true successes, % detected)")
        print(f"  F1-Score:  {metrics['f1']:.2%}  (Harmonic mean of precision & recall)")
        print(f"  ROC-AUC:   {metrics['roc_auc']:.2%}  (Area under ROC curve)")
        print(f"{'='*70}")
        
        # Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        print(f"              Failure  Success")
        print(f"  Actual  Failure   {cm[0,0]:4d}    {cm[0,1]:4d}")
        print(f"          Success   {cm[1,0]:4d}    {cm[1,1]:4d}")
        
        # Classification Report
        print(f"\nDetailed Classification Report:")
        print(classification_report(y_test, y_pred, target_names=['Failure', 'Success']))
    
    return metrics


def get_feature_importance(
    model: RandomForestClassifier,
    feature_names: list,
    top_n: int = 15
) -> pd.DataFrame:
    """
    Get feature importance from trained model.
    
    Returns:
        DataFrame with features sorted by importance
    """
    importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': model.feature_importances_
    }).sort_values('importance', ascending=False)
    
    return importance_df.head(top_n)


# ==================== MAIN PIPELINE ====================

def run_ml_pipeline(
    input_file: str = "data/processed/startups_scored.csv",
    test_size: float = 0.2,
    random_state: int = 42,
    save_model: bool = True
) -> Dict:
    """
    Complete ML pipeline from data loading to evaluation.
    
    Returns:
        Dictionary with model, metrics, and results
    """
    print("\n" + "="*70)
    print("VENTURE-SCOPE: MACHINE LEARNING PIPELINE")
    print("="*70)
    
    # 1. Load data
    print(f"\nStep 1: Loading data from {input_file}")
    df = pd.read_csv(input_file)
    print(f"  Loaded {len(df):,} startups")
    
    # 2. Prepare dataset (filter to acquired/ipo/closed only)
    df_ml = prepare_ml_dataset(df, include_operating=False, verbose=True)
    
    # 3. Feature engineering
    df_ml = engineer_features(df_ml, verbose=True)
    
    # 4. Select features
    features = select_features(df_ml)
    print(f"\nStep 2: Selected {len(features)} features for ML")
    
    # 5. Prepare X and y
    X = df_ml[features].fillna(0)  # Fill any remaining NaN with 0
    y = df_ml['success']
    
    print(f"\nFinal dataset shape:")
    print(f"  X: {X.shape}")
    print(f"  y: {y.shape}")
    print(f"  Success rate: {y.mean():.1%}")
    
    # 6. Train/Test split
    print(f"\nStep 3: Train/Test split ({int((1-test_size)*100)}%/{int(test_size*100)}%)")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, 
        test_size=test_size, 
        random_state=random_state,
        stratify=y  # Keep same success rate in train & test
    )
    print(f"  Training set: {len(X_train):,} samples")
    print(f"  Test set:     {len(X_test):,} samples")
    
    # 7. Train model
    print(f"\nStep 4: Training model...")
    model = train_model(X_train, y_train, verbose=True)
    
    # 8. Evaluate
    print(f"\nStep 5: Evaluating model...")
    metrics = evaluate_model(model, X_test, y_test, verbose=True)
    
    # 9. Feature importance
    print(f"\nStep 6: Analyzing feature importance...")
    importance = get_feature_importance(model, features, top_n=15)
    print("\nTop 15 Most Important Features:")
    print(importance.to_string(index=False))
    
    # 10. Save model
    if save_model:
        model_path = Path("models/random_forest.pkl")
        model_path.parent.mkdir(parents=True, exist_ok=True)
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"\nModel saved to: {model_path}")
    
    print("\n" + "="*70)
    print("PIPELINE COMPLETED SUCCESSFULLY")
    print("="*70)
    
    return {
        'model': model,
        'metrics': metrics,
        'feature_importance': importance,
        'X_test': X_test,
        'y_test': y_test,
        'features': features
    }


# ==================== TESTING ====================

if __name__ == "__main__":
    results = run_ml_pipeline()