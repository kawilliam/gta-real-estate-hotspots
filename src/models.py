"""
Model Implementation Module

Author: Utsav Patel (Modeler)
Date: 2024-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import Dict, Any, Tuple
import pickle

from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.model_selection import TimeSeriesSplit
import xgboost as xgb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModels:
    def __init__(self, features_df: pd.DataFrame, target_col: str = 'price_mean'):
        self.features_df = features_df
        self.target_col = target_col
        self.models = {}
        self.results = {}
    
    def prepare_features(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare features and target for modeling"""
        logger.info("Preparing features for modeling...")
        
        # Select numeric features only
        numeric_cols = self.features_df.select_dtypes(include=[np.number]).columns.tolist()
        
        # Remove target and irrelevant columns
        feature_cols = [col for col in numeric_cols 
                       if col != self.target_col 
                       and 'lat' not in col.lower() 
                       and 'lon' not in col.lower()]
        
        X = self.features_df[feature_cols].fillna(0)
        y = self.features_df[self.target_col]
        
        # Remove rows where target is missing
        valid_idx = y.notna()
        X = X[valid_idx]
        y = y[valid_idx]
        
        logger.info(f"Prepared {X.shape[1]} features for {X.shape[0]} samples")
        return X, y
    
    def naive_baseline(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """Naive baseline (persistence model)"""
        logger.info("Training naive baseline...")
        
        # For progress report, use simple train/test split
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Persistence model: predict last known value
        # For now, use mean of training set
        y_pred = np.full_like(y_test, y_train.mean())
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.results['naive'] = metrics
        logger.info(f"Naive baseline: RMSE = {metrics['rmse']:.2f}")
        
        return metrics
    
    def lasso_regression(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """LASSO regression with feature selection"""
        logger.info("Training LASSO regression...")
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Standardize features
        from sklearn.preprocessing import StandardScaler
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train LASSO
        lasso = Lasso(alpha=0.1, random_state=42)
        lasso.fit(X_train_scaled, y_train)
        
        # Predictions
        y_pred = lasso.predict(X_test_scaled)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred),
            'selected_features': sum(lasso.coef_ != 0)
        }
        
        self.models['lasso'] = lasso
        self.results['lasso'] = metrics
        logger.info(f"LASSO: RMSE = {metrics['rmse']:.2f}, {metrics['selected_features']} features selected")
        
        return metrics
    
    def xgboost_model(self, X: pd.DataFrame, y: pd.Series) -> Dict[str, float]:
        """XGBoost gradient boosting model"""
        logger.info("Training XGBoost...")
        
        split_idx = int(0.8 * len(X))
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]
        
        # Train XGBoost
        xgb_model = xgb.XGBRegressor(
            n_estimators=100,
            max_depth=6,
            learning_rate=0.1,
            random_state=42
        )
        
        xgb_model.fit(X_train, y_train)
        
        # Predictions
        y_pred = xgb_model.predict(X_test)
        
        metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'mae': mean_absolute_error(y_test, y_pred),
            'r2': r2_score(y_test, y_pred)
        }
        
        self.models['xgboost'] = xgb_model
        self.results['xgboost'] = metrics
        logger.info(f"XGBoost: RMSE = {metrics['rmse']:.2f}")
        
        return metrics
    
    def run_all_models(self) -> pd.DataFrame:
        """Run all baseline models and return results"""
        logger.info("Running all baseline models...")
        
        X, y = self.prepare_features()
        
        if len(X) == 0:
            logger.error("No data available for modeling")
            return pd.DataFrame()
        
        # Run models
        self.naive_baseline(X, y)
        self.lasso_regression(X, y)
        self.xgboost_model(X, y)
        
        # Create results summary
        results_df = pd.DataFrame(self.results).T
        logger.info("✓ All baseline models completed")
        
        return results_df
    
    def save_models(self, output_dir: Path):
        """Save trained models"""
        output_dir.mkdir(parents=True, exist_ok=True)
        
        for name, model in self.models.items():
            model_path = output_dir / f"{name}_model.pkl"
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
            logger.info(f"Saved {name} model to {model_path}")