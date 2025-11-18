"""
XGBoost Model Implementation

This script implements:
1. XGBoost gradient boosting regressor
2. Hyperparameter tuning via GridSearchCV
3. Feature importance analysis (gain, split, cover)
4. Model comparison with baselines
5. Results export and visualization

Author: Utsav Patel (Modeler) & Kyle Williamson (Data Engineer)
Date: 2024-11-25
"""

import pandas as pd
import numpy as np
import pickle
import json
from pathlib import Path
import logging
from typing import Dict, Tuple, List
import xgboost as xgb
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class XGBoostModel:
    """XGBoost model for GTA Real Estate Hotspots prediction"""
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        output_dir: str = 'results'
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Features and targets
        self.feature_cols = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Models
        self.xgb_model = None
        self.best_params = None
        
        # Results
        self.results = {}
        self.feature_importance = {}
        
    def load_data(self) -> 'XGBoostModel':
        """Load train, validation, and test datasets"""
        logger.info("="*60)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*60)
        
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)
        
        logger.info(f"Train set: {len(self.train_df)} records")
        logger.info(f"Validation set: {len(self.val_df)} records")
        logger.info(f"Test set: {len(self.test_df)} records")
        
        return self
    
    def prepare_features(self) -> 'XGBoostModel':
        """Prepare feature matrices and target vectors"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PREPARING FEATURES")
        logger.info("="*60)
        
        # Define feature columns
        exclude_cols = [
            'FSA', 'Year', 'Permit_Count_Next_Year',
            'Permit_Growth', 'Permit_Growth_Pct', 'Is_Hotspot'
        ]
        
        self.feature_cols = [
            col for col in self.train_df.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Using {len(self.feature_cols)} features")
        
        # Extract features and target
        self.X_train = self.train_df[self.feature_cols].copy()
        self.y_train = self.train_df['Permit_Growth'].copy()
        
        self.X_val = self.val_df[self.feature_cols].copy()
        self.y_val = self.val_df['Permit_Growth'].copy()
        
        self.X_test = self.test_df[self.feature_cols].copy()
        self.y_test = self.test_df['Permit_Growth'].copy()
        
        # Handle missing values
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        logger.info(f"Feature matrix shapes:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  X_val: {self.X_val.shape}")
        logger.info(f"  X_test: {self.X_test.shape}")
        
        return self
    
    def train_xgboost_with_tuning(self, quick_mode: bool = False) -> 'XGBoostModel':
        """Train XGBoost with hyperparameter tuning"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: XGBOOST WITH HYPERPARAMETER TUNING")
        logger.info("="*60)
        
        if quick_mode:
            logger.info("Running in QUICK MODE (reduced parameter grid)")
            param_grid = {
                'n_estimators': [100, 200],
                'max_depth': [4, 6],
                'learning_rate': [0.05, 0.1],
                'subsample': [0.8],
                'colsample_bytree': [0.8]
            }
        else:
            logger.info("Running in FULL MODE (comprehensive grid search)")
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [3, 4, 6, 8],
                'learning_rate': [0.01, 0.05, 0.1],
                'subsample': [0.7, 0.8, 0.9],
                'colsample_bytree': [0.7, 0.8, 0.9],
                'min_child_weight': [1, 3, 5]
            }
        
        logger.info(f"Parameter grid size: {np.prod([len(v) for v in param_grid.values()])} combinations")
        
        # Base model
        base_model = xgb.XGBRegressor(
            objective='reg:squarederror',
            random_state=42,
            n_jobs=-1
        )
        
        # Grid search with cross-validation
        logger.info("Starting GridSearchCV (5-fold)...")
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=5,
            scoring='neg_root_mean_squared_error',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(self.X_train, self.y_train)
        
        # Best parameters
        self.best_params = grid_search.best_params_
        logger.info("\nBest hyperparameters found:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info(f"Best CV RMSE: {-grid_search.best_score_:.2f}")
        
        # Train final model with best parameters
        self.xgb_model = grid_search.best_estimator_
        
        return self
    
    def evaluate_model(self) -> 'XGBoostModel':
        """Evaluate XGBoost model on validation and test sets"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: MODEL EVALUATION")
        logger.info("="*60)
        
        # Make predictions
        y_val_pred = self.xgb_model.predict(self.X_val)
        y_test_pred = self.xgb_model.predict(self.X_test)
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(self.y_val, y_val_pred, 'Validation')
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred, 'Test')
        
        self.results['xgboost'] = {
            'model_type': 'XGBoost Gradient Boosting',
            'description': 'Gradient boosted decision trees',
            'best_params': self.best_params,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        logger.info("\nXGBoost Results:")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"  Validation MAE:  {val_metrics['mae']:.2f}")
        logger.info(f"  Validation R²:   {val_metrics['r2']:.4f}")
        logger.info(f"  Test RMSE:       {test_metrics['rmse']:.2f}")
        logger.info(f"  Test MAE:        {test_metrics['mae']:.2f}")
        logger.info(f"  Test R²:         {test_metrics['r2']:.4f}")
        
        return self
    
    def analyze_feature_importance(self) -> 'XGBoostModel':
        """Analyze feature importance using multiple metrics"""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        # Use feature_importances_ attribute (more reliable)
        importance_gain = self.xgb_model.feature_importances_
        
        # Create DataFrame
        importance_data = []
        for i, feature in enumerate(self.feature_cols):
            importance_data.append({
                'feature': feature,
                'importance_gain': float(importance_gain[i])
            })
        
        importance_df = pd.DataFrame(importance_data)
        
        # Sort by gain
        importance_df = importance_df.sort_values('importance_gain', ascending=False)
        
        self.feature_importance = importance_df
        
        logger.info("\nTop 10 features by importance (gain):")
        for i, row in importance_df.head(10).iterrows():
            logger.info(f"  {row['feature']}: {row['importance_gain']:.4f}")
        
        return self
    
    def compare_with_baselines(self, baseline_results_path: str) -> 'XGBoostModel':
        """Compare XGBoost with baseline models"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: COMPARISON WITH BASELINES")
        logger.info("="*60)
        
        # Load baseline results
        baseline_path = Path(baseline_results_path)
        if not baseline_path.exists():
            logger.warning(f"Baseline results not found: {baseline_path}")
            logger.warning("Skipping comparison")
            return self
        
        with open(baseline_path, 'r') as f:
            baseline_results = json.load(f)
        
        # Create comparison table
        comparison_data = []
        
        # Add baseline models
        for model_name, model_results in baseline_results.items():
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'RMSE': model_results['test']['rmse'],
                'MAE': model_results['test']['mae'],
                'R²': model_results['test']['r2']
            })
        
        # Add XGBoost
        comparison_data.append({
            'Model': 'XGBoost',
            'RMSE': self.results['xgboost']['test']['rmse'],
            'MAE': self.results['xgboost']['test']['mae'],
            'R²': self.results['xgboost']['test']['r2']
        })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvements
        naive_rmse = baseline_results['naive_baseline']['test']['rmse']
        lasso_rmse = baseline_results['lasso_regression']['test']['rmse']
        xgb_rmse = self.results['xgboost']['test']['rmse']
        
        improvement_vs_naive = ((naive_rmse - xgb_rmse) / naive_rmse) * 100
        improvement_vs_lasso = ((lasso_rmse - xgb_rmse) / lasso_rmse) * 100
        
        logger.info("\nModel Comparison (Test Set):")
        logger.info(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        logger.info("-" * 55)
        for _, row in comparison_df.iterrows():
            logger.info(f"{row['Model']:<25} {row['RMSE']:<10.2f} {row['MAE']:<10.2f} {row['R²']:<10.4f}")
        
        logger.info(f"\nXGBoost improvements:")
        logger.info(f"  vs. Naive Baseline: {improvement_vs_naive:.1f}%")
        logger.info(f"  vs. LASSO: {improvement_vs_lasso:.1f}%")
        
        # Save comparison
        comparison_path = self.output_dir / 'all_models_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison table: {comparison_path}")
        
        return self
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict:
        """Calculate regression metrics"""
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        rmse = np.sqrt(mean_squared_error(y_true_clean, y_pred_clean))
        mae = mean_absolute_error(y_true_clean, y_pred_clean)
        r2 = r2_score(y_true_clean, y_pred_clean)
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': int(len(y_true_clean))
        }
    
    def save_results(self) -> 'XGBoostModel':
        """Save model and results"""
        logger.info("\n" + "="*60)
        logger.info("STEP 7: SAVING RESULTS")
        logger.info("="*60)
        
        # Create models directory
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save XGBoost model
        model_path = models_dir / 'xgboost_model.pkl'
        with open(model_path, 'wb') as f:
            pickle.dump(self.xgb_model, f)
        logger.info(f"Saved XGBoost model: {model_path}")
        
        # Save results as JSON
        results_path = self.output_dir / 'xgboost_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results JSON: {results_path}")
        
        # Save feature importance
        importance_path = self.output_dir / 'xgboost_feature_importance.csv'
        self.feature_importance.to_csv(importance_path, index=False)
        logger.info(f"Saved feature importance: {importance_path}")
        
        return self
    
    def generate_summary_report(self) -> 'XGBoostModel':
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*60)
        logger.info("XGBOOST MODEL SUMMARY REPORT")
        logger.info("="*60)
        
        logger.info("\n--- MODEL CONFIGURATION ---")
        logger.info("Best hyperparameters:")
        for param, value in self.best_params.items():
            logger.info(f"  {param}: {value}")
        
        logger.info("\n--- PERFORMANCE SUMMARY (Test Set) ---")
        test_metrics = self.results['xgboost']['test']
        logger.info(f"RMSE: {test_metrics['rmse']:.2f} permits")
        logger.info(f"MAE:  {test_metrics['mae']:.2f} permits")
        logger.info(f"R²:   {test_metrics['r2']:.4f}")
        
        logger.info("\n--- TOP 5 FEATURES (by gain) ---")
        for i, row in self.feature_importance.head(5).iterrows():
            logger.info(f"{row['feature']}: {row['importance_gain']:.4f}")
        
        logger.info("\n--- OUTPUT FILES ---")
        logger.info(f"  Model: {self.output_dir}/models/xgboost_model.pkl")
        logger.info(f"  Results: {self.output_dir}/xgboost_results.json")
        logger.info(f"  Feature importance: {self.output_dir}/xgboost_feature_importance.csv")
        logger.info(f"  Comparison: {self.output_dir}/all_models_comparison.csv")
        
        logger.info("\n" + "="*60)
        logger.info("XGBOOST MODEL COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return self
    
    def run_pipeline(self, quick_mode: bool = False, baseline_results_path: str = None) -> 'XGBoostModel':
        """Execute complete XGBoost pipeline"""
        (self
            .load_data()
            .prepare_features()
            .train_xgboost_with_tuning(quick_mode=quick_mode)
            .evaluate_model()
            .analyze_feature_importance())
        
        if baseline_results_path:
            self.compare_with_baselines(baseline_results_path)
        
        (self
            .save_results()
            .generate_summary_report())
        
        return self


def main():
    """Main execution function"""
    
    # Configure paths
    TRAIN_PATH = "data/processed/train_set.csv"
    VAL_PATH = "data/processed/val_set.csv"
    TEST_PATH = "data/processed/test_set.csv"
    OUTPUT_DIR = "results/xgboost_model"
    BASELINE_RESULTS = "results/baseline_models/baseline_results.json"
    
    logger.info("="*60)
    logger.info("GTA REAL ESTATE HOTSPOTS - XGBOOST MODEL")
    logger.info("="*60)
    logger.info(f"Train data: {TRAIN_PATH}")
    logger.info(f"Validation data: {VAL_PATH}")
    logger.info(f"Test data: {TEST_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*60 + "\n")
    
    try:
        pipeline = XGBoostModel(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_path=TEST_PATH,
            output_dir=OUTPUT_DIR
        )
        
        # Run in quick mode for faster testing (set to False for full grid search)
        pipeline.run_pipeline(
            quick_mode=True,  # Set to False for comprehensive tuning
            baseline_results_path=BASELINE_RESULTS
        )
        
        logger.info("\n✓ All processing completed successfully!")
        logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
        logger.info("\nNext steps:")
        logger.info("  1. Review results in results/xgboost_model/")
        logger.info("  2. Check all_models_comparison.csv")
        logger.info("  3. Run SAR model: python src/step4_sar_model.py")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()