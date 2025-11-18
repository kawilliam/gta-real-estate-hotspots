"""
Baseline Models Implementation

This script implements:
1. Naive baseline (persistence model)
2. LASSO regression with feature selection
3. Evaluation framework with RMSE, MAE, R² metrics
4. Model comparison and results export

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
from sklearn.linear_model import Lasso, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import seaborn as sns

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class BaselineModels:
    """Baseline models for GTA Real Estate Hotspots prediction"""
    
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
        self.naive_model = None
        self.lasso_model = None
        self.scaler = None
        
        # Results
        self.results = {}
        
    def load_data(self) -> 'BaselineModels':
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
    
    def prepare_features(self) -> 'BaselineModels':
        """Prepare feature matrices and target vectors"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PREPARING FEATURES")
        logger.info("="*60)
        
        # Define feature columns (exclude metadata and target columns)
        exclude_cols = [
            'FSA', 'Year', 'Permit_Count_Next_Year',
            'Permit_Growth', 'Permit_Growth_Pct', 'Is_Hotspot'
        ]
        
        self.feature_cols = [
            col for col in self.train_df.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Using {len(self.feature_cols)} features:")
        for col in self.feature_cols:
            logger.info(f"  - {col}")
        
        # Extract features and target
        self.X_train = self.train_df[self.feature_cols].copy()
        self.y_train = self.train_df['Permit_Growth'].copy()
        
        self.X_val = self.val_df[self.feature_cols].copy()
        self.y_val = self.val_df['Permit_Growth'].copy()
        
        self.X_test = self.test_df[self.feature_cols].copy()
        self.y_test = self.test_df['Permit_Growth'].copy()
        
        # Handle missing values (fill with 0 for now)
        self.X_train = self.X_train.fillna(0)
        self.X_val = self.X_val.fillna(0)
        self.X_test = self.X_test.fillna(0)
        
        logger.info(f"\nFeature matrix shapes:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  X_val: {self.X_val.shape}")
        logger.info(f"  X_test: {self.X_test.shape}")
        
        logger.info(f"\nTarget statistics (Permit_Growth):")
        logger.info(f"  Train - Mean: {self.y_train.mean():.2f}, Std: {self.y_train.std():.2f}")
        logger.info(f"  Val   - Mean: {self.y_val.mean():.2f}, Std: {self.y_val.std():.2f}")
        logger.info(f"  Test  - Mean: {self.y_test.mean():.2f}, Std: {self.y_test.std():.2f}")
        
        return self
    
    def train_naive_baseline(self) -> 'BaselineModels':
        """Train naive persistence baseline model"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: NAIVE BASELINE MODEL")
        logger.info("="*60)
        
        logger.info("Naive baseline: Predict next-year growth = last-year growth")
        logger.info("  Formula: Δy(t+1) = Δy(t)")
        
        # For validation: Use 'Permit_Growth_1yr' feature as prediction
        # This represents the last observed growth
        y_val_pred_naive = self.X_val['Permit_Growth_1yr'].values
        
        # For test: Use validation year's actual growth
        y_test_pred_naive = self.X_test['Permit_Growth_1yr'].values
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(self.y_val, y_val_pred_naive, 'Validation')
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred_naive, 'Test')
        
        self.results['naive_baseline'] = {
            'model_type': 'Naive Persistence',
            'description': 'Predict Δy(t+1) = Δy(t)',
            'validation': val_metrics,
            'test': test_metrics
        }
        
        logger.info("\nNaive Baseline Results:")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"  Validation MAE:  {val_metrics['mae']:.2f}")
        logger.info(f"  Validation R²:   {val_metrics['r2']:.4f}")
        logger.info(f"  Test RMSE:       {test_metrics['rmse']:.2f}")
        logger.info(f"  Test MAE:        {test_metrics['mae']:.2f}")
        logger.info(f"  Test R²:         {test_metrics['r2']:.4f}")
        
        return self
    
    def train_lasso_regression(self, alpha_range: Tuple[float, float] = (0.01, 10.0)) -> 'BaselineModels':
        """Train LASSO regression with cross-validation for alpha selection"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: LASSO REGRESSION")
        logger.info("="*60)
        
        # Standardize features
        logger.info("Standardizing features...")
        self.scaler = StandardScaler()
        X_train_scaled = self.scaler.fit_transform(self.X_train)
        X_val_scaled = self.scaler.transform(self.X_val)
        X_test_scaled = self.scaler.transform(self.X_test)
        
        # Use LassoCV to find optimal alpha
        logger.info(f"Running cross-validation for alpha selection...")
        logger.info(f"  Alpha range: {alpha_range[0]} to {alpha_range[1]}")
        
        alphas = np.logspace(np.log10(alpha_range[0]), np.log10(alpha_range[1]), 50)
        
        lasso_cv = LassoCV(
            alphas=alphas,
            cv=5,
            max_iter=10000,
            random_state=42,
            n_jobs=-1
        )
        
        lasso_cv.fit(X_train_scaled, self.y_train)
        
        best_alpha = lasso_cv.alpha_
        logger.info(f"  Best alpha (CV): {best_alpha:.4f}")
        
        # Train final model with best alpha
        self.lasso_model = Lasso(alpha=best_alpha, max_iter=10000, random_state=42)
        self.lasso_model.fit(X_train_scaled, self.y_train)
        
        # Count selected features
        n_features_selected = np.sum(self.lasso_model.coef_ != 0)
        n_features_total = len(self.lasso_model.coef_)
        
        logger.info(f"  Features selected: {n_features_selected}/{n_features_total}")
        
        # Get selected features and their coefficients
        selected_features = []
        for i, (feature, coef) in enumerate(zip(self.feature_cols, self.lasso_model.coef_)):
            if coef != 0:
                selected_features.append({
                    'feature': feature,
                    'coefficient': float(coef),
                    'abs_coefficient': abs(float(coef))
                })
        
        # Sort by absolute coefficient value
        selected_features = sorted(selected_features, key=lambda x: x['abs_coefficient'], reverse=True)
        
        logger.info("\nTop 10 most important features (by coefficient magnitude):")
        for i, feat in enumerate(selected_features[:10], 1):
            logger.info(f"  {i}. {feat['feature']}: {feat['coefficient']:.4f}")
        
        # Make predictions
        y_val_pred = self.lasso_model.predict(X_val_scaled)
        y_test_pred = self.lasso_model.predict(X_test_scaled)
        
        # Calculate metrics
        val_metrics = self._calculate_metrics(self.y_val, y_val_pred, 'Validation')
        test_metrics = self._calculate_metrics(self.y_test, y_test_pred, 'Test')
        
        self.results['lasso_regression'] = {
            'model_type': 'LASSO Regression',
            'description': 'Linear regression with L1 regularization',
            'alpha': float(best_alpha),
            'n_features_selected': int(n_features_selected),
            'n_features_total': int(n_features_total),
            'selected_features': selected_features,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        logger.info("\nLASSO Regression Results:")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"  Validation MAE:  {val_metrics['mae']:.2f}")
        logger.info(f"  Validation R²:   {val_metrics['r2']:.4f}")
        logger.info(f"  Test RMSE:       {test_metrics['rmse']:.2f}")
        logger.info(f"  Test MAE:        {test_metrics['mae']:.2f}")
        logger.info(f"  Test R²:         {test_metrics['r2']:.4f}")
        
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
    
    def compare_models(self) -> 'BaselineModels':
        """Generate model comparison report"""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: MODEL COMPARISON")
        logger.info("="*60)
        
        # Create comparison DataFrame
        comparison_data = []
        
        for model_name, model_results in self.results.items():
            # Validation metrics
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Dataset': 'Validation',
                'RMSE': model_results['validation']['rmse'],
                'MAE': model_results['validation']['mae'],
                'R²': model_results['validation']['r2']
            })
            
            # Test metrics
            comparison_data.append({
                'Model': model_name.replace('_', ' ').title(),
                'Dataset': 'Test',
                'RMSE': model_results['test']['rmse'],
                'MAE': model_results['test']['mae'],
                'R²': model_results['test']['r2']
            })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvement over naive baseline
        naive_test_rmse = self.results['naive_baseline']['test']['rmse']
        lasso_test_rmse = self.results['lasso_regression']['test']['rmse']
        improvement_pct = ((naive_test_rmse - lasso_test_rmse) / naive_test_rmse) * 100
        
        logger.info("\nModel Comparison (Test Set):")
        logger.info(f"{'Model':<25} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        logger.info("-" * 55)
        for _, row in comparison_df[comparison_df['Dataset'] == 'Test'].iterrows():
            logger.info(f"{row['Model']:<25} {row['RMSE']:<10.2f} {row['MAE']:<10.2f} {row['R²']:<10.4f}")
        
        logger.info(f"\nLASSO improvement over Naive: {improvement_pct:.1f}%")
        
        if improvement_pct > 0:
            logger.info("  LASSO model outperforms naive baseline!")
        else:
            logger.info("  Naive baseline performs better (unusual, check data)")
        
        # Save comparison table
        comparison_path = self.output_dir / 'model_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison table: {comparison_path}")
        
        return self
    
    def save_results(self) -> 'BaselineModels':
        """Save models and results"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("="*60)
        
        # Create models directory
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save LASSO model and scaler
        if self.lasso_model is not None:
            lasso_path = models_dir / 'lasso_model.pkl'
            with open(lasso_path, 'wb') as f:
                pickle.dump(self.lasso_model, f)
            logger.info(f"Saved LASSO model: {lasso_path}")
            
            scaler_path = models_dir / 'feature_scaler.pkl'
            with open(scaler_path, 'wb') as f:
                pickle.dump(self.scaler, f)
            logger.info(f"Saved feature scaler: {scaler_path}")
        
        # Save results as JSON
        results_path = self.output_dir / 'baseline_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results JSON: {results_path}")
        
        # Save feature importance from LASSO
        if 'lasso_regression' in self.results:
            features_df = pd.DataFrame(
                self.results['lasso_regression']['selected_features']
            )
            features_path = self.output_dir / 'lasso_feature_importance.csv'
            features_df.to_csv(features_path, index=False)
            logger.info(f"Saved feature importance: {features_path}")
        
        return self
    
    def generate_summary_report(self) -> 'BaselineModels':
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*60)
        logger.info("BASELINE MODELS SUMMARY REPORT")
        logger.info("="*60)
        
        logger.info("\n--- MODELS TRAINED ---")
        logger.info("1. Naive Baseline (Persistence Model)")
        logger.info("2. LASSO Regression (L1 Regularization)")
        
        logger.info("\n--- PERFORMANCE SUMMARY (Test Set) ---")
        for model_name, model_results in self.results.items():
            logger.info(f"\n{model_name.replace('_', ' ').title()}:")
            test_metrics = model_results['test']
            logger.info(f"  RMSE: {test_metrics['rmse']:.2f} permits")
            logger.info(f"  MAE:  {test_metrics['mae']:.2f} permits")
            logger.info(f"  R²:   {test_metrics['r2']:.4f}")
        
        logger.info("\n--- KEY FINDINGS ---")
        
        # Best model
        test_rmses = {name: results['test']['rmse'] for name, results in self.results.items()}
        best_model = min(test_rmses, key=test_rmses.get)
        logger.info(f"Best model (lowest RMSE): {best_model.replace('_', ' ').title()}")
        
        # Feature selection
        if 'lasso_regression' in self.results:
            n_selected = self.results['lasso_regression']['n_features_selected']
            n_total = self.results['lasso_regression']['n_features_total']
            logger.info(f"LASSO selected {n_selected}/{n_total} features ({n_selected/n_total*100:.1f}%)")
        
        logger.info("\n--- OUTPUT FILES ---")
        logger.info(f"  Models: {self.output_dir}/models/")
        logger.info(f"  Results: {self.output_dir}/baseline_results.json")
        logger.info(f"  Comparison: {self.output_dir}/model_comparison.csv")
        logger.info(f"  Features: {self.output_dir}/lasso_feature_importance.csv")
        
        logger.info("\n" + "="*60)
        logger.info("BASELINE MODELS COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return self
    
    def run_pipeline(self) -> 'BaselineModels':
        """Execute complete baseline models pipeline"""
        (self
            .load_data()
            .prepare_features()
            .train_naive_baseline()
            .train_lasso_regression()
            .compare_models()
            .save_results()
            .generate_summary_report())
        
        return self


def main():
    """Main execution function"""
    
    # Configure paths
    TRAIN_PATH = "data/processed/train_set.csv"
    VAL_PATH = "data/processed/val_set.csv"
    TEST_PATH = "data/processed/test_set.csv"
    OUTPUT_DIR = "results/baseline_models"
    
    logger.info("="*60)
    logger.info("GTA REAL ESTATE HOTSPOTS - BASELINE MODELS")
    logger.info("="*60)
    logger.info(f"Train data: {TRAIN_PATH}")
    logger.info(f"Validation data: {VAL_PATH}")
    logger.info(f"Test data: {TEST_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*60 + "\n")
    
    try:
        pipeline = BaselineModels(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_path=TEST_PATH,
            output_dir=OUTPUT_DIR
        )
        
        pipeline.run_pipeline()
        
        logger.info("\n✓ All processing completed successfully!")
        logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
        logger.info("\nNext steps:")
        logger.info("  1. Review results in results/baseline_models/")
        logger.info("  2. Check model_comparison.csv for performance")
        logger.info("  3. Run XGBoost model: python src/step3_xgboost_model.py")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()