"""
Spatial Autoregressive (SAR) Model Implementation

This script implements:
1. Spatial Autoregressive (Spatial Lag) Model: y = ρWy + Xβ + ε
2. Spatial weight matrix construction from network
3. Spatial coefficient significance testing
4. Comparison with non-spatial models
5. Spatial diagnostics and analysis

Author: Yadon Kassahun (Network Architect) & Kyle Williamson (Data Engineer)
Date: 2025-11-22
"""

import pandas as pd
import numpy as np
import pickle
import json
import networkx as nx
from pathlib import Path
import logging
from typing import Dict, Tuple, List
from scipy import sparse
from scipy.stats import t as t_dist
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class SARModel:
    """Spatial Autoregressive Model for GTA Real Estate Hotspots"""
    
    def __init__(
        self,
        train_path: str,
        val_path: str,
        test_path: str,
        network_path: str,
        output_dir: str = 'results'
    ):
        self.train_path = Path(train_path)
        self.val_path = Path(val_path)
        self.test_path = Path(test_path)
        self.network_path = Path(network_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Data
        self.train_df = None
        self.val_df = None
        self.test_df = None
        self.network = None
        
        # Spatial weights
        self.W_train = None
        self.W_val = None
        self.W_test = None
        
        # Features and targets
        self.feature_cols = None
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.y_val = None
        self.X_test = None
        self.y_test = None
        
        # Models
        self.sar_model = None
        self.ols_model = None  # For comparison
        
        # Results
        self.results = {}
        
    def load_data(self) -> 'SARModel':
        """Load train, validation, test datasets and spatial network"""
        logger.info("="*60)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*60)
        
        self.train_df = pd.read_csv(self.train_path)
        self.val_df = pd.read_csv(self.val_path)
        self.test_df = pd.read_csv(self.test_path)
        
        logger.info(f"Train set: {len(self.train_df)} records")
        logger.info(f"Validation set: {len(self.val_df)} records")
        logger.info(f"Test set: {len(self.test_df)} records")
        
        # Load spatial network
        logger.info(f"\nLoading spatial network: {self.network_path}")
        with open(self.network_path, 'rb') as f:
            self.network = pickle.load(f)
        
        logger.info(f"Network loaded: {self.network.number_of_nodes()} nodes, {self.network.number_of_edges()} edges")
        
        return self
    
    def prepare_features(self) -> 'SARModel':
        """Prepare feature matrices and target vectors"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: PREPARING FEATURES")
        logger.info("="*60)
        
        # Define feature columns (exclude spatial lag features to avoid circularity)
        exclude_cols = [
            'FSA', 'Year', 'Permit_Count_Next_Year',
            'Permit_Growth', 'Permit_Growth_Pct', 'Is_Hotspot',
            'Spatial_Lag_Permits', 'Spatial_Lag_Value'  # Exclude pre-computed spatial lags
        ]
        
        self.feature_cols = [
            col for col in self.train_df.columns 
            if col not in exclude_cols
        ]
        
        logger.info(f"Using {len(self.feature_cols)} features (excluding pre-computed spatial lags):")
        for col in self.feature_cols:
            logger.info(f"  - {col}")
        
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
        
        logger.info(f"\nFeature matrix shapes:")
        logger.info(f"  X_train: {self.X_train.shape}")
        logger.info(f"  X_val: {self.X_val.shape}")
        logger.info(f"  X_test: {self.X_test.shape}")
        
        return self
    
    def create_spatial_weights(self) -> 'SARModel':
        """Create spatial weight matrices from network"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CREATING SPATIAL WEIGHT MATRICES")
        logger.info("="*60)
        
        logger.info("Creating spatial weights from network adjacency...")
        
        # Create weight matrices for each split
        self.W_train = self._create_weight_matrix(self.train_df, normalize=True)
        self.W_val = self._create_weight_matrix(self.val_df, normalize=True)
        self.W_test = self._create_weight_matrix(self.test_df, normalize=True)
        
        logger.info(f"Train W matrix: {self.W_train.shape}, {self.W_train.nnz} non-zero entries")
        logger.info(f"Val W matrix: {self.W_val.shape}, {self.W_val.nnz} non-zero entries")
        logger.info(f"Test W matrix: {self.W_test.shape}, {self.W_test.nnz} non-zero entries")
        
        # Calculate spatial statistics
        self._calculate_spatial_statistics()
        
        return self
    
    def _create_weight_matrix(self, df: pd.DataFrame, normalize: bool = True) -> sparse.csr_matrix:
        """Create spatial weight matrix from network for given dataframe
        
        For panel data (FSA-year), create block-diagonal W where each block
        is the spatial weight matrix for that year's FSAs.
        """
        
        # Get number of observations
        n_obs = len(df)
        
        # Create FSA-to-observation index mapping
        fsa_list = df['FSA'].tolist()
        
        # Create adjacency matrix from network
        rows = []
        cols = []
        weights = []
        
        for i, fsa_i in enumerate(fsa_list):
            if fsa_i not in self.network:
                continue
            
            neighbors = list(self.network.neighbors(fsa_i))
            
            for fsa_j in neighbors:
                # Find all observations with FSA j in same year as observation i
                year_i = df.iloc[i]['Year']
                
                for j, (fsa_j_obs, year_j) in enumerate(zip(df['FSA'], df['Year'])):
                    if fsa_j_obs == fsa_j and year_i == year_j:
                        # Get edge weight (distance) - use inverse distance for weight
                        edge_data = self.network.get_edge_data(fsa_i, fsa_j)
                        if edge_data and 'distance_km' in edge_data:
                            distance = edge_data['distance_km']
                            weight = 1.0 / (distance + 0.1)  # Inverse distance weight
                        else:
                            weight = 1.0  # Default weight
                        
                        rows.append(i)
                        cols.append(j)
                        weights.append(weight)
        
        # Create sparse matrix
        W = sparse.csr_matrix((weights, (rows, cols)), shape=(n_obs, n_obs))
        
        # Row-normalize (each row sums to 1)
        if normalize:
            row_sums = np.array(W.sum(axis=1)).flatten()
            row_sums[row_sums == 0] = 1  # Avoid division by zero
            D_inv = sparse.diags(1.0 / row_sums)
            W = D_inv @ W
        
        return W
    
    def _calculate_spatial_statistics(self) -> None:
        """Calculate spatial autocorrelation statistics"""
        logger.info("\nSpatial autocorrelation analysis:")
        
        # Moran's I for training data
        morans_i = self._calculate_morans_i(self.y_train.values, self.W_train)
        logger.info(f"  Moran's I (train): {morans_i:.4f}")
        
        if morans_i > 0:
            logger.info("  Interpretation: Positive spatial autocorrelation (clustering)")
        elif morans_i < 0:
            logger.info("  Interpretation: Negative spatial autocorrelation (dispersion)")
        else:
            logger.info("  Interpretation: No spatial autocorrelation (random)")
    
    def _calculate_morans_i(self, y: np.ndarray, W: sparse.csr_matrix) -> float:
        """Calculate Moran's I statistic for spatial autocorrelation"""
        n = len(y)
        
        # Demean
        y_mean = y.mean()
        y_dev = y - y_mean
        
        # Calculate Moran's I
        numerator = n * (y_dev @ W @ y_dev)
        denominator = W.sum() * (y_dev @ y_dev)
        
        if denominator == 0:
            return 0.0
        
        morans_i = numerator / denominator
        return morans_i
    
    def train_ols_baseline(self) -> 'SARModel':
        """Train OLS model (non-spatial baseline for comparison)"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4A: OLS BASELINE (NON-SPATIAL)")
        logger.info("="*60)
        
        logger.info("Training OLS regression (without spatial lag)...")
        
        # Add intercept
        X_train_with_intercept = np.column_stack([np.ones(len(self.X_train)), self.X_train])
        X_val_with_intercept = np.column_stack([np.ones(len(self.X_val)), self.X_val])
        X_test_with_intercept = np.column_stack([np.ones(len(self.X_test)), self.X_test])
        
        # OLS estimation: β = (X'X)^(-1) X'y
        XtX = X_train_with_intercept.T @ X_train_with_intercept
        Xty = X_train_with_intercept.T @ self.y_train.values
        
        # Add small ridge for numerical stability
        XtX += np.eye(XtX.shape[0]) * 1e-6
        
        beta_ols = np.linalg.solve(XtX, Xty)
        
        # Predictions
        y_train_pred = X_train_with_intercept @ beta_ols
        y_val_pred = X_val_with_intercept @ beta_ols
        y_test_pred = X_test_with_intercept @ beta_ols
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train.values, y_train_pred, 'Train')
        val_metrics = self._calculate_metrics(self.y_val.values, y_val_pred, 'Validation')
        test_metrics = self._calculate_metrics(self.y_test.values, y_test_pred, 'Test')
        
        self.ols_model = {
            'beta': beta_ols,
            'feature_names': ['Intercept'] + self.feature_cols
        }
        
        self.results['ols'] = {
            'model_type': 'OLS (Non-Spatial)',
            'description': 'Ordinary Least Squares without spatial lag',
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        logger.info("\nOLS Results:")
        logger.info(f"  Train RMSE:      {train_metrics['rmse']:.2f}")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"  Test RMSE:       {test_metrics['rmse']:.2f}")
        logger.info(f"  Test MAE:        {test_metrics['mae']:.2f}")
        logger.info(f"  Test R²:         {test_metrics['r2']:.4f}")
        
        return self
    
    def train_sar_model(self) -> 'SARModel':
        """Train Spatial Autoregressive (Spatial Lag) Model"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4B: SPATIAL AUTOREGRESSIVE (SAR) MODEL")
        logger.info("="*60)
        
        logger.info("Training SAR model: y = ρWy + Xβ + ε")
        logger.info("Using 2SLS/GMM estimation procedure...")
        
        # Prepare data
        y = self.y_train.values.reshape(-1, 1)
        X = self.X_train.values
        W = self.W_train
        n = len(y)
        
        # Add intercept to X
        X_with_intercept = np.column_stack([np.ones(n), X])
        k = X_with_intercept.shape[1]
        
        # Two-Stage Least Squares (2SLS) estimation
        # Stage 1: Instrument for Wy using WX
        WX = W @ X_with_intercept
        Wy = (W @ y).flatten()
        
        # Combine instruments: Z = [X, WX]
        Z = np.column_stack([X_with_intercept, WX])
        
        # Stage 2: Regress [Wy, X] on Z to get predicted values
        # Then regress y on [predicted_Wy, X]
        
        # Endogenous variable matrix: H = [Wy, X]
        H = np.column_stack([Wy.reshape(-1, 1), X_with_intercept])
        
        # 2SLS formula: θ = (H'Z(Z'Z)^(-1)Z'H)^(-1) (H'Z(Z'Z)^(-1)Z'y)
        ZtZ_inv = np.linalg.inv(Z.T @ Z + np.eye(Z.shape[1]) * 1e-6)
        P_Z = Z @ ZtZ_inv @ Z.T  # Projection matrix
        
        # First stage: Predicted H
        H_pred = P_Z @ H
        
        # Second stage: Coefficients
        HtH_pred = H_pred.T @ H_pred
        Hty = H_pred.T @ y
        
        try:
            coeffs = np.linalg.solve(HtH_pred + np.eye(HtH_pred.shape[0]) * 1e-6, Hty).flatten()
        except:
            logger.warning("2SLS solution unstable, using simplified estimation")
            # Fallback: simple IV estimation
            coeffs = np.linalg.lstsq(H_pred, y, rcond=None)[0].flatten()
        
        rho = coeffs[0]  # Spatial autoregressive coefficient
        beta = coeffs[1:]  # Other coefficients
        
        # Calculate standard errors
        # Residuals
        y_fitted = rho * Wy + X_with_intercept @ beta
        residuals = y.flatten() - y_fitted
        sigma2 = (residuals @ residuals) / (n - len(coeffs))
        
        # Variance-covariance matrix (simplified)
        try:
            var_cov = sigma2 * np.linalg.inv(HtH_pred + np.eye(HtH_pred.shape[0]) * 1e-6)
            std_errors = np.sqrt(np.maximum(np.diag(var_cov), 0))
        except:
            std_errors = np.ones(len(coeffs)) * np.nan
        
        # T-statistics and p-values
        t_stats = coeffs / (std_errors + 1e-10)
        p_values = 2 * (1 - t_dist.cdf(np.abs(t_stats), df=n - len(coeffs)))
        
        logger.info("\n--- SAR Model Coefficients ---")
        logger.info(f"{'Variable':<30} {'Coeff':<12} {'Std Err':<12} {'t-stat':<10} {'p-value':<10}")
        logger.info("-" * 74)
        logger.info(f"{'ρ (Spatial Lag)':<30} {rho:<12.4f} {std_errors[0]:<12.4f} {t_stats[0]:<10.4f} {p_values[0]:<10.4f}")
        
        if p_values[0] < 0.05:
            logger.info(f"  ✓ Spatial coefficient ρ is SIGNIFICANT (p < 0.05)")
        else:
            logger.info(f"  ✗ Spatial coefficient ρ is NOT significant (p >= 0.05)")
        
        logger.info(f"{'Intercept':<30} {beta[0]:<12.4f} {std_errors[1]:<12.4f} {t_stats[1]:<10.4f} {p_values[1]:<10.4f}")
        
        # Store model
        self.sar_model = {
            'rho': float(rho),
            'beta': beta,
            'coefficients': coeffs,
            'std_errors': std_errors,
            't_stats': t_stats,
            'p_values': p_values,
            'sigma2': float(sigma2),
            'feature_names': ['Spatial_Lag'] + ['Intercept'] + self.feature_cols
        }
        
        # Make predictions
        y_train_pred = self._predict_sar(self.X_train.values, self.y_train.values, self.W_train, rho, beta)
        y_val_pred = self._predict_sar(self.X_val.values, self.y_val.values, self.W_val, rho, beta)
        y_test_pred = self._predict_sar(self.X_test.values, self.y_test.values, self.W_test, rho, beta)
        
        # Calculate metrics
        train_metrics = self._calculate_metrics(self.y_train.values, y_train_pred, 'Train')
        val_metrics = self._calculate_metrics(self.y_val.values, y_val_pred, 'Validation')
        test_metrics = self._calculate_metrics(self.y_test.values, y_test_pred, 'Test')
        
        self.results['sar'] = {
            'model_type': 'SAR (Spatial Lag)',
            'description': 'Spatial Autoregressive Model with spatial dependency',
            'rho': float(rho),
            'rho_pvalue': float(p_values[0]),
            'rho_significant': bool(p_values[0] < 0.05),
            'train': train_metrics,
            'validation': val_metrics,
            'test': test_metrics
        }
        
        logger.info("\nSAR Model Results:")
        logger.info(f"  Spatial coefficient ρ: {rho:.4f} (p={p_values[0]:.4f})")
        logger.info(f"  Train RMSE:      {train_metrics['rmse']:.2f}")
        logger.info(f"  Validation RMSE: {val_metrics['rmse']:.2f}")
        logger.info(f"  Test RMSE:       {test_metrics['rmse']:.2f}")
        logger.info(f"  Test MAE:        {test_metrics['mae']:.2f}")
        logger.info(f"  Test R²:         {test_metrics['r2']:.4f}")
        
        return self
    
    def _predict_sar(self, X: np.ndarray, y_true: np.ndarray, W: sparse.csr_matrix, 
                     rho: float, beta: np.ndarray) -> np.ndarray:
        """Make predictions using SAR model"""
        n = len(X)
        X_with_intercept = np.column_stack([np.ones(n), X])
        
        # For prediction: y = (I - ρW)^(-1) Xβ
        # Approximate using: y ≈ Xβ + ρWy_true (using true values for spatial lag)
        Wy = W @ y_true
        y_pred = rho * Wy + X_with_intercept @ beta
        
        return y_pred
    
    def compare_spatial_vs_nonspatial(self, baseline_results_path: str = None) -> 'SARModel':
        """Compare SAR with non-spatial models"""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: SPATIAL VS NON-SPATIAL COMPARISON")
        logger.info("="*60)
        
        # Create comparison table
        comparison_data = []
        
        # Add OLS
        comparison_data.append({
            'Model': 'OLS (Non-Spatial)',
            'Type': 'Non-Spatial',
            'RMSE': self.results['ols']['test']['rmse'],
            'MAE': self.results['ols']['test']['mae'],
            'R²': self.results['ols']['test']['r2']
        })
        
        # Add SAR
        comparison_data.append({
            'Model': f"SAR (ρ={self.sar_model['rho']:.3f})",
            'Type': 'Spatial',
            'RMSE': self.results['sar']['test']['rmse'],
            'MAE': self.results['sar']['test']['mae'],
            'R²': self.results['sar']['test']['r2']
        })
        
        # Load baseline results if available
        if baseline_results_path:
            baseline_path = Path(baseline_results_path)
            if baseline_path.exists():
                with open(baseline_path, 'r') as f:
                    baseline_results = json.load(f)
                
                for model_name, model_results in baseline_results.items():
                    comparison_data.append({
                        'Model': model_name.replace('_', ' ').title(),
                        'Type': 'Non-Spatial',
                        'RMSE': model_results['test']['rmse'],
                        'MAE': model_results['test']['mae'],
                        'R²': model_results['test']['r2']
                    })
        
        comparison_df = pd.DataFrame(comparison_data)
        
        # Calculate improvements
        ols_rmse = self.results['ols']['test']['rmse']
        sar_rmse = self.results['sar']['test']['rmse']
        improvement_pct = ((ols_rmse - sar_rmse) / ols_rmse) * 100
        
        logger.info("\nModel Comparison (Test Set):")
        logger.info(f"{'Model':<30} {'Type':<15} {'RMSE':<10} {'MAE':<10} {'R²':<10}")
        logger.info("-" * 75)
        for _, row in comparison_df.iterrows():
            logger.info(f"{row['Model']:<30} {row['Type']:<15} {row['RMSE']:<10.2f} {row['MAE']:<10.2f} {row['R²']:<10.4f}")
        
        logger.info(f"\nSAR improvement over OLS: {improvement_pct:.1f}%")
        
        if improvement_pct > 0:
            logger.info("  ✓ SAR model outperforms non-spatial OLS!")
        else:
            logger.info("  ✗ Spatial effects not significant for this problem")
        
        # Save comparison
        comparison_path = self.output_dir / 'spatial_comparison.csv'
        comparison_df.to_csv(comparison_path, index=False)
        logger.info(f"\nSaved comparison table: {comparison_path}")
        
        return self
    
    def _calculate_metrics(self, y_true: np.ndarray, y_pred: np.ndarray, dataset_name: str) -> Dict:
        """Calculate regression metrics"""
        
        # Remove any NaN values
        mask = ~(np.isnan(y_true) | np.isnan(y_pred))
        y_true_clean = y_true[mask]
        y_pred_clean = y_pred[mask]
        
        rmse = np.sqrt(np.mean((y_true_clean - y_pred_clean)**2))
        mae = np.mean(np.abs(y_true_clean - y_pred_clean))
        
        ss_res = np.sum((y_true_clean - y_pred_clean)**2)
        ss_tot = np.sum((y_true_clean - y_true_clean.mean())**2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0
        
        return {
            'rmse': float(rmse),
            'mae': float(mae),
            'r2': float(r2),
            'n_samples': int(len(y_true_clean))
        }
    
    def save_results(self) -> 'SARModel':
        """Save models and results"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: SAVING RESULTS")
        logger.info("="*60)
        
        # Create models directory
        models_dir = self.output_dir / 'models'
        models_dir.mkdir(exist_ok=True)
        
        # Save SAR model
        sar_path = models_dir / 'sar_model.pkl'
        with open(sar_path, 'wb') as f:
            pickle.dump(self.sar_model, f)
        logger.info(f"Saved SAR model: {sar_path}")
        
        # Save OLS model
        ols_path = models_dir / 'ols_model.pkl'
        with open(ols_path, 'wb') as f:
            pickle.dump(self.ols_model, f)
        logger.info(f"Saved OLS model: {ols_path}")
        
        # Save results as JSON
        results_path = self.output_dir / 'sar_results.json'
        with open(results_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Saved results JSON: {results_path}")
        
        # Save SAR coefficients
        if self.sar_model:
            coeffs_df = pd.DataFrame({
                'variable': self.sar_model['feature_names'],
                'coefficient': self.sar_model['coefficients'],
                'std_error': self.sar_model['std_errors'],
                't_statistic': self.sar_model['t_stats'],
                'p_value': self.sar_model['p_values']
            })
            coeffs_path = self.output_dir / 'sar_coefficients.csv'
            coeffs_df.to_csv(coeffs_path, index=False)
            logger.info(f"Saved SAR coefficients: {coeffs_path}")
        
        return self
    
    def generate_summary_report(self) -> 'SARModel':
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*60)
        logger.info("SPATIAL AUTOREGRESSIVE MODEL SUMMARY")
        logger.info("="*60)
        
        logger.info("\n--- SPATIAL STATISTICS ---")
        morans_i = self._calculate_morans_i(self.y_train.values, self.W_train)
        logger.info(f"Moran's I: {morans_i:.4f}")
        
        logger.info("\n--- SAR MODEL RESULTS ---")
        rho = self.sar_model['rho']
        p_val = self.sar_model['p_values'][0]
        logger.info(f"Spatial coefficient (ρ): {rho:.4f}")
        logger.info(f"P-value: {p_val:.4f}")
        logger.info(f"Significance: {'YES (p < 0.05)' if p_val < 0.05 else 'NO (p >= 0.05)'}")
        
        logger.info("\n--- PERFORMANCE COMPARISON (Test Set) ---")
        logger.info(f"OLS (Non-Spatial):  RMSE = {self.results['ols']['test']['rmse']:.2f}")
        logger.info(f"SAR (Spatial):      RMSE = {self.results['sar']['test']['rmse']:.2f}")
        
        improvement = ((self.results['ols']['test']['rmse'] - self.results['sar']['test']['rmse']) / 
                       self.results['ols']['test']['rmse']) * 100
        logger.info(f"SAR improvement:    {improvement:.1f}%")
        
        logger.info("\n--- OUTPUT FILES ---")
        logger.info(f"  Models: {self.output_dir}/models/")
        logger.info(f"  Results: {self.output_dir}/sar_results.json")
        logger.info(f"  Coefficients: {self.output_dir}/sar_coefficients.csv")
        logger.info(f"  Comparison: {self.output_dir}/spatial_comparison.csv")
        
        logger.info("\n" + "="*60)
        logger.info("SAR MODEL COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return self
    
    def run_pipeline(self, baseline_results_path: str = None) -> 'SARModel':
        """Execute complete SAR pipeline"""
        (self
            .load_data()
            .prepare_features()
            .create_spatial_weights()
            .train_ols_baseline()
            .train_sar_model())
        
        if baseline_results_path:
            self.compare_spatial_vs_nonspatial(baseline_results_path)
        
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
    NETWORK_PATH = "data/processed/networks/spatial_network_distance.gpickle"
    OUTPUT_DIR = "results/sar_model"
    BASELINE_RESULTS = "results/baseline_models/baseline_results.json"
    
    logger.info("="*60)
    logger.info("GTA REAL ESTATE HOTSPOTS - SAR MODEL")
    logger.info("="*60)
    logger.info(f"Train data: {TRAIN_PATH}")
    logger.info(f"Validation data: {VAL_PATH}")
    logger.info(f"Test data: {TEST_PATH}")
    logger.info(f"Network: {NETWORK_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*60 + "\n")
    
    try:
        pipeline = SARModel(
            train_path=TRAIN_PATH,
            val_path=VAL_PATH,
            test_path=TEST_PATH,
            network_path=NETWORK_PATH,
            output_dir=OUTPUT_DIR
        )
        
        pipeline.run_pipeline(baseline_results_path=BASELINE_RESULTS)
        
        logger.info("\n✓ All processing completed successfully!")
        logger.info(f"✓ Results saved to: {OUTPUT_DIR}")
        logger.info("\nNext steps:")
        logger.info("  1. Review results in results/sar_model/")
        logger.info("  2. Check spatial_comparison.csv")
        logger.info("  3. Begin writing final report!")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()