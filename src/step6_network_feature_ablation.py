"""
Network Feature Ablation Study

Re-runs models with network features included and tests their contribution.

Author: Kyle Williamson + Utsav Patel
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import json
from pathlib import Path
import logging
from sklearn.linear_model import Lasso
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class NetworkFeatureAblation:
    """Ablation study for network features"""
    
    def __init__(self):
        self.train_df = pd.read_csv('data/processed/train_set.csv')
        self.test_df = pd.read_csv('data/processed/test_set.csv')
        self.results = {}
        
        # Define feature groups
        self.feature_groups = {
            'Temporal': ['Year_Numeric', 'Permit_Growth_1yr', 'Permit_Growth_2yr'],
            'Historical': ['Permit_Count_Lag1', 'Permit_Count_Lag2', 'Construction_Value_Lag1'],
            'Spatial_Lag': ['Spatial_Lag_Permits', 'Spatial_Lag_Value'],
            'Geographic': ['Centroid_Lat', 'Centroid_Lon', 'Distance_To_Downtown_km'],
            'Current': ['Permit_Count', 'Total_Construction_Value', 'Value_Per_Permit'],
            'Rolling': ['Permit_Count_Rolling_Mean', 'Permit_Count_Rolling_Std'],
            'Network_Centrality': [
                'Network_Degree', 'Degree_Centrality', 'Betweenness_Centrality',
                'Closeness_Centrality', 'Eigenvector_Centrality', 'PageRank',
                'Clustering_Coefficient'
            ],
            'Network_Structure': ['Community', 'Is_Hub', 'Path_To_Downtown']
        }
    
    def prepare_data(self, feature_groups_to_include):
        """Prepare data with selected feature groups"""
        # Collect features
        features = []
        for group in feature_groups_to_include:
            features.extend(self.feature_groups[group])
        
        # Filter to available features
        available_features = [f for f in features if f in self.train_df.columns]
        
        # Prepare train/test
        X_train = self.train_df[available_features].fillna(0)
        y_train = self.train_df['Permit_Growth'].fillna(0)
        X_test = self.test_df[available_features].fillna(0)
        y_test = self.test_df['Permit_Growth'].fillna(0)
        
        # Standardize
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        return X_train_scaled, X_test_scaled, y_train, y_test, available_features
    
    def run_ablation(self):
        """Run ablation study"""
        logger.info("="*60)
        logger.info("NETWORK FEATURE ABLATION STUDY")
        logger.info("="*60)
        
        ablation_configs = [
            {
                'name': 'Baseline (no network)',
                'groups': ['Temporal', 'Historical', 'Spatial_Lag', 'Geographic', 'Current', 'Rolling']
            },
            {
                'name': 'With Network Centrality',
                'groups': ['Temporal', 'Historical', 'Spatial_Lag', 'Geographic', 'Current', 'Rolling', 'Network_Centrality']
            },
            {
                'name': 'With Network Structure',
                'groups': ['Temporal', 'Historical', 'Spatial_Lag', 'Geographic', 'Current', 'Rolling', 'Network_Structure']
            },
            {
                'name': 'With All Network Features',
                'groups': ['Temporal', 'Historical', 'Spatial_Lag', 'Geographic', 'Current', 'Rolling', 'Network_Centrality', 'Network_Structure']
            },
            {
                'name': 'Network Features Only',
                'groups': ['Network_Centrality', 'Network_Structure']
            }
        ]
        
        results = []
        
        for config in ablation_configs:
            logger.info(f"\n{config['name']}")
            logger.info("-"*60)
            
            X_train, X_test, y_train, y_test, features = self.prepare_data(config['groups'])
            
            logger.info(f"Features: {len(features)}")
            
            # Train LASSO
            model = Lasso(alpha=2.12, random_state=42, max_iter=10000)
            model.fit(X_train, y_train)
            
            # Predict
            y_pred = model.predict(X_test)
            
            # Metrics
            rmse = np.sqrt(mean_squared_error(y_test, y_pred))
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)
            
            logger.info(f"  RMSE: {rmse:.2f}")
            logger.info(f"  MAE:  {mae:.2f}")
            logger.info(f"  R²:   {r2:.4f}")
            
            results.append({
                'config': config['name'],
                'num_features': len(features),
                'rmse': float(rmse),
                'mae': float(mae),
                'r2': float(r2)
            })
        
        # Save results
        self.results['ablation_study'] = results
        
        # Calculate improvements
        baseline_rmse = results[0]['rmse']
        for i, result in enumerate(results):
            if i > 0:
                improvement = ((baseline_rmse - result['rmse']) / baseline_rmse) * 100
                logger.info(f"\n{result['config']} vs Baseline: {improvement:+.2f}% RMSE change")
        
        return self
    
    def analyze_network_feature_importance(self):
        """Analyze importance of individual network features"""
        logger.info("\n" + "="*60)
        logger.info("NETWORK FEATURE IMPORTANCE ANALYSIS")
        logger.info("="*60)
        
        # Train with all features
        X_train, X_test, y_train, y_test, features = self.prepare_data(
            ['Temporal', 'Historical', 'Spatial_Lag', 'Geographic', 'Current', 'Rolling', 
             'Network_Centrality', 'Network_Structure']
        )
        
        model = Lasso(alpha=2.12, random_state=42, max_iter=10000)
        model.fit(X_train, y_train)
        
        # Get coefficients
        coef_dict = {feat: coef for feat, coef in zip(features, model.coef_)}
        
        # Filter network features
        network_features = self.feature_groups['Network_Centrality'] + self.feature_groups['Network_Structure']
        network_coefs = {k: v for k, v in coef_dict.items() if k in network_features}
        
        # Sort by absolute value
        sorted_coefs = sorted(network_coefs.items(), key=lambda x: abs(x[1]), reverse=True)
        
        logger.info("\nNetwork Feature Coefficients (LASSO):")
        for feat, coef in sorted_coefs:
            logger.info(f"  {feat:30s}: {coef:8.4f}")
        
        self.results['network_feature_coefficients'] = network_coefs
        
        return self
    
    def save_results(self):
        """Save results"""
        output_path = Path('results/network_analysis/network_feature_ablation.json')
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nResults saved to: {output_path}")
        return self
    
    def run(self):
        """Run complete analysis"""
        (self
            .run_ablation()
            .analyze_network_feature_importance()
            .save_results())
        
        logger.info("\n" + "="*60)
        logger.info("ABLATION STUDY COMPLETE")
        logger.info("="*60)
        
        return self


if __name__ == "__main__":
    ablation = NetworkFeatureAblation()
    ablation.run()