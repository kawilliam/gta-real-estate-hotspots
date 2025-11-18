"""
Complete Data Pipeline and Feature Engineering

This script handles:
1. Data loading and cleaning
2. Temporal split creation (2018-2021 train, 2022 val, 2023 test)
3. Feature engineering (temporal, spatial, development)
4. Target variable creation
5. Dataset saving and reporting

Author: Kyle Williamson (Data Engineer)
Date: 2024-11-25
"""

import pandas as pd
import numpy as np
import pickle
import networkx as nx
from pathlib import Path
import logging
from typing import Dict, Tuple
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class GTA_HotspotsDataPipeline:
    """Complete data pipeline for GTA Real Estate Hotspots project"""
    
    def __init__(
        self,
        permits_path: str,
        network_path: str,
        output_dir: str = 'data/processed'
    ):
        self.permits_path = Path(permits_path)
        self.network_path = Path(network_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.network = None
        self.train_df = None
        self.val_df = None
        self.test_df = None
        
        # Downtown Toronto coordinates (approximate city center)
        self.downtown_lat = 43.6532
        self.downtown_lon = -79.3832
    
    def load_data(self) -> 'GTA_HotspotsDataPipeline':
        """Load building permits data and spatial network"""
        logger.info("="*60)
        logger.info("STEP 1: LOADING DATA")
        logger.info("="*60)
        
        # Load building permits
        logger.info(f"Loading building permits from {self.permits_path}")
        self.df = pd.read_csv(self.permits_path)
        logger.info(f"  Loaded {len(self.df)} records")
        logger.info(f"  Columns: {list(self.df.columns)}")
        
        # Load spatial network
        logger.info(f"Loading spatial network from {self.network_path}")
        with open(self.network_path, 'rb') as f:
            self.network = pickle.load(f)
        logger.info(f"  Loaded network: {self.network.number_of_nodes()} nodes, "
                   f"{self.network.number_of_edges()} edges")
        
        return self
    
    def clean_data(self) -> 'GTA_HotspotsDataPipeline':
        """Clean and filter data to valid Toronto FSAs"""
        logger.info("\n" + "="*60)
        logger.info("STEP 2: CLEANING DATA")
        logger.info("="*60)
        
        initial_count = len(self.df)
        
        # Remove records with missing or empty FSA
        self.df = self.df[self.df['FSA'].notna()].copy()
        self.df = self.df[self.df['FSA'].str.strip() != ''].copy()
        logger.info(f"Removed {initial_count - len(self.df)} records with missing FSA")
        
        # Filter to valid Toronto FSAs (M + digit + letter pattern)
        self.df = self.df[
            self.df['FSA'].str.match(r'^M\d[A-Z]$', na=False)
        ].copy()
        logger.info(f"Filtered to {len(self.df)} records with valid Toronto FSAs")
        logger.info(f"Unique FSAs: {self.df['FSA'].nunique()}")
        
        # Ensure numeric columns are proper type
        self.df['Year'] = pd.to_numeric(self.df['Year'], errors='coerce')
        self.df['Permit_Count'] = pd.to_numeric(self.df['Permit_Count'], errors='coerce')
        self.df['Total_Construction_Value'] = pd.to_numeric(
            self.df['Total_Construction_Value'], errors='coerce'
        )
        
        # Remove records with missing key values
        self.df = self.df[
            self.df['Year'].notna() & 
            self.df['Permit_Count'].notna()
        ].copy()
        
        # Sort by FSA and Year for time series operations
        self.df = self.df.sort_values(['FSA', 'Year']).reset_index(drop=True)
        
        logger.info(f"Final cleaned dataset: {len(self.df)} records")
        logger.info(f"Year range: {self.df['Year'].min():.0f} - {self.df['Year'].max():.0f}")
        
        return self
    
    def create_target_variables(self) -> 'GTA_HotspotsDataPipeline':
        """Create target variables for next-year permit growth prediction"""
        logger.info("\n" + "="*60)
        logger.info("STEP 3: CREATING TARGET VARIABLES")
        logger.info("="*60)
        
        # Calculate next-year permit count (target)
        self.df['Permit_Count_Next_Year'] = self.df.groupby('FSA')['Permit_Count'].shift(-1)
        
        # Primary target: Absolute growth in permit count
        self.df['Permit_Growth'] = (
            self.df['Permit_Count_Next_Year'] - self.df['Permit_Count']
        )
        
        # Alternative target: Percentage growth
        self.df['Permit_Growth_Pct'] = (
            (self.df['Permit_Count_Next_Year'] - self.df['Permit_Count']) / 
            (self.df['Permit_Count'] + 1)
        ) * 100
        
        # Binary hotspot classification (top 20% growth)
        valid_growth = self.df['Permit_Growth'].dropna()
        if len(valid_growth) > 0:
            growth_threshold = valid_growth.quantile(0.80)
            self.df['Is_Hotspot'] = (
                self.df['Permit_Growth'] >= growth_threshold
            ).astype(int)
        else:
            self.df['Is_Hotspot'] = 0
        
        # Log statistics
        logger.info("Target variable statistics:")
        logger.info(f"  Permit_Growth:")
        logger.info(f"    Mean: {self.df['Permit_Growth'].mean():.2f}")
        logger.info(f"    Std: {self.df['Permit_Growth'].std():.2f}")
        logger.info(f"    Min: {self.df['Permit_Growth'].min():.2f}")
        logger.info(f"    Max: {self.df['Permit_Growth'].max():.2f}")
        logger.info(f"  Is_Hotspot: {self.df['Is_Hotspot'].sum()} hotspots "
                   f"({self.df['Is_Hotspot'].mean()*100:.1f}%)")
        
        return self
    
    def engineer_temporal_features(self) -> 'GTA_HotspotsDataPipeline':
        """Create temporal lag and growth rate features"""
        logger.info("\n" + "="*60)
        logger.info("STEP 4: ENGINEERING TEMPORAL FEATURES")
        logger.info("="*60)
        
        # Lag features (t-1, t-2)
        self.df['Permit_Count_Lag1'] = self.df.groupby('FSA')['Permit_Count'].shift(1)
        self.df['Permit_Count_Lag2'] = self.df.groupby('FSA')['Permit_Count'].shift(2)
        
        # Historical growth rates
        self.df['Permit_Growth_1yr'] = (
            self.df['Permit_Count'] - self.df['Permit_Count_Lag1']
        )
        self.df['Permit_Growth_2yr'] = (
            self.df['Permit_Count'] - self.df['Permit_Count_Lag2']
        )
        
        # Construction value lag features
        self.df['Construction_Value_Lag1'] = self.df.groupby('FSA')[
            'Total_Construction_Value'
        ].shift(1)
        
        # Year as numeric feature
        self.df['Year_Numeric'] = self.df['Year']
        
        logger.info("Created temporal features:")
        logger.info("  - Permit_Count_Lag1, Permit_Count_Lag2")
        logger.info("  - Permit_Growth_1yr, Permit_Growth_2yr")
        logger.info("  - Construction_Value_Lag1")
        logger.info("  - Year_Numeric")
        
        return self
    
    def engineer_development_features(self) -> 'GTA_HotspotsDataPipeline':
        """Create development activity features"""
        logger.info("\n" + "="*60)
        logger.info("STEP 5: ENGINEERING DEVELOPMENT FEATURES")
        logger.info("="*60)
        
        # Construction value per permit (intensity metric)
        self.df['Value_Per_Permit'] = (
            self.df['Total_Construction_Value'] / (self.df['Permit_Count'] + 1)
        )
        
        # Rolling average features (3-year window)
        self.df['Permit_Count_Rolling_Mean'] = self.df.groupby('FSA')[
            'Permit_Count'
        ].transform(lambda x: x.rolling(window=3, min_periods=1).mean())
        
        self.df['Permit_Count_Rolling_Std'] = self.df.groupby('FSA')[
            'Permit_Count'
        ].transform(lambda x: x.rolling(window=3, min_periods=1).std())
        
        logger.info("Created development features:")
        logger.info("  - Value_Per_Permit")
        logger.info("  - Permit_Count_Rolling_Mean")
        logger.info("  - Permit_Count_Rolling_Std")
        
        return self
    
    def engineer_spatial_features(self) -> 'GTA_HotspotsDataPipeline':
        """Create spatial features using network and geography"""
        logger.info("\n" + "="*60)
        logger.info("STEP 6: ENGINEERING SPATIAL FEATURES")
        logger.info("="*60)
        
        # Calculate distance to downtown for each FSA
        def calc_distance_to_downtown(row):
            if pd.notna(row['Centroid_Lat']) and pd.notna(row['Centroid_Lon']):
                lat_diff = row['Centroid_Lat'] - self.downtown_lat
                lon_diff = row['Centroid_Lon'] - self.downtown_lon
                return np.sqrt(lat_diff**2 + lon_diff**2) * 111
            return np.nan
        
        self.df['Distance_To_Downtown_km'] = self.df.apply(
            calc_distance_to_downtown, axis=1
        )
        
        # Extract spatial lag features from network (if available)
        # Note: Network has these as node attributes, we need to join them
        spatial_lag_dict = {}
        for node in self.network.nodes():
            if node.strip():  # Skip empty FSA codes
                node_data = self.network.nodes[node]
                if 'Permit_Count_spatial_lag' in node_data:
                    spatial_lag_dict[node] = {
                        'Spatial_Lag_Permits': node_data['Permit_Count_spatial_lag'],
                        'Spatial_Lag_Value': node_data.get(
                            'Total_Construction_Value_spatial_lag', np.nan
                        )
                    }
        
        # Join spatial lag features
        spatial_lag_df = pd.DataFrame.from_dict(spatial_lag_dict, orient='index')
        spatial_lag_df.index.name = 'FSA'
        spatial_lag_df = spatial_lag_df.reset_index()
        
        self.df = self.df.merge(spatial_lag_df, on='FSA', how='left')
        
        # Calculate node degree (connectivity) from network
        degree_dict = dict(self.network.degree())
        self.df['Network_Degree'] = self.df['FSA'].map(degree_dict)
        
        logger.info("Created spatial features:")
        logger.info("  - Distance_To_Downtown_km")
        logger.info("  - Spatial_Lag_Permits (from network)")
        logger.info("  - Spatial_Lag_Value (from network)")
        logger.info("  - Network_Degree (connectivity)")
        
        return self
    
    def create_temporal_splits(self) -> 'GTA_HotspotsDataPipeline':
        """Split data into train (2018-2021), validation (2022), test (2023)"""
        logger.info("\n" + "="*60)
        logger.info("STEP 7: CREATING TEMPORAL SPLITS")
        logger.info("="*60)
        
        # Filter to years of interest
        self.df = self.df[self.df['Year'].between(2018, 2023)].copy()
        
        # Create splits
        self.train_df = self.df[self.df['Year'].between(2018, 2021)].copy()
        self.val_df = self.df[self.df['Year'] == 2022].copy()
        self.test_df = self.df[self.df['Year'] == 2023].copy()
        
        # Remove records where target is missing
        # (Last year for each FSA won't have next year data)
        self.train_df = self.train_df[self.train_df['Permit_Growth'].notna()].copy()
        self.val_df = self.val_df[self.val_df['Permit_Growth'].notna()].copy()
        self.test_df = self.test_df[self.test_df['Permit_Growth'].notna()].copy()
        
        logger.info(f"Train set (2018-2021):")
        logger.info(f"  Records: {len(self.train_df)}")
        logger.info(f"  Unique FSAs: {self.train_df['FSA'].nunique()}")
        logger.info(f"  Years: {sorted(self.train_df['Year'].unique())}")
        
        logger.info(f"Validation set (2022):")
        logger.info(f"  Records: {len(self.val_df)}")
        logger.info(f"  Unique FSAs: {self.val_df['FSA'].nunique()}")
        
        logger.info(f"Test set (2023):")
        logger.info(f"  Records: {len(self.test_df)}")
        logger.info(f"  Unique FSAs: {self.test_df['FSA'].nunique()}")
        
        return self
    
    def save_datasets(self) -> 'GTA_HotspotsDataPipeline':
        """Save processed datasets and metadata"""
        logger.info("\n" + "="*60)
        logger.info("STEP 8: SAVING PROCESSED DATASETS")
        logger.info("="*60)
        
        # Save full processed dataset
        full_path = self.output_dir / 'full_processed_data.csv'
        self.df.to_csv(full_path, index=False)
        logger.info(f"Saved full dataset: {full_path}")
        
        # Save temporal splits
        train_path = self.output_dir / 'train_set.csv'
        val_path = self.output_dir / 'val_set.csv'
        test_path = self.output_dir / 'test_set.csv'
        
        self.train_df.to_csv(train_path, index=False)
        self.val_df.to_csv(val_path, index=False)
        self.test_df.to_csv(test_path, index=False)
        
        logger.info(f"Saved train set: {train_path}")
        logger.info(f"Saved validation set: {val_path}")
        logger.info(f"Saved test set: {test_path}")
        
        # Save feature list
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['FSA', 'Year', 'Permit_Growth', 
                                     'Permit_Count_Next_Year', 'Permit_Growth_Pct',
                                     'Is_Hotspot']]
        
        feature_metadata = {
            'total_features': len(feature_cols),
            'feature_columns': feature_cols,
            'target_variable': 'Permit_Growth',
            'alternative_targets': ['Permit_Growth_Pct', 'Is_Hotspot'],
            'train_records': len(self.train_df),
            'val_records': len(self.val_df),
            'test_records': len(self.test_df),
            'unique_fsas': self.df['FSA'].nunique()
        }
        
        metadata_path = self.output_dir / 'feature_metadata.json'
        with open(metadata_path, 'w') as f:
            json.dump(feature_metadata, f, indent=2)
        logger.info(f"Saved feature metadata: {metadata_path}")
        
        return self
    
    def generate_summary_report(self) -> 'GTA_HotspotsDataPipeline':
        """Generate comprehensive summary report"""
        logger.info("\n" + "="*60)
        logger.info("DATA PIPELINE SUMMARY REPORT")
        logger.info("="*60)
        
        logger.info("\n--- DATASET OVERVIEW ---")
        logger.info(f"Total records (2018-2023): {len(self.df)}")
        logger.info(f"Unique FSAs: {self.df['FSA'].nunique()}")
        logger.info(f"Year range: {self.df['Year'].min():.0f}-{self.df['Year'].max():.0f}")
        
        logger.info("\n--- TARGET VARIABLE (Permit_Growth) ---")
        logger.info(f"Mean: {self.df['Permit_Growth'].mean():.2f} permits/year")
        logger.info(f"Std Dev: {self.df['Permit_Growth'].std():.2f}")
        logger.info(f"Median: {self.df['Permit_Growth'].median():.2f}")
        logger.info(f"Min: {self.df['Permit_Growth'].min():.2f}")
        logger.info(f"Max: {self.df['Permit_Growth'].max():.2f}")
        
        logger.info("\n--- TEMPORAL SPLITS ---")
        logger.info(f"Train (2018-2021): {len(self.train_df)} records")
        logger.info(f"  FSAs: {self.train_df['FSA'].nunique()}")
        logger.info(f"  Avg permits/FSA: {self.train_df['Permit_Count'].mean():.1f}")
        
        logger.info(f"Validation (2022): {len(self.val_df)} records")
        logger.info(f"  FSAs: {self.val_df['FSA'].nunique()}")
        logger.info(f"  Avg permits/FSA: {self.val_df['Permit_Count'].mean():.1f}")
        
        logger.info(f"Test (2023): {len(self.test_df)} records")
        logger.info(f"  FSAs: {self.test_df['FSA'].nunique()}")
        logger.info(f"  Avg permits/FSA: {self.test_df['Permit_Count'].mean():.1f}")
        
        logger.info("\n--- FEATURE SUMMARY ---")
        feature_cols = [col for col in self.train_df.columns 
                       if col not in ['FSA', 'Year', 'Permit_Growth',
                                     'Permit_Count_Next_Year', 'Permit_Growth_Pct',
                                     'Is_Hotspot']]
        logger.info(f"Total features: {len(feature_cols)}")
        logger.info("Feature categories:")
        logger.info("  - Temporal: lag features, historical growth")
        logger.info("  - Development: permit counts, construction value")
        logger.info("  - Spatial: distance to downtown, spatial lags, network degree")
        
        logger.info("\n--- DATA QUALITY ---")
        missing_counts = self.train_df[feature_cols].isnull().sum()
        if missing_counts.sum() > 0:
            logger.info("Missing values in training set:")
            for col in missing_counts[missing_counts > 0].index:
                pct = missing_counts[col] / len(self.train_df) * 100
                logger.info(f"  {col}: {missing_counts[col]} ({pct:.1f}%)")
        else:
            logger.info("No missing values in feature columns!")
        
        logger.info("\n" + "="*60)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        
        return self
    
    def run_pipeline(self) -> 'GTA_HotspotsDataPipeline':
        """Execute complete pipeline"""
        (self
            .load_data()
            .clean_data()
            .create_target_variables()
            .engineer_temporal_features()
            .engineer_development_features()
            .engineer_spatial_features()
            .create_temporal_splits()
            .save_datasets()
            .generate_summary_report())
        
        return self


def main():
    """Main execution function"""
    
    # Configure paths - UPDATE THESE to your actual paths
    PERMITS_PATH = "data/processed/fsa_aggregated/building_permits_fsa.csv"
    NETWORK_PATH = "data/processed/networks/spatial_network_distance.gpickle"
    OUTPUT_DIR = "data/processed"
    
    logger.info("="*60)
    logger.info("GTA REAL ESTATE HOTSPOTS - DATA PIPELINE")
    logger.info("="*60)
    logger.info(f"Permits data: {PERMITS_PATH}")
    logger.info(f"Network file: {NETWORK_PATH}")
    logger.info(f"Output directory: {OUTPUT_DIR}")
    logger.info("="*60 + "\n")
    
    try:
        pipeline = GTA_HotspotsDataPipeline(
            permits_path=PERMITS_PATH,
            network_path=NETWORK_PATH,
            output_dir=OUTPUT_DIR
        )
        
        pipeline.run_pipeline()
        
        logger.info("\n✓ All processing completed successfully!")
        logger.info(f"✓ Processed datasets saved to: {OUTPUT_DIR}")
        logger.info("\nNext steps:")
        logger.info("  1. Review processed data in data/processed/")
        logger.info("  2. Run model training: python step2_baseline_models.py")
        logger.info("  3. Check feature_metadata.json for feature list")
        
    except Exception as e:
        logger.error(f"\n✗ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()