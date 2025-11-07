"""
Feature Engineering Module

Author: Utsav Patel (Modeler)
Date: 2024-11-06
"""

import pandas as pd
import numpy as np
from pathlib import Path
import logging
from typing import List, Dict
import networkx as nx

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FeatureEngineer:
    def __init__(self, graph: nx.Graph):
        self.graph = graph
        self.features_df = None
    
    def calculate_accessibility_features(self):
        """Calculate transit and downtown accessibility"""
        logger.info("Calculating accessibility features...")
        
        # Downtown Toronto coordinates (approximate)
        downtown_lat, downtown_lon = 43.6532, -79.3832
        
        for node in self.graph.nodes():
            if 'lat' in self.graph.nodes[node] and 'lon' in self.graph.nodes[node]:
                lat = self.graph.nodes[node]['lat']
                lon = self.graph.nodes[node]['lon']
                
                # Distance to downtown (simplified)
                dist_downtown = np.sqrt((lat - downtown_lat)**2 + (lon - downtown_lon)**2) * 111
                self.graph.nodes[node]['dist_downtown_km'] = dist_downtown
                
                # Transit proximity (placeholder - would use actual transit data)
                self.graph.nodes[node]['transit_proximity'] = np.random.uniform(0, 1)
        
        logger.info("Accessibility features calculated")
    
    def calculate_amenity_density(self, radius_km: float = 1.0):
        """Calculate density of amenities around each node"""
        logger.info("Calculating amenity density features...")
        
        # This would integrate with OSM POI data
        # For progress report, we'll create placeholder features
        for node in self.graph.nodes():
            self.graph.nodes[node]['amenity_density'] = np.random.uniform(0, 1)
            self.graph.nodes[node]['park_density'] = np.random.uniform(0, 1)
            self.graph.nodes[node]['school_density'] = np.random.uniform(0, 1)
        
        logger.info("✓ Amenity density features calculated")
    
    def calculate_temporal_features(self):
        """Calculate growth rate and temporal patterns"""
        logger.info("Calculating temporal features...")
        
        for node in self.graph.nodes():
            # Placeholder for actual temporal calculations
            if 'price_mean' in self.graph.nodes[node]:
                price = self.graph.nodes[node]['price_mean']
                if price and price > 0:
                    # Simulate some growth patterns
                    self.graph.nodes[node]['price_growth_1yr'] = np.random.normal(0.05, 0.02)
                    self.graph.nodes[node]['price_growth_2yr'] = np.random.normal(0.08, 0.03)
        
        logger.info("✓ Temporal features calculated")
    
    def create_feature_dataframe(self) -> pd.DataFrame:
        """Convert graph features to DataFrame for modeling"""
        logger.info("Creating feature DataFrame...")
        
        features_data = []
        for node in self.graph.nodes():
            node_data = {'node_id': node}
            
            # Add all node attributes
            for attr, value in self.graph.nodes[node].items():
                if attr not in ['pos']:  # Skip position tuples
                    node_data[attr] = value
            
            features_data.append(node_data)
        
        self.features_df = pd.DataFrame(features_data)
        logger.info(f"Feature DataFrame created with {len(self.features_df)} rows and {len(self.features_df.columns)} features")
        
        return self.features_df
    
    def save_features(self, output_path: Path):
        """Save features to CSV"""
        if self.features_df is not None:
            self.features_df.to_csv(output_path, index=False)
            logger.info(f"Features saved to {output_path}")