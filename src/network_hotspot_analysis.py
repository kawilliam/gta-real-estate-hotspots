"""
Network-Hotspot Relationship Analysis

Analyzes the relationship between network structure and real estate hotspots.

Key Questions:
1. Do high-centrality nodes have higher development growth?
2. Do hubs outperform peripheral nodes?
3. Are certain communities hotspot clusters?

Author: Kyle Williamson + Yadon Kassahun + Hari Patel
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from pathlib import Path
import logging
import json
from scipy import stats

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkHotspotAnalyzer:
    """Analyzes relationship between network structure and hotspots"""
    
    def __init__(self, 
                 data_path: str = 'data/processed/train_set.csv',
                 network_path: str = 'data/processed/networks/spatial_network_enriched.gpickle',
                 output_dir: str = 'results/network_analysis'):
        self.data_path = Path(data_path)
        self.network_path = Path(network_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.df = None
        self.graph = None
        self.results = {}
    
    def load_data(self):
        """Load training data and network"""
        logger.info("="*60)
        logger.info("LOADING DATA")
        logger.info("="*60)
        
        logger.info(f"Loading training data from {self.data_path}")
        self.df = pd.read_csv(self.data_path)
        logger.info(f"  Loaded {len(self.df)} records")
        
        logger.info(f"Loading network from {self.network_path}")
        with open(self.network_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"  Loaded network: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        
        return self
    
    def correlation_analysis(self):
        """Analyze correlation between centrality and growth"""
        logger.info("\n" + "="*60)
        logger.info("CORRELATION ANALYSIS: Centrality vs Permit Growth")
        logger.info("="*60)
        
        # Network features
        network_features = [
            'Degree_Centrality',
            'Betweenness_Centrality',
            'Closeness_Centrality',
            'Eigenvector_Centrality',
            'PageRank',
            'Clustering_Coefficient'
        ]
        
        # Calculate correlations
        correlations = {}
        p_values = {}
        
        for feature in network_features:
            if feature in self.df.columns:
                # Remove NaN values
                valid_data = self.df[[feature, 'Permit_Growth']].dropna()
                
                if len(valid_data) > 0:
                    corr, p_val = stats.pearsonr(
                        valid_data[feature], 
                        valid_data['Permit_Growth']
                    )
                    correlations[feature] = corr
                    p_values[feature] = p_val
                    
                    sig_marker = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
                    logger.info(f"{feature:30s}: r={corr:6.3f}, p={p_val:.4f} {sig_marker}")
        
        self.results['correlations'] = correlations
        self.results['correlation_p_values'] = p_values
        
        # Create correlation plot
        self._plot_correlations(correlations, p_values)
        
        return self
    
    def hub_vs_periphery_analysis(self):
        """Compare growth between hubs and peripheral nodes"""
        logger.info("\n" + "="*60)
        logger.info("HUB VS PERIPHERY ANALYSIS")
        logger.info("="*60)
        
        if 'Is_Hub' not in self.df.columns:
            logger.warning("Is_Hub column not found. Skipping analysis.")
            return self
        
        # Group by hub status
        hub_growth = self.df[self.df['Is_Hub'] == 1]['Permit_Growth'].dropna()
        periphery_growth = self.df[self.df['Is_Hub'] == 0]['Permit_Growth'].dropna()
        
        logger.info(f"Hub nodes: n={len(hub_growth)}")
        logger.info(f"  Mean growth: {hub_growth.mean():.2f} permits/year")
        logger.info(f"  Median growth: {hub_growth.median():.2f}")
        logger.info(f"  Std dev: {hub_growth.std():.2f}")
        
        logger.info(f"Periphery nodes: n={len(periphery_growth)}")
        logger.info(f"  Mean growth: {periphery_growth.mean():.2f} permits/year")
        logger.info(f"  Median growth: {periphery_growth.median():.2f}")
        logger.info(f"  Std dev: {periphery_growth.std():.2f}")
        
        # Statistical test
        if len(hub_growth) > 0 and len(periphery_growth) > 0:
            t_stat, p_val = stats.ttest_ind(hub_growth, periphery_growth)
            logger.info(f"\nT-test: t={t_stat:.3f}, p={p_val:.4f}")
            
            if p_val < 0.05:
                logger.info("✓ SIGNIFICANT difference between hubs and periphery!")
            else:
                logger.info("✗ No significant difference between hubs and periphery")
            
            self.results['hub_vs_periphery'] = {
                'hub_mean': float(hub_growth.mean()),
                'periphery_mean': float(periphery_growth.mean()),
                't_statistic': float(t_stat),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
        
        # Create boxplot
        self._plot_hub_periphery_comparison(hub_growth, periphery_growth)
        
        return self
    
    def community_analysis(self):
        """Analyze growth patterns across communities"""
        logger.info("\n" + "="*60)
        logger.info("COMMUNITY ANALYSIS")
        logger.info("="*60)
        
        if 'Community' not in self.df.columns:
            logger.warning("Community column not found. Skipping analysis.")
            return self
        
        # Group by community
        community_stats = self.df.groupby('Community')['Permit_Growth'].agg([
            'count', 'mean', 'median', 'std'
        ]).sort_values('mean', ascending=False)
        
        logger.info("Growth by community:")
        logger.info(community_stats.to_string())
        
        # ANOVA test
        communities = []
        for comm_id in self.df['Community'].unique():
            if comm_id >= 0:  # Skip invalid communities
                comm_data = self.df[self.df['Community'] == comm_id]['Permit_Growth'].dropna()
                if len(comm_data) > 0:
                    communities.append(comm_data)
        
        if len(communities) > 2:
            f_stat, p_val = stats.f_oneway(*communities)
            logger.info(f"\nANOVA test: F={f_stat:.3f}, p={p_val:.4f}")
            
            if p_val < 0.05:
                logger.info("✓ SIGNIFICANT differences between communities!")
            else:
                logger.info("✗ No significant differences between communities")
            
            self.results['community_anova'] = {
                'f_statistic': float(f_stat),
                'p_value': float(p_val),
                'significant': bool(p_val < 0.05)
            }
        
        # Create community comparison plot
        self._plot_community_comparison()
        
        return self
    
    def centrality_growth_scatterplots(self):
        """Create scatterplots showing centrality vs growth"""
        logger.info("\n" + "="*60)
        logger.info("GENERATING CENTRALITY-GROWTH SCATTERPLOTS")
        logger.info("="*60)
        
        centrality_features = [
            ('Degree_Centrality', 'Degree Centrality'),
            ('Betweenness_Centrality', 'Betweenness Centrality'),
            ('PageRank', 'PageRank')
        ]
        
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        
        for idx, (feature, title) in enumerate(centrality_features):
            if feature in self.df.columns:
                valid_data = self.df[[feature, 'Permit_Growth']].dropna()
                
                axes[idx].scatter(
                    valid_data[feature], 
                    valid_data['Permit_Growth'],
                    alpha=0.5,
                    s=50
                )
                
                # Add regression line
                z = np.polyfit(valid_data[feature], valid_data['Permit_Growth'], 1)
                p = np.poly1d(z)
                x_line = np.linspace(valid_data[feature].min(), 
                                    valid_data[feature].max(), 100)
                axes[idx].plot(x_line, p(x_line), "r--", linewidth=2)
                
                # Calculate R²
                corr = valid_data[feature].corr(valid_data['Permit_Growth'])
                r_squared = corr ** 2
                
                axes[idx].set_xlabel(title, fontsize=11)
                axes[idx].set_ylabel('Permit Growth' if idx == 0 else '', fontsize=11)
                axes[idx].set_title(f'{title}\nR² = {r_squared:.3f}', fontsize=12)
                axes[idx].grid(True, alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'centrality_growth_scatterplots.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
        
        return self
    
    def _plot_correlations(self, correlations, p_values):
        """Create correlation bar plot"""
        fig, ax = plt.subplots(figsize=(10, 6))
        
        features = list(correlations.keys())
        corr_values = list(correlations.values())
        colors = ['green' if p_values[f] < 0.05 else 'gray' for f in features]
        
        bars = ax.barh(features, corr_values, color=colors, alpha=0.7)
        ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
        ax.set_xlabel('Correlation with Permit Growth', fontsize=12)
        ax.set_title('Network Centrality vs Development Growth\n(Green = Significant p<0.05)', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add correlation values as text
        for i, (feat, val) in enumerate(zip(features, corr_values)):
            ax.text(val + 0.01 if val > 0 else val - 0.01, i, f'{val:.3f}',
                   va='center', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'centrality_correlations.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def _plot_hub_periphery_comparison(self, hub_growth, periphery_growth):
        """Create hub vs periphery boxplot"""
        fig, ax = plt.subplots(figsize=(8, 6))
        
        data_to_plot = [hub_growth, periphery_growth]
        labels = ['Hub Nodes', 'Periphery Nodes']
        
        bp = ax.boxplot(data_to_plot, labels=labels, patch_artist=True)
        
        # Color boxes
        colors = ['#ff6b6b', '#4ecdc4']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)
        
        ax.set_ylabel('Permit Growth (permits/year)', fontsize=12)
        ax.set_title('Development Growth: Hub vs Periphery Nodes', 
                    fontsize=14, fontweight='bold')
        ax.grid(axis='y', alpha=0.3)
        
        # Add sample sizes
        ax.text(1, ax.get_ylim()[1]*0.95, f'n={len(hub_growth)}',
               ha='center', fontsize=10)
        ax.text(2, ax.get_ylim()[1]*0.95, f'n={len(periphery_growth)}',
               ha='center', fontsize=10)
        
        plt.tight_layout()
        output_path = self.output_dir / 'hub_vs_periphery_boxplot.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def _plot_community_comparison(self):
        """Create community comparison plot"""
        if 'Community' not in self.df.columns:
            return
        
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Filter valid communities
        valid_df = self.df[self.df['Community'] >= 0].copy()
        
        # Boxplot by community
        valid_df.boxplot(column='Permit_Growth', by='Community', ax=ax)
        
        ax.set_xlabel('Community ID', fontsize=12)
        ax.set_ylabel('Permit Growth (permits/year)', fontsize=12)
        ax.set_title('Development Growth by Network Community', 
                    fontsize=14, fontweight='bold')
        plt.suptitle('')  # Remove default title
        ax.grid(axis='y', alpha=0.3)
        
        plt.tight_layout()
        output_path = self.output_dir / 'community_growth_comparison.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved: {output_path}")
    
    def save_results(self):
        """Save analysis results to JSON"""
        output_path = self.output_dir / 'network_hotspot_analysis.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.results, f, indent=2)
        
        logger.info(f"\nSaved results to: {output_path}")
        return self
    
    def run_full_analysis(self):
        """Run complete network-hotspot analysis"""
        logger.info("="*60)
        logger.info("NETWORK-HOTSPOT RELATIONSHIP ANALYSIS")
        logger.info("="*60)
        
        (self
            .load_data()
            .correlation_analysis()
            .hub_vs_periphery_analysis()
            .community_analysis()
            .centrality_growth_scatterplots()
            .save_results())
        
        logger.info("\n" + "="*60)
        logger.info("ANALYSIS COMPLETE")
        logger.info("="*60)
        logger.info(f"Results saved to: {self.output_dir}")
        
        return self


def main():
    """Main execution"""
    analyzer = NetworkHotspotAnalyzer()
    analyzer.run_full_analysis()
    
    print("\n✓ Network-hotspot analysis complete!")
    print("  Check results/network_analysis/ for visualizations and JSON results")


if __name__ == "__main__":
    main()