"""
Advanced Network Structure Analysis

This module performs deep network analysis including:
- Community detection
- Hub/periphery identification
- Path-based metrics
- Network role classification

Author: Kyle Williamson (Data Engineer) + Yadon Kassahun (Network Architect)
Date: 2025-11-27
"""

import pandas as pd
import numpy as np
import networkx as nx
import pickle
from pathlib import Path
import logging
import json
from collections import Counter

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class NetworkStructureAnalyzer:
    """Performs advanced network structure analysis"""
    
    def __init__(self, network_path: str):
        self.network_path = Path(network_path)
        self.graph = None
        self.analysis_results = {}
        
        # Downtown Toronto node (will find closest FSA)
        self.downtown_lat = 43.6532
        self.downtown_lon = -79.3832
        self.downtown_node = None
    
    def load_network(self):
        """Load the spatial network"""
        logger.info(f"Loading network from {self.network_path}")
        with open(self.network_path, 'rb') as f:
            self.graph = pickle.load(f)
        logger.info(f"Loaded network: {self.graph.number_of_nodes()} nodes, "
                   f"{self.graph.number_of_edges()} edges")
        return self
    
    def find_downtown_node(self):
        """Find the FSA node closest to downtown Toronto"""
        logger.info("Finding downtown node...")
        
        min_dist = float('inf')
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            if 'Centroid_Lat' in node_data and 'Centroid_Lon' in node_data:
                lat = node_data['Centroid_Lat']
                lon = node_data['Centroid_Lon']
                dist = np.sqrt((lat - self.downtown_lat)**2 + 
                              (lon - self.downtown_lon)**2)
                if dist < min_dist:
                    min_dist = dist
                    self.downtown_node = node
        
        logger.info(f"Downtown node identified: {self.downtown_node}")
        return self
    
    def detect_communities(self):
        """Detect communities using Louvain algorithm"""
        logger.info("Detecting network communities...")
        
        # Convert to undirected for community detection
        G_undirected = self.graph.to_undirected()
        
        # Louvain community detection
        try:
            import community as community_louvain
            communities = community_louvain.best_partition(G_undirected)
        except ImportError:
            # Fallback to greedy modularity if python-louvain not installed
            logger.warning("python-louvain not found, using greedy modularity")
            from networkx.algorithms import community
            communities_sets = community.greedy_modularity_communities(G_undirected)
            communities = {}
            for idx, comm in enumerate(communities_sets):
                for node in comm:
                    communities[node] = idx
        
        # Add community labels to graph
        for node, comm_id in communities.items():
            self.graph.nodes[node]['community'] = comm_id
        
        num_communities = len(set(communities.values()))
        logger.info(f"Detected {num_communities} communities")
        
        # Log community sizes
        comm_sizes = Counter(communities.values())
        logger.info("Community sizes:")
        for comm_id, size in sorted(comm_sizes.items()):
            logger.info(f"  Community {comm_id}: {size} nodes")
        
        self.analysis_results['num_communities'] = num_communities
        self.analysis_results['community_sizes'] = dict(comm_sizes)
        
        return self
    
    def classify_hub_periphery(self):
        """Classify nodes as hubs or periphery based on centrality"""
        logger.info("Classifying nodes as hubs vs periphery...")
        
        # Get degree centrality values
        degree_cents = [self.graph.nodes[n].get('degree_centrality', 0) 
                       for n in self.graph.nodes()]
        betweenness_cents = [self.graph.nodes[n].get('betweenness_centrality', 0) 
                            for n in self.graph.nodes()]
        
        # Calculate thresholds (top 20% = hubs)
        degree_threshold = np.percentile(degree_cents, 80)
        betweenness_threshold = np.percentile(betweenness_cents, 80)
        
        # Classify nodes
        hub_count = 0
        for node in self.graph.nodes():
            degree_c = self.graph.nodes[node].get('degree_centrality', 0)
            between_c = self.graph.nodes[node].get('betweenness_centrality', 0)
            
            # Hub if high in either degree OR betweenness
            is_hub = (degree_c >= degree_threshold) or (between_c >= betweenness_threshold)
            
            self.graph.nodes[node]['is_hub'] = 1 if is_hub else 0
            self.graph.nodes[node]['node_type'] = 'hub' if is_hub else 'periphery'
            
            if is_hub:
                hub_count += 1
        
        periphery_count = self.graph.number_of_nodes() - hub_count
        logger.info(f"Classified nodes: {hub_count} hubs, {periphery_count} periphery")
        
        self.analysis_results['num_hubs'] = hub_count
        self.analysis_results['num_periphery'] = periphery_count
        
        return self
    
    def calculate_path_metrics(self):
        """Calculate shortest path distances to downtown"""
        logger.info("Calculating shortest path metrics to downtown...")
        
        if self.downtown_node is None:
            logger.error("Downtown node not identified. Run find_downtown_node() first.")
            return self
        
        # Calculate shortest path lengths from downtown to all nodes
        try:
            path_lengths = nx.single_source_shortest_path_length(
                self.graph, self.downtown_node
            )
            
            for node in self.graph.nodes():
                if node in path_lengths:
                    self.graph.nodes[node]['path_to_downtown'] = path_lengths[node]
                else:
                    # Node not reachable (disconnected component)
                    self.graph.nodes[node]['path_to_downtown'] = 999
            
            reachable = [v for v in path_lengths.values() if v < 999]
            logger.info(f"Path lengths to downtown:")
            logger.info(f"  Mean: {np.mean(reachable):.2f} hops")
            logger.info(f"  Max: {max(reachable)} hops")
            logger.info(f"  Unreachable nodes: {sum(1 for n in self.graph.nodes() if self.graph.nodes[n]['path_to_downtown'] == 999)}")
            
        except nx.NetworkXError as e:
            logger.error(f"Error calculating paths: {e}")
        
        return self
    
    def classify_network_roles(self):
        """Classify nodes into network roles based on multiple metrics"""
        logger.info("Classifying network roles...")
        
        for node in self.graph.nodes():
            node_data = self.graph.nodes[node]
            
            # Get centrality values
            degree_c = node_data.get('degree_centrality', 0)
            between_c = node_data.get('betweenness_centrality', 0)
            close_c = node_data.get('closeness_centrality', 0)
            
            # Classify into roles
            if between_c > 0.1 and degree_c > 0.05:
                role = 'connector'  # High betweenness = bridges communities
            elif degree_c > 0.1:
                role = 'hub'  # High degree = many connections
            elif close_c > 0.3:
                role = 'central'  # High closeness = near center
            else:
                role = 'peripheral'  # Low on all metrics
            
            self.graph.nodes[node]['network_role'] = role
        
        # Count roles
        role_counts = Counter([self.graph.nodes[n]['network_role'] 
                              for n in self.graph.nodes()])
        
        logger.info("Network role distribution:")
        for role, count in role_counts.items():
            logger.info(f"  {role}: {count} nodes")
        
        self.analysis_results['role_distribution'] = dict(role_counts)
        
        return self
    
    def save_enriched_network(self, output_path: str):
        """Save network with all new analysis features"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'wb') as f:
            pickle.dump(self.graph, f)
        
        logger.info(f"Saved enriched network to: {output_path}")
        return self
    
    def save_analysis_report(self, output_path: str):
        """Save analysis results as JSON"""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, 'w') as f:
            json.dump(self.analysis_results, f, indent=2)
        
        logger.info(f"Saved analysis report to: {output_path}")
        return self
    
    def run_full_analysis(self):
        """Run complete network structure analysis pipeline"""
        logger.info("="*60)
        logger.info("NETWORK STRUCTURE ANALYSIS PIPELINE")
        logger.info("="*60)
        
        (self
            .load_network()
            .find_downtown_node()
            .detect_communities()
            .classify_hub_periphery()
            .calculate_path_metrics()
            .classify_network_roles()
            .save_enriched_network('data/processed/networks/spatial_network_enriched.gpickle')
            .save_analysis_report('results/network_structure_analysis.json'))
        
        logger.info("="*60)
        logger.info("NETWORK STRUCTURE ANALYSIS COMPLETE")
        logger.info("="*60)
        
        return self


def main():
    """Main execution"""
    analyzer = NetworkStructureAnalyzer(
        network_path='data/processed/networks/spatial_network_distance.gpickle'
    )
    
    analyzer.run_full_analysis()
    
    print("\n✓ Network structure analysis complete!")
    print("  Next: Re-run step1_data_pipeline_features.py with enriched network")


if __name__ == "__main__":
    main()