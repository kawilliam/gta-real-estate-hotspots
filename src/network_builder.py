"""
Network Construction Module

This module handles the construction of spatial networks for GTA real estate analysis:
- Aggregate data to FSA (Forward Sortation Area) level
- Build nodes representing geographic areas
- Create edges based on spatial proximity, adjacency, and travel time
- Calculate network properties and metrics

Author: Yadon Kassahun (Network Architect)
Date: 2024-11-06
"""

import os
import sys
import pandas as pd
import numpy as np
import geopandas as gpd
import networkx as nx
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from scipy.spatial import distance_matrix
from shapely.geometry import Point, Polygon
import pickle

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS = PROJECT_ROOT / "results"

DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
(DATA_PROCESSED / "networks").mkdir(parents=True, exist_ok=True)

class FSAAggregator:
    """
    Aggregates various data sources to Forward Sortation Area (FSA) level.
    
    FSA is the first 3 characters of a Canadian postal code (e.g., M5V).
    This provides a manageable geographic unit for analysis.
    """

    def __init__(self):
        logger.info("Initialized FSA Aggregator")
        self.fsa_data = {}

    def load_fsa_boundaries(self) -> gpd.GeoDataFrame:
        """
        Load or create FSA boundary polygons.
        
        For now, we'll create centroids based on available data.
        In production, would use official FSA boundary shapefiles.
        
        Returns:
            GeoDataFrame with FSA geometries
        """
        logger.info("Loading FSA boundaries...")

        logger.warning("Using simplified FSA centroids. Consider adding boundary shapefiles.")

        return None
    
    def aggregate_real_estate(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate real estate data to FSA level.
        
        Args:
            df: Real estate DataFrame with neighborhood/FSA information
            
        Returns:
            Aggregated DataFrame at FSA level
        """
        logger.info("Aggergating real estate data to FSA level...")

        fsa_col = None
        for col in ['FSA', 'PostalCode', 'Postal_Code', 'Neighbourhood']:
            if col in df.columns:
                fsa_col = col
                break
        
        if fsa_col is None:
            logger.warning("No FSA/Neighbourhood column found in real estate data")
            return pd.DataFrame()
        
        logger.info(f"Using column: {fsa_col}")
        
        if 'postal' in fsa_col.lower():
            df['FSA'] = df[fsa_col].str[:3]
        else:
            df['FSA'] = df[fsa_col]

        price_cols = [col for col in df.columns 
                     if 'price' in col.lower() or 'value' in col.lower()]

        agg_dict = {}

        for col in price_cols:
            if df[col].dtype in ['float64', 'int64']:
                agg_dict[col] = ['mean', 'median', 'std', 'count']
        
        group_cols = ['FSA']
        if 'Year' in df.columns:
            group_cols.append('Year')
        
        if agg_dict:
            df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
            df_agg.columns = ['_'.join(col).strip('_') if col[1] else col[0] 
                             for col in df_agg.columns.values]
            
            logger.info(f"Aggregated to {len(df_agg)} FSA-year combinations")
            return df_agg
        else:
            logger.warning("No numeric columns found for aggregation")
            return pd.DataFrame()
        
    def aggregate_building_permits(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate building permits to FSA level using postal codes.
        
        Args:
            df: Building permits DataFrame with POSTAL column
            
        Returns:
            Aggregated DataFrame with permit counts and values by FSA
        """
        logger.info("Aggregating building permits to FSA level...")
        
        # Check for postal code column
        if 'POSTAL' not in df.columns:
            logger.error("POSTAL column not found in permits data")
            logger.error(f"Available columns: {list(df.columns)[:20]}")
            return pd.DataFrame()
        
        logger.info(f"Using POSTAL column for FSA extraction")
        
        # Extract FSA from postal code (first 3 characters)
        df['FSA'] = df['POSTAL'].astype(str).str[:3].str.upper()
        
        # Remove invalid FSAs (NaN, too short, etc.)
        df = df[df['FSA'].str.len() == 3].copy()
        
        logger.info(f"Extracted {df['FSA'].nunique()} unique FSAs from postal codes")
        
        # Add year if date columns exist
        date_cols = [col for col in df.columns if 'DATE' in col.upper()]
        if date_cols:
            for col in date_cols:
                try:
                    df[col] = pd.to_datetime(df[col], errors='coerce')
                    if df[col].notna().any():
                        df['Year'] = df[col].dt.year
                        logger.info(f"Extracted year from {col}")
                        break
                except:
                    pass
        
        # Prepare aggregation
        group_cols = ['FSA']
        if 'Year' in df.columns:
            group_cols.append('Year')
        
        agg_dict = {}
        
        # Count permits
        agg_dict['PERMIT_NUM'] = 'count'
        
        # Sum estimated cost if available
        if 'EST_CONST_COST' in df.columns:
            df['EST_CONST_COST_NUM'] = pd.to_numeric(df['EST_CONST_COST'], errors='coerce')
            agg_dict['EST_CONST_COST_NUM'] = 'sum'
        
        # Aggregate
        df_agg = df.groupby(group_cols).agg(agg_dict).reset_index()
        
        # Rename columns
        df_agg.columns = list(group_cols) + ['Permit_Count', 'Total_Construction_Value'][:len(agg_dict)]
        
        # For network building, we need approximate coordinates
        # Use FSA centroids (we'll calculate from first 3 chars of postal code)
        # This is a simplified approach - in production use proper geocoding
        
        # Create pseudo-coordinates based on FSA
        # Toronto FSAs range roughly: M1A-M9W
        # We'll create a simple mapping
        fsa_coords = {}
        for fsa in df_agg['FSA'].unique():
            if fsa.startswith('M'):
                # Extract the number and letter
                try:
                    num = int(fsa[1])
                    letter_ord = ord(fsa[2]) - ord('A')
                    
                    # Map to approximate Toronto coordinates
                    # This is a rough approximation
                    lat = 43.65 + (num - 5) * 0.05  # Spread N-S
                    lon = -79.40 + letter_ord * 0.02  # Spread E-W
                    
                    fsa_coords[fsa] = (lat, lon)
                except:
                    # Default to downtown Toronto
                    fsa_coords[fsa] = (43.65, -79.38)
        
        # Add coordinates to aggregated data
        df_agg['Centroid_Lat'] = df_agg['FSA'].map(lambda x: fsa_coords.get(x, (43.65, -79.38))[0])
        df_agg['Centroid_Lon'] = df_agg['FSA'].map(lambda x: fsa_coords.get(x, (43.65, -79.38))[1])
        
        logger.info(f"Aggregated to {len(df_agg)} FSA-level records")
        
        return df_agg
    
    def create_fsa_master_table(self, 
                                real_estate_df: Optional[pd.DataFrame] = None,
                                permits_df: Optional[pd.DataFrame] = None,
                                year: Optional[int] = None) -> pd.DataFrame:
        """
        Create master table combining all data sources at FSA level.
        
        Args:
            real_estate_df: Aggregated real estate data
            permits_df: Aggregated permits data
            year: Specific year to filter (optional)
            
        Returns:
            Combined DataFrame at FSA level
        """
        logger.info("Creating FSA master table...")
        
        dfs_to_merge = []

        if real_estate_df is not None and not real_estate_df.empty:
            df_re = real_estate_df.copy()
            if year and 'Year' in df_re.columns:
                df_re = df_re[df_re['Year'] == year]
            dfs_to_merge.append(('real_estate', df_re))
            logger.info(f"Added real estate data: {len(df_re)} records")
     
        if permits_df is not None and not permits_df.empty:
            df_pm = permits_df.copy()
            if year and 'Year' in df_pm.columns:
                df_pm = df_pm[df_pm['Year'] == year]
            dfs_to_merge.append(('permits', df_pm))
            logger.info(f"Added permits data: {len(df_pm)} records")
        
        if not dfs_to_merge:
            logger.warning("No data sources provided for master table")
            return pd.DataFrame()
     
        master_df = dfs_to_merge[0][1]
    
        for name, df in dfs_to_merge[1:]:
            merge_keys = ['FSA'] if 'FSA' in master_df.columns else ['FSA_TEMP']
            if year is None and 'Year' in master_df.columns and 'Year' in df.columns:
                merge_keys.append('Year')
            
            master_df = master_df.merge(df, on=merge_keys, how='outer', suffixes=('', f'_{name}'))
            logger.info(f"Merged {name} data")
        
        logger.info(f"Master table created with {len(master_df)} records and {len(master_df.columns)} columns")
        
        return master_df
    
class SpatialNetworkBuilder:
    """
    Builds spatial networks from FSA-level data.
    
    Creates a graph where:
    - Nodes = FSA areas with attributes (prices, permits, demographics)
    - Edges = connections based on proximity, adjacency, or travel time
    """
    
    def __init__(self, fsa_data: pd.DataFrame):
        """
        Initialize network builder.
        
        Args:
            fsa_data: Master DataFrame at FSA level with coordinates
        """
        self.fsa_data = fsa_data
        self.graph = None
        logger.info(f"Initialized Network Builder with {len(fsa_data)} FSA areas")
    
    def _extract_coordinates(self) -> Dict[str, Tuple[float, float]]:
        """
        Extract centroid coordinates for each FSA.
        
        Returns:
            Dictionary mapping FSA to (lat, lon) tuples
        """
        coords = {}

        lat_col = None
        lon_col = None
        
        for col in self.fsa_data.columns:
            if 'lat' in col.lower() and lat_col is None:
                lat_col = col
            if 'lon' in col.lower() and lon_col is None:
                lon_col = col
        
        if lat_col is None or lon_col is None:
            logger.error("Could not find latitude/longitude columns")
            return coords

        fsa_col = 'FSA' if 'FSA' in self.fsa_data.columns else 'FSA_TEMP'
        
        for _, row in self.fsa_data.iterrows():
            fsa = row[fsa_col]
            lat = row[lat_col]
            lon = row[lon_col]
            
            if pd.notna(lat) and pd.notna(lon):
                coords[fsa] = (lat, lon)
        
        logger.info(f"Extracted coordinates for {len(coords)} FSA areas")
        return coords
    
    def build_graph(self, 
                   edge_method: str = 'distance',
                   distance_threshold_km: float = 2.0,
                   k_neighbors: int = 5) -> nx.Graph:
        """
        Build spatial network graph.
        
        Args:
            edge_method: Method for creating edges
                - 'distance': Connect nodes within distance threshold
                - 'knn': Connect each node to k nearest neighbors
                - 'delaunay': Delaunay triangulation (natural neighbors)
            distance_threshold_km: Max distance for edge creation (km)
            k_neighbors: Number of neighbors for KNN method
            
        Returns:
            NetworkX graph
        """
        logger.info("=" * 60)
        logger.info(f"Building Spatial Network - Method: {edge_method}")
        logger.info("=" * 60)
 
        G = nx.Graph()
 
        coords = self._extract_coordinates()
        
        if not coords:
            logger.error("No valid coordinates found. Cannot build graph.")
            return G
  
        fsa_col = 'FSA' if 'FSA' in self.fsa_data.columns else 'FSA_TEMP'
        
        for _, row in self.fsa_data.iterrows():
            fsa = row[fsa_col]
            
            if fsa not in coords:
                continue
  
            attrs = {
                'lat': coords[fsa][0],
                'lon': coords[fsa][1],
                'pos': coords[fsa]  
            }

            for col in self.fsa_data.columns:
                if col not in [fsa_col, 'LATITUDE', 'LONGITUDE', 'Centroid_Lat', 'Centroid_Lon']:
                    value = row[col]
                    if pd.notna(value):
                        attrs[col] = value
            
            G.add_node(fsa, **attrs)
        
        logger.info(f"Added {G.number_of_nodes()} nodes to graph")

        if edge_method == 'distance':
            self._add_distance_edges(G, coords, distance_threshold_km)
        elif edge_method == 'knn':
            self._add_knn_edges(G, coords, k_neighbors)
        elif edge_method == 'delaunay':
            self._add_delaunay_edges(G, coords)
        else:
            logger.error(f"Unknown edge method: {edge_method}")
        
        logger.info(f"Added {G.number_of_edges()} edges to graph")
        logger.info(f"Average degree: {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
        
        self.graph = G
        return G
    
    def _haversine_distance(self, coord1: Tuple[float, float], 
                           coord2: Tuple[float, float]) -> float:
        """
        Calculate haversine distance between two lat/lon coordinates.
        
        Args:
            coord1: (lat1, lon1)
            coord2: (lat2, lon2)
            
        Returns:
            Distance in kilometers
        """
        lat1, lon1 = coord1
        lat2, lon2 = coord2
  
        R = 6371.0
  
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlon = np.radians(lon2 - lon1)
        dlat = np.radians(lat2 - lat1)
    
        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        return R * c
    
    def _add_distance_edges(self, G: nx.Graph, 
                           coords: Dict[str, Tuple[float, float]],
                           threshold_km: float) -> None:
        """
        Add edges between nodes within distance threshold.
        
        Args:
            G: NetworkX graph to modify
            coords: Dictionary of FSA coordinates
            threshold_km: Maximum distance in kilometers
        """
        logger.info(f"Creating edges for nodes within {threshold_km} km...")
        
        nodes = list(coords.keys())
        edge_count = 0
        
        for i, node1 in enumerate(nodes):
            for node2 in nodes[i+1:]:
                dist = self._haversine_distance(coords[node1], coords[node2])
                
                if dist <= threshold_km:
                    G.add_edge(node1, node2, weight=dist, distance_km=dist)
                    edge_count += 1
        
        logger.info(f"Created {edge_count} edges")

    def _add_knn_edges(self, G: nx.Graph,
                       coords: Dict[str, Tuple[float, float]],
                       k: int) -> None:
        """
        Add edges to k-nearest neighbors for each node.
        
        Args:
            G: NetworkX graph to modify
            coords: Dictionary of FSA coordinates
            k: Number of nearest neighbors
        """
        logger.info(f"Creating edges to {k} nearest neighbors for each node...")
        
        nodes = list(coords.keys())
        
        coord_matrix = np.array([coords[node] for node in nodes])
        
        from scipy.spatial.distance import pdist, squareform
        distances = squareform(pdist(coord_matrix, metric='euclidean'))
        
        edge_count = 0
        edges_added = set()
        
        for i, node1 in enumerate(nodes):
            nearest_indices = np.argsort(distances[i])[1:k+1]
            
            for j in nearest_indices:
                node2 = nodes[j]
  
                edge = tuple(sorted([node1, node2]))
                if edge not in edges_added:
                    dist = self._haversine_distance(coords[node1], coords[node2])
                    G.add_edge(node1, node2, weight=dist, distance_km=dist)
                    edges_added.add(edge)
                    edge_count += 1
        
        logger.info(f"Created {edge_count} edges")
    
    def _add_delaunay_edges(self, G: nx.Graph,
                           coords: Dict[str, Tuple[float, float]]) -> None:
        """
        Add edges based on Delaunay triangulation (natural neighbors).
        
        Args:
            G: NetworkX graph to modify
            coords: Dictionary of FSA coordinates
        """
        logger.info("Creating edges using Delaunay triangulation...")
        
        try:
            from scipy.spatial import Delaunay
        except ImportError:
            logger.error("scipy not available for Delaunay triangulation")
            return
        
        nodes = list(coords.keys())
        points = np.array([coords[node] for node in nodes])

        tri = Delaunay(points)
        
        edge_count = 0
        edges_added = set()

        for simplex in tri.simplices:
            for i in range(3):
                for j in range(i+1, 3):
                    node1 = nodes[simplex[i]]
                    node2 = nodes[simplex[j]]
                    
                    edge = tuple(sorted([node1, node2]))
                    if edge not in edges_added:
                        dist = self._haversine_distance(coords[node1], coords[node2])
                        G.add_edge(node1, node2, weight=dist, distance_km=dist)
                        edges_added.add(edge)
                        edge_count += 1
        
        logger.info(f"Created {edge_count} edges")
    
    def calculate_network_metrics(self) -> Dict[str, float]:
        """
        Calculate various network metrics and properties.
        
        Returns:
            Dictionary of network metrics
        """
        if self.graph is None:
            logger.error("Graph not built yet. Call build_graph() first.")
            return {}
        
        logger.info("Calculating network metrics...")
        
        G = self.graph
        metrics = {}

        metrics['num_nodes'] = G.number_of_nodes()
        metrics['num_edges'] = G.number_of_edges()
        metrics['density'] = nx.density(G)

        degrees = dict(G.degree())
        metrics['avg_degree'] = np.mean(list(degrees.values()))
        metrics['max_degree'] = np.max(list(degrees.values()))
        metrics['min_degree'] = np.min(list(degrees.values()))

        metrics['is_connected'] = nx.is_connected(G)
        metrics['num_components'] = nx.number_connected_components(G)
        
        if nx.is_connected(G):
            metrics['diameter'] = nx.diameter(G)
            metrics['avg_shortest_path_length'] = nx.average_shortest_path_length(G)
        else:
            largest_cc = max(nx.connected_components(G), key=len)
            G_largest = G.subgraph(largest_cc).copy()
            metrics['diameter_largest_component'] = nx.diameter(G_largest)
            metrics['avg_shortest_path_length_largest'] = nx.average_shortest_path_length(G_largest)

        metrics['avg_clustering_coefficient'] = nx.average_clustering(G)

        if G.number_of_nodes() <= 1000:
            degree_centrality = nx.degree_centrality(G)
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
            
            betweenness = nx.betweenness_centrality(G)
            metrics['avg_betweenness_centrality'] = np.mean(list(betweenness.values()))
        else:
            logger.info("Graph too large for full centrality calculation. Skipping.")
        
        logger.info("Network metrics calculated:")
        for key, value in metrics.items():
            logger.info(f"  {key}: {value}")
        
        return metrics
    
    def add_spatial_lag_features(self, feature_cols: List[str]) -> None:
        """
        Add spatial lag features (weighted average of neighbors' values).
        
        Args:
            feature_cols: List of column names to create spatial lags for
        """
        if self.graph is None:
            logger.error("Graph not built yet. Call build_graph() first.")
            return
        
        logger.info(f"Adding spatial lag features for: {feature_cols}")
        
        G = self.graph
        
        for col in feature_cols:
            has_feature = any(col in G.nodes[node] for node in G.nodes())
            
            if not has_feature:
                logger.warning(f"Feature '{col}' not found in node attributes. Skipping.")
                continue

            for node in G.nodes():
                neighbors = list(G.neighbors(node))
                
                if not neighbors:
                    G.nodes[node][f'{col}_spatial_lag'] = None
                    continue

                neighbor_values = []
                weights = []
                
                for neighbor in neighbors:
                    if col in G.nodes[neighbor]:
                        value = G.nodes[neighbor][col]
                        if pd.notna(value):
                            neighbor_values.append(value)
                            edge_data = G.get_edge_data(node, neighbor)
                            dist = edge_data.get('distance_km', 1.0)
                            weights.append(1.0 / (dist + 0.1)) 

                if neighbor_values:
                    weighted_avg = np.average(neighbor_values, weights=weights)
                    G.nodes[node][f'{col}_spatial_lag'] = weighted_avg
                else:
                    G.nodes[node][f'{col}_spatial_lag'] = None
            
            logger.info(f"Added spatial lag for '{col}'")
    
    def save_graph(self, filename: str, format: str = 'gpickle') -> None:
        """
        Save graph to file.
        
        Args:
            filename: Output filename
            format: File format ('gpickle', 'graphml', 'gexf', 'edgelist')
        """
        if self.graph is None:
            logger.error("No graph to save. Build graph first.")
            return
        
        output_path = DATA_PROCESSED / "networks" / filename
        
        logger.info(f"Saving graph to: {output_path}")
        
        if format == 'gpickle':
            import pickle
            with open(output_path, 'wb') as f:
                pickle.dump(self.graph, f)
        elif format == 'graphml':
            nx.write_graphml(self.graph, output_path)
        elif format == 'gexf':
            nx.write_gexf(self.graph, output_path)
        elif format == 'edgelist':
            nx.write_edgelist(self.graph, output_path)
        else:
            logger.error(f"Unknown format: {format}")
            return
        
        logger.info(f"Graph saved successfully")
    
    @staticmethod
    def load_graph(filename: str, format: str = 'gpickle') -> nx.Graph:
        """
        Load graph from file.
        
        Args:
            filename: Input filename
            format: File format ('gpickle', 'graphml', 'gexf', 'edgelist')
            
        Returns:
            NetworkX graph
        """
        input_path = DATA_PROCESSED / "networks" / filename
        
        logger.info(f"Loading graph from: {input_path}")
        
        if format == 'gpickle':
            import pickle
            with open(input_path, 'rb') as f:
                G = pickle.load(f)
        elif format == 'graphml':
            G = nx.read_graphml(input_path)
        elif format == 'gexf':
            G = nx.read_gexf(input_path)
        elif format == 'edgelist':
            G = nx.read_edgelist(input_path)
        else:
            logger.error(f"Unknown format: {format}")
            return None
        
        logger.info(f"Loaded graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
        
        return G
    
def build_network_pipeline(edge_method: str = 'distance',
                          distance_threshold: float = 2.0,
                          k_neighbors: int = 5,
                          year: Optional[int] = None) -> nx.Graph:
    """
    Complete network building pipeline.
    
    Args:
        edge_method: Method for edge creation ('distance', 'knn', 'delaunay')
        distance_threshold: Distance threshold in km (for distance method)
        k_neighbors: Number of neighbors (for knn method)
        year: Specific year to filter data (optional)
        
    Returns:
        Constructed NetworkX graph
    """
    logger.info("\n" + "=" * 60)
    logger.info("NETWORK CONSTRUCTION PIPELINE")
    logger.info("=" * 60 + "\n")

    aggregator = FSAAggregator()

    real_estate_files = list((DATA_RAW / "real_estate").glob("*.csv"))
    df_real_estate_agg = None
    
    if real_estate_files:
        logger.info("[Step 1/5] Loading and aggregating real estate data...")
        latest_file = max(real_estate_files, key=lambda p: p.stat().st_mtime)
        df_real_estate = pd.read_csv(latest_file)
        df_real_estate_agg = aggregator.aggregate_real_estate(df_real_estate)

        output_file = DATA_PROCESSED / "fsa_aggregated" / "real_estate_fsa.csv"
        output_file.parent.mkdir(parents=True, exist_ok=True)
        df_real_estate_agg.to_csv(output_file, index=False)
        logger.info(f"Saved aggregated real estate data to: {output_file}")
    else:
        logger.warning("No real estate data found. Skipping.")

    permits_files = list((DATA_RAW / "permits").glob("*.csv"))
    df_permits_agg = None
    
    if permits_files:
        logger.info("\n[Step 2/5] Loading and aggregating building permits...")
        latest_file = max(permits_files, key=lambda p: p.stat().st_mtime)
        df_permits = pd.read_csv(latest_file, low_memory=False)

        date_cols = [col for col in df_permits.columns if 'DATE' in col.upper()]
        if date_cols:
            for col in date_cols:
                df_permits[col] = pd.to_datetime(df_permits[col], errors='coerce')
            df_permits['Year'] = df_permits[date_cols[0]].dt.year
        
        df_permits_agg = aggregator.aggregate_building_permits(df_permits)

        output_file = DATA_PROCESSED / "fsa_aggregated" / "building_permits_fsa.csv"
        df_permits_agg.to_csv(output_file, index=False)
        logger.info(f"Saved aggregated permits data to: {output_file}")
    else:
        logger.warning("No building permits data found. Skipping.")

    logger.info("\n[Step 3/5] Creating FSA master table...")
    df_master = aggregator.create_fsa_master_table(
        real_estate_df=df_real_estate_agg,
        permits_df=df_permits_agg,
        year=year
    )
    
    if df_master.empty:
        logger.error("Master table is empty. Cannot build network.")
        return None

    master_file = DATA_PROCESSED / "fsa_master_table.csv"
    df_master.to_csv(master_file, index=False)
    logger.info(f"Saved master table to: {master_file}")
    logger.info(f"Master table shape: {df_master.shape}")

    logger.info("\n[Step 4/5] Building spatial network...")
    builder = SpatialNetworkBuilder(df_master)
    
    G = builder.build_graph(
        edge_method=edge_method,
        distance_threshold_km=distance_threshold,
        k_neighbors=k_neighbors
    )
    
    if G.number_of_nodes() == 0:
        logger.error("Graph has no nodes. Check data and coordinates.")
        return None
  
    logger.info("\n[Step 5/5] Calculating network metrics...")
    metrics = builder.calculate_network_metrics()
  
    metrics_file = RESULTS / "tables" / "network_metrics.json"
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    
    import json
    def convert_to_native(obj):
        if hasattr(obj, 'item'):
            return obj.item()
        elif isinstance(obj, dict):
            return {k: convert_to_native(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_native(item) for item in obj]
        else:
            return obj

    metrics_native = convert_to_native(metrics)

    with open(metrics_file, 'w') as f:
        json.dump(metrics_native, f, indent=2)
    logger.info(f"Saved network metrics to: {metrics_file}")
   
    logger.info("\nAdding spatial lag features...")
    feature_cols = [col for col in df_master.columns 
                   if any(x in col.lower() for x in ['price', 'value', 'permit', 'count'])]
    
    if feature_cols:
        builder.add_spatial_lag_features(feature_cols[:5])
        logger.info(f"Added spatial lags for: {feature_cols[:5]}")

    # Save graph
    logger.info("\nSaving network graph...")
    graph_filename = f"spatial_network_{edge_method}"
    if year:
        graph_filename += f"_{year}"
    graph_filename += ".gpickle"

    builder.save_graph(graph_filename, format='gpickle')

    # Skip GraphML - causes issues with tuple attributes
    logger.info("Skipping GraphML export (use gpickle for Python, or export nodes/edges separately for other tools)")
    
    logger.info("\n" + "=" * 60)
    logger.info("NETWORK CONSTRUCTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Nodes: {G.number_of_nodes()}")
    logger.info(f"Edges: {G.number_of_edges()}")
    logger.info(f"Graph saved to: data/processed/networks/{graph_filename}")
    logger.info("=" * 60)
    
    return G


def main():
    """
    Main function for CLI execution.
    
    Usage:
        python src/network_builder.py --method distance --threshold 2.0
        python src/network_builder.py --method knn --k 5
        python src/network_builder.py --method delaunay
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description='Build spatial network for GTA Real Estate Hotspots project'
    )
    
    parser.add_argument(
        '--method',
        type=str,
        default='distance',
        choices=['distance', 'knn', 'delaunay'],
        help='Edge creation method (default: distance)'
    )
    
    parser.add_argument(
        '--threshold',
        type=float,
        default=2.0,
        help='Distance threshold in km for distance method (default: 2.0)'
    )
    
    parser.add_argument(
        '--k',
        type=int,
        default=5,
        help='Number of neighbors for KNN method (default: 5)'
    )
    
    parser.add_argument(
        '--year',
        type=int,
        default=None,
        help='Filter data to specific year (optional)'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run with test parameters (faster)'
    )
    
    args = parser.parse_args()

    if args.test:
        logger.info("Running in TEST mode")
        args.threshold = 5.0  
        args.k = 3

    try:
        G = build_network_pipeline(
            edge_method=args.method,
            distance_threshold=args.threshold,
            k_neighbors=args.k,
            year=args.year
        )
        
        if G is not None:
            print("\n" + "=" * 60)
            print("NETWORK SUMMARY")
            print("=" * 60)
            print(f"Method:        {args.method}")
            print(f"Nodes:         {G.number_of_nodes():,}")
            print(f"Edges:         {G.number_of_edges():,}")
            print(f"Avg Degree:    {sum(dict(G.degree()).values()) / G.number_of_nodes():.2f}")
            print("=" * 60)
            
            print("\nNext steps:")
            print("  1. Visualize network: jupyter notebook notebooks/02_network_construction.ipynb")
            print("  2. Engineer features: python src/features.py")
            print("  3. Train models: python src/models.py")
        else:
            logger.error("Network construction failed.")
            sys.exit(1)
            
    except Exception as e:
        logger.error(f"Network construction failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()