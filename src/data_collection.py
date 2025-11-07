"""
Data Collection Module

This module handles data acquisition from various sources, including:
- Toronto Open Data (real estate transactions)
- OpenStreetMap (transit and amenities)
- Building permits
- Census data

Author: Kyle Williamson (Data Engineer)
Date: 2024-11-06
"""

import os
import sys
import pandas as pd
import requests
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, List
import json

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"

DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)

class TorontoOpenDataCollector:
    """
    Collector for Toronto Open Data portal.
    
    This class handles fetching real estate and building permit data
    from Toronto's Open Data portal.
    """

    def __init__(self):
        """Initialize the Toronto Open Data collector."""
        self.base_url = "https://ckan0.cf.opendata.inter.prod-toronto.ca/api/3/action"
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'GTA-Hotspots-Research/0.1 (York University)'
        })

    def _make_request(self, endpoint: str, params: Optional[Dict] = None) -> Dict:
        """
        Make a request to Toronto Open Data API.
        
        Args:
            endpoint: API endpoint
            params: Query parameters
            
        Returns:
            JSON response as a dictionary
        
        Raises:
            requests.RequestException: If request fails
        """
        url = f"{self.base_url}/{endpoint}"

        try:
            response = self.session.get(url, params=params, timeout=30)
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            logger.error(f"Request failed: {e}")
            raise

    def get_dataset_metadata(self, package_id: str) -> Dict:
        """
        Get metadata for a specific dataset.
        
        Args:
            package_id: Dataset identifier from Toronto Open Data

        Returns:
            Dictionary containing dataset metadata
        """
        logger.info(f"Fetching metadata for package: {package_id}")

        params = {'id': package_id}
        response = self._make_request("package_show", params)

        if response.get("success"):
            return response['result']
        else:
            raise ValueError(f"Failed to fetch metadata for {package_id}")
        
    def download_resource(self, resource_url: str, output_path: Path) -> None:
        """
        Download a resource file from Toronto Open Data.
        
        Args:
            resource_url: URL of the resource to download
            output_path: Path where file should be saved
        """

        logger.info(f"Downloading resource from {resource_url}")
        logger.info(f"Saving to: {output_path}")

        try:
            response = self.session.get(resource_url, stream=True, timeout=60)
            response.raise_for_status()

            output_path.parent.mkdir(parents=True, exist_ok=True)

            total_size = int(response.headers.get('content-length', 0))

            with open(output_path, 'wb') as f:
                if total_size == 0:
                    f.write(response.content)
                else:
                    downloaded = 0
                    for chunk in response.iter_content(chunk_size=8192):
                        if chunk:
                            f.write(chunk)
                            downloaded += len(chunk)

                            progress = (downloaded / total_size) * 100
                            if downloaded % (total_size // 10 + 1) < 8192:
                                logger.info(f"Progress: {progress:.1f}%")

            logger.info(f"Download completed: {output_path}")

        except requests.RequestException as e:
            logger.error(f"Download failed: {e}")
            raise

    def fetch_building_permits(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Fetch building permits data from Toronto Open Data.
        
        Dataset: Building Permits - Cleared Permits
        Package ID: building-permits-cleared-permits

        Args:
            output_dir: Directory to save raw data (default: data/raw/permits/)
        
        Returns:
            DataFrame containing building permits data
        """    
        logger.info("=" * 60)
        logger.info("Fetching Building Permits Data")
        logger.info("=" * 60)

        if output_dir is None:
            output_dir = DATA_RAW / "permits"

        output_dir.mkdir(parents=True, exist_ok=True)

        package_id = "building-permits-cleared-permits"

        try:
            metadata = self.get_dataset_metadata(package_id)

            resources = metadata.get("resources", [])
            csv_resources = [r for r in resources if r['format'].upper() == 'CSV']
            
            if not csv_resources:
                raise ValueError("No CSV resources found for building permits")
            
            resource = csv_resources[0]
            resource_url = resource['url']
            resource_name = resource['name']

            logger.info(f"Found resource: {resource_name}")

            output_file = output_dir / f"building_permits_{datetime.now().strftime('%Y%m%d')}.csv"
            self.download_resource(resource_url, output_file)

            logger.info("Loading data into DataFrame...")
            df = pd.read_csv(output_file)

            logger.info(f"Loaded {len(df):,} building permit records")
            logger.info(f"Columns: {list(df.columns)}")

            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch building permits: {e}")
            raise

    def fetch_real_estate_data(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Fetch real estate/housing data from Toronto Open Data.
        
        Dataset: Wellbeing Toronto - Housing
        Package ID: wellbeing-toronto-housing
        
        Args:
            output_dir: Directory to save raw data (default: data/raw/real_estate/)
            
        Returns:
            DataFrame containing real estate data
        """
        logger.info("=" * 60)
        logger.info("Fetching Real Estate Data")
        logger.info("=" * 60)

        if output_dir is None:
            output_dir = DATA_RAW / "real_estate"
            
        output_dir.mkdir(parents=True, exist_ok=True)

        package_id = "neighbourhood-profiles"

        try:
            metadata = self.get_dataset_metadata(package_id)

            resources = metadata.get('resources', [])
            csv_resources = [r for r in resources if r['format'].upper() == 'CSV']

            if not csv_resources:
                raise ValueError("No CSV resources found for real estate data")
            
            resource = csv_resources[0]
            resource_url = resource['url']
            resource_name = resource['name']

            logger.info(f"Found resource: {resource_name}")

            output_file = output_dir / f"real_estate_{datetime.now().strftime('%Y%m%d')}.csv"
            self.download_resource(resource_url, output_file)

            logger.info("Loading data into DataFrame...")
            df = pd.read_csv(output_file)

            logger.info(f"Loaded {len(df):,} real estate records")
            logger.info(f"Columns: {list(df.columns)}")
            logger.info(f"Data range: {df['Year'].min() if 'Year' in df.columns else 'N/A'} - {df['Year'].max() if 'Year' in df.columns else 'N/A'}")

            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch real estate data: {e}")
            raise

    def fetch_neighbourhood_profiles(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """"
        Fetch neighbourhood profile data (includes demographics and socioeconomic data).
        
        Dataset: Neighbourhood Profiles
        Package ID: neighbourhood-profiles
        
        Args:
            output_dir: Directory to save raw data (default: data/raw/demographics/)
            
        Returns:
            DataFrame containing neighbourhood profiles
        """
        logger.info("=" * 60)
        logger.info("Fetching Neighbourhood Profiles")
        logger.info("=" * 60)
        
        if output_dir is None:
            output_dir = DATA_RAW / "demographics"
        
        output_dir.mkdir(parents=True, exist_ok=True)
        
        package_id = "neighbourhood-profiles"

        try:
            metadata = self.get_dataset_metadata(package_id)

            resources = metadata.get('resources', [])
            csv_resources = [r for r in resources if r['format'].upper() == 'CSV']

            if not csv_resources:
                raise ValueError("No CSV resources found for neighbourhood profiles")
            
            census_2021 = [r for r in csv_resources if '2021' in r.get('name', '')]
            resource = census_2021[0] if census_2021 else csv_resources[0]

            resource_url = resource['url']
            resource_name = resource['name']

            logger.info(f"Found resource: {resource_name}")

            output_file = output_dir / f"neighbourhood_profiles_{datetime.now().strftime('%Y%m%d')}.csv"
            self.download_resource(resource_url, output_file)

            logger.info("Loading data into DataFrame...")
            df = pd.read_csv(output_file)

            logger.info(f"Loaded neighbourhood profile with shape: {df.shape}")

            return df
        
        except Exception as e:
            logger.error(f"Failed to fetch neighbourhood profiles: {e}")
            raise

class OpenStreetMapCollector:
    """
    Collector for OpenStreetMap data via OSMnx.
    
    This class handles fetching transit networks, road networks,
    and points of interest (amenities) from OpenStreetMap.
    """

    def __init__(self, place_name: str = "Toronto, Ontario, Canada"):
        """
        Initialize the OSM collector.
        
        Args:
            place_name: Geographic area to query (default: Toronto)
        """
        self.place_name = place_name
        logger.info(f"Initialized OSM Collector for: {place_name}")

        try:
            import osmnx as ox
            self.ox = ox

            ox.settings.log_console = True
            ox.settings.use_cache = True
        except ImportError:
            logger.error("OSMnx not installed. Install with: pip install osmnx")
            raise
    
    def fetch_road_network(self, network_type: str = "drive", 
                           output_dir: Optional[Path] = None) -> object:
        """
        Fetch road network from OpenStreetMap.
        
        Args:
            network_type: Type of network ('drive', 'walk', 'bike', 'all')
            output_dir: Directory to save network (default: data/raw/transit/)
            
        Returns:
            NetworkX MultiDiGraph of the road network
        """
        logger.info("=" * 60)
        logger.info(f"Fetching {network_type.upper()} Road Network")
        logger.info("=" * 60)

        if output_dir is None:
            output_dir = DATA_RAW / "transit"
        
        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            logger.info(f"Downloading network for: {self.place_name}")

            G = self.ox.graph_from_place(
                self.place_name,
                network_type=network_type,
                simplify=True
            )

            logger.info(f"Network downloaded: {len(G.nodes)} nodes, {len(G.edges)} edges")

            output_file = output_dir / f"road_network_{network_type}_{datetime.now().strftime('%Y%m%d').graphml}"
            self.ox.save_graphml(G, output_file)
            logger.info(f"Network saved to: {output_file}")

            return G
        
        except Exception as e:
            logger.error(f"Failed to fetch road network: {e}")
            raise

    def fetch_pois(self, tags: Dict[str, List[str]],
                   output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Fetch Points of Interest (POIs) from OpenStreetMap.
        
        Args:
            tags: Dictionary of OSM tags to query
                  Example: {'amenity': ['school', 'restaurant'], 
                           'leisure': ['park']}
            output_dir: Directory to save POIs (default: data/raw/amenities/)
            
        Returns:
            GeoDataFrame containing POI locations and attributes
        """
        logger.info("=" * 60)
        logger.info("Fetching Points of Interest")
        logger.info("=" * 60)

        if output_dir is None:
            output_dir = DATA_RAW / "amenities"

        output_dir.mkdir(parents=True, exist_ok=True)

        try:
            import geopandas as gpd

            logger.info(f"Querying POIs with tags: {tags}")
            logger.info(f"Area: {self.place_name}")

            pois = self.ox.geometries_from_place(
                self.place_name,
                tags=tags
            )

            logger.info(f"Found {len(pois)} POIs")

            output_file = output_dir / f"pois_{datetime.now().strftime('%Y%m%d')}.geojson"
            pois.to_file(output_file, driver='GeoJSON')
            logger.info(f"POIs saved to: {output_file}")

            return pois
        
        except Exception as e:
            logger.error(f"Failed to fetch POId: {e}")
            raise

    def fetch_transit_stations(self, output_dir: Optional[Path] = None) -> pd.DataFrame:
        """
        Fetch public transit stations (subway, GO train, etc.).
        
        Args:
            output_dir: Directory to save stations (default: data/raw/transit/)
            
        Returns:
            GeoDataFrame containing transit station locations
        """
        logger.info("=" * 60)
        logger.info("Fetching Transit Stations")
        logger.info("=" * 60)

        tags = {
            'public_transport': ['station', 'stop_position'],
            'railway': ['station', 'subway_entrance'],
            'amenity': ['bus_station']
        }

        if output_dir is None:
            output_dir = DATA_RAW / "transit"

        try:
            stations = self.fetch_pois(tags, output_dir)

            station_types = ['station', 'stop_position', 'subway_entrance', 'bus_station']
            mask = stations.apply(
                lambda row: any(row.get(key) in station_types
                                for key in ['public_transport', 'railway', 'amenity']),
                axis=1
            )
            stations = stations[mask]

            logger.info(f"Filtered to {len(stations)} transit stations")

            output_file = output_dir / f"transit_stations_{datetime.now().strftime('%Y%m%d')}.geojson"
            stations.to_file(output_file, driver='GeoJSON')

            return stations
        
        except Exception as e:
            logger.error(f"Failed to fetch transit stations: {e}")
            raise

def collect_all_data(sources: Optional[List[str]] = None) -> Dict[str, pd.DataFrame]:
    """"
    Collect all data sources for the project.
    
    Args:
        sources: List of data sources to collect. 
                Options: ['real_estate', 'permits', 'demographics', 
                        'roads', 'pois', 'transit']
                If None, collects all sources.
    
    Returns:
        Dictionary mapping source names to DataFrames
    """
    if sources is None:
        sources = ['real_estate', 'permits', 'demographics', 'roads', 'pois', 'transit']

    results = {}

    toronto_collector = TorontoOpenDataCollector()
    osm_collector = OpenStreetMapCollector(place_name="Toronto, Ontario, Canada")

    logger.info("\n" + "=" * 60)
    logger.info("STARTING DATA COLLECTION PIPELINNE")
    logger.info("=" * 60 + "\n")

    if 'real_estate' in sources:
        try:
            logger.info("\n[1/6] Collecting Real Estate Data...")
            results['real_estate'] = toronto_collector.fetch_real_estate_data()
        except Exception as e:
            logger.error(f"Real estate collection failed: {e}")

    if 'permits' in sources:
        try:
            logger.info("\n[2/6] Collecting Building Permits...")
            results['permits'] = toronto_collector.fetch_building_permits()
        except Exception as e:
            logger.error(f"Building permits collection failed: {e}")

    if 'demographics' in sources:
        try:
            logger.info("\n[3/6] Collecting Demographics...")
            results['demographics'] = toronto_collector.fetch_neighbourhood_profiles()
        except Exception as e:
            logger.error(f"Demographics collection failed: {e}")

    if 'roads' in sources:
        try:
            logger.info("\n[4/6] Collecting Road Network...")
            results['roads'] = osm_collector.fetch_road_network(network_type='drive')
        except Exception as e:
            logger.error(f"Road network collection failed: {e}")
    
    if 'pois' in sources:
        try:
            logger.info("\n[5/6] Collecting Points of Interest...")
            amenity_tags = {
                'amenity': ['school', 'restaurant', 'cafe', 'bank', 'hospital'],
                'leisure': ['park', 'playground'],
                'shop': ['supermarket', 'convenience']
            }
            results['pois'] = osm_collector.fetch_pois(amenity_tags)
        except Exception as e:
            logger.error(f"POI collection failed: {e}")
    
    if 'transit' in sources:
        try:
            logger.info("\n[6/6] Collecting Transit Stations...")
            results['transit'] = osm_collector.fetch_transit_stations()
        except Exception as e:
            logger.error(f"Transit stations collection failed: {e}")

    logger.info("\n" + "=" * 60)
    logger.info("DATA COLLECTION COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Successfully collected {len(results)} data sources")
    logger.info(f"Sources: {list(results.keys())}")

    return results
    
def main():
    """
    Main function for CLI execution.
    
    Usage:
        python src/data_collection.py --all
        python src/data_collection.py --sources real_estate permits
        python src/data_collection.py --help
    """
    import argparse

    parser = argparse.ArgumentParser(
        description='Collect data for GTA Real Estate Hotspots project'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Collect all data sources'
    )

    parser.add_argument(
        '--sources',
        nargs='+',
        choices=['real_estate', 'permits', 'demographics', 'roads', 'pois', 'transit'],
        help='Specific data sources to collect'
    )
    
    parser.add_argument(
        '--test',
        action='store_true',
        help='Run a quick test with limited data'
    )

    args = parser.parse_args()

    if args.all:
        sources = None
    elif args.sources:
        sources = args.sources
    elif args.test:
        logger.info("Running in TEST mode - collecting only real_estate")
        sources = ['real_estate']
    else:
        logger.info("No sources specified. Collecting Toronto Open Data sources by default.")
        logger.info("Use --all to collect everything including OSM data.")
        sources = ['real_estate', 'permits', 'demographics']

    try:
        results = collect_all_data(sources)
        
        # Print summary
        print("\n" + "=" * 60)
        print("COLLECTION SUMMARY")
        print("=" * 60)
        for source_name, data in results.items():
            if hasattr(data, '__len__'):
                print(f"{source_name:15} : {len(data):,} records")
            else:
                print(f"{source_name:15} : NetworkX graph")
        print("=" * 60)
        
        print("\nData saved to: data/raw/")
        print("Next steps:")
        print("  1. Explore data: jupyter notebook notebooks/01_data_exploration.ipynb")
        print("  2. Build network: python src/network_builder.py")
        
    except Exception as e:
        logger.error(f"Data collection failed: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()