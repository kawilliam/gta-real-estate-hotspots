# GTA Real Estate Hotspots: A Graph-Based Network Approach

**Course:** EECS 4414 - Information Networks  
**Institution:** York University  
**Team Members:**
- Kyle Williamson (218953901) - Data Engineer
- Yadon Kassahun (219744291) - Network Architect
- Utsav Patel (219577840) - Modeler
- Hari Patel (219952670) - Analyst/Writer

## Project Overview

This project uses graph-based spatial network analysis to predict real estate hotspots in the Greater Toronto Area (GTA). We model neighborhoods as nodes and their connectivity (transit/roads) as edges to forecast next-year price growth areas.

## Research Questions

1. Can graph-based spatial networks predict hotspots better than baseline models?
2. Which features (accessibility, amenities, development) are most predictive of price growth?
3. Do spatial models generalize across different GTA regions?
4. Can we provide interpretable explanations for predictions?

## Project Structure
```
gta-real-estate-hotspots/
├── data/               # Raw and processed datasets
├── notebooks/          # Jupyter notebooks for exploration and analysis
├── src/                # Source code for data processing and modeling
├── models/             # Saved trained models
├── results/            # Figures, tables, and evaluation results
├── reports/            # Project deliverables (proposal, midterm, final)
└── tests/              # Unit tests
```

## Timeline

- **Week 1** (Oct 14-20): Data acquisition and validation
- **Weeks 2-3** (Oct 21-Nov 3): Network construction, feature engineering, EDA
- **Weeks 4-5** (Nov 4-17): Baseline models (Naive, LASSO, XGBoost)
- **Week 6** (Nov 18-24): Spatial Autoregressive (SAR) model
- **Week 7** (Nov 25-Dec 1): GWR/GCN extensions (if time permits)
- **Week 8** (Dec 2-8): Final evaluation, visualization, report

## Data Sources

### Primary Sources
- **Real Estate Transactions:** Toronto Open Data
- **Transit Networks:** OpenStreetMap, TTC/GO Transit GTFS
- **Development Activity:** City of Toronto building permits API
- **Amenities:** OpenStreetMap POI data (Overpass API)
- **Demographics:** Statistics Canada 2021 Census

### Backup Sources
- Rentals.ca API
- Web scraping Realtor.ca (as last resort)

## Installation
```bash
# Clone repository
git clone https://github.com/YOUR_USERNAME/gta-real-estate-hotspots.git
cd gta-real-estate-hotspots

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

## Usage
```bash
# Run data collection
python src/data_collection.py

# Build spatial network
python src/network_builder.py

# Train models
python src/models.py --model lasso
python src/models.py --model xgboost
python src/models.py --model sar
```

## Current Status

- [x] Project proposal submitted
- [ ] Data acquisition in progress
- [ ] Network construction not started
- [ ] Baseline models not started
- [ ] Spatial models not started

## Contact

For questions or issues, contact: kyle.williamson@yorku.ca