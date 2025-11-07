# Data Directory

## Directory Structure
```
data/
├── raw/                    # Original, immutable data
│   ├── real_estate/       # Property transactions and listings
│   ├── transit/           # GTFS feeds, transit networks
│   ├── permits/           # Building permits data
│   ├── amenities/         # POI data from OpenStreetMap
│   └── demographics/      # Census data
├── processed/             # Cleaned, transformed data
│   ├── fsa_aggregated/   # Data aggregated to FSA level
│   ├── networks/         # Graph structures (adjacency matrices, edge lists)
│   └── features/         # Engineered features for modeling
└── README.md
```

## Data Sources

### 1. Real Estate Transactions
**Primary Source:** Toronto Open Data  
**URL:** https://open.toronto.ca/dataset/wellbeing-toronto-housing/  
**Coverage:** 2018-2024  
**Granularity:** Forward Sortation Area (FSA)  
**Update Frequency:** Quarterly  
**Status:** ⏳ In Progress

**Fields of Interest:**
- `FSA`: Forward Sortation Area code
- `Year`: Transaction year
- `Average_Price`: Mean sale price
- `Median_Price`: Median sale price
- `Transaction_Count`: Number of sales

**Backup:** Rentals.ca API, Web scraping Realtor.ca

---

### 2. Transit Networks
**Primary Source:** OpenStreetMap (OSM)  
**URL:** https://www.openstreetmap.org/  
**API:** Overpass API via OSMnx library  
**Coverage:** GTA Region  
**Status:** 🔜 Not Started

**Data Retrieved:**
- Road network (drive, walk, bike)
- Transit station locations
- Transit routes

**Secondary Source:** GTFS Feeds  
- TTC: https://open.toronto.ca/dataset/ttc-routes-and-schedules/
- GO Transit: https://www.gotransit.com/en/trip-planning/plan-your-trip/gtfs

---

### 3. Development Activity
**Primary Source:** City of Toronto Building Permits  
**URL:** https://open.toronto.ca/dataset/building-permits-cleared-permits/  
**Coverage:** 2018-2024  
**Update Frequency:** Daily  
**Status:** 🔜 Not Started

**Fields of Interest:**
- `LATITUDE`, `LONGITUDE`: Location
- `ISSUED_DATE`: Permit issue date
- `PERMIT_TYPE`: Type of construction
- `PROPOSED_BUILDING_TYPE`: Residential, commercial, etc.
- `ESTIMATED_VALUE`: Construction cost

---

### 4. Amenities (Points of Interest)
**Primary Source:** OpenStreetMap via Overpass API  
**Coverage:** GTA Region  
**Status:** 🔜 Not Started

**Categories to Extract:**
- `amenity=school`: Schools
- `leisure=park`: Parks
- `shop=*`: Retail locations
- `amenity=restaurant`: Restaurants
- `public_transport=station`: Transit stations

---

### 5. Demographics
**Primary Source:** Statistics Canada 2021 Census  
**URL:** https://www12.statcan.gc.ca/census-recensement/  
**Granularity:** Dissemination Area (DA), aggregated to FSA  
**Status:** 🔜 Not Started

**Variables of Interest:**
- Population density
- Median household income
- Age distribution
- Education levels

---

## Data Processing Pipeline

1. **Collection** → `data/raw/`
2. **Cleaning** → Remove duplicates, handle missing values
3. **Spatial Aggregation** → Aggregate to FSA level
4. **Feature Engineering** → Calculate accessibility, amenity density, etc.
5. **Save Processed** → `data/processed/`

---

## Data Quality Checks

- [ ] No missing FSA codes
- [ ] Date ranges are consistent (2018-2024)
- [ ] No duplicate transactions
- [ ] Spatial coordinates are within GTA bounds
- [ ] All FSAs have minimum required features

---

## Status Legend

- ✅ Complete
- ⏳ In Progress
- 🔜 Not Started
- ❌ Blocked/Issue

---

**Last Updated:** 2024-11-06  
**Updated By:** Kyle Williamson (Data Engineer)