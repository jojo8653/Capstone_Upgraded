import pandas as pd
import numpy as np
import streamlit as st
from datetime import datetime, timedelta
import warnings
import base64
import os
from hotel_ml_availability import HotelAvailabilityPredictor
warnings.filterwarnings('ignore')

# Core configuration
CONFIG = {
    'emergency_radius_km': 150,
    'safe_distance_km': 200,
    'price_surge_factor': 1.2,
    'worker_relocation_radius': 50,  # Closer to epicenter for crew efficiency
    'priority_booking_hours': 72,
}

# Country code to full name mapping - COMPREHENSIVE VERSION
COUNTRY_MAPPING = {
    # Europe
    'PRT': 'Portugal', 'GBR': 'United Kingdom', 'FRA': 'France', 'ESP': 'Spain', 
    'DEU': 'Germany', 'ITA': 'Italy', 'IRL': 'Ireland', 'BEL': 'Belgium',
    'NLD': 'Netherlands', 'CHE': 'Switzerland', 'AUT': 'Austria', 'POL': 'Poland', 
    'NOR': 'Norway', 'SWE': 'Sweden', 'DNK': 'Denmark', 'FIN': 'Finland',
    'CZE': 'Czech Republic', 'HUN': 'Hungary', 'GRC': 'Greece', 'TUR': 'Turkey',
    'ROU': 'Romania', 'BGR': 'Bulgaria', 'HRV': 'Croatia', 'SVN': 'Slovenia',
    'SVK': 'Slovakia', 'LTU': 'Lithuania', 'LVA': 'Latvia', 'EST': 'Estonia',
    'RUS': 'Russia', 'UKR': 'Ukraine', 'BLR': 'Belarus', 'MDA': 'Moldova',
    'ALB': 'Albania', 'MKD': 'North Macedonia', 'MNE': 'Montenegro', 'SRB': 'Serbia',
    'BIH': 'Bosnia and Herzegovina', 'LUX': 'Luxembourg', 'MLT': 'Malta', 'CYP': 'Cyprus',
    'ISL': 'Iceland', 'AND': 'Andorra', 'MCO': 'Monaco', 'SMR': 'San Marino',
    'VAT': 'Vatican City', 'LIE': 'Liechtenstein',
    
    # Americas
    'USA': 'United States', 'CAN': 'Canada', 'MEX': 'Mexico', 'BRA': 'Brazil',
    'ARG': 'Argentina', 'CHL': 'Chile', 'COL': 'Colombia', 'PER': 'Peru',
    'VEN': 'Venezuela', 'URY': 'Uruguay', 'PRY': 'Paraguay', 'BOL': 'Bolivia',
    'ECU': 'Ecuador', 'GUY': 'Guyana', 'SUR': 'Suriname', 'GUF': 'French Guiana',
    'CRI': 'Costa Rica', 'PAN': 'Panama', 'NIC': 'Nicaragua', 'HND': 'Honduras',
    'GTM': 'Guatemala', 'BLZ': 'Belize', 'SLV': 'El Salvador', 'CUB': 'Cuba',
    'JAM': 'Jamaica', 'HTI': 'Haiti', 'DOM': 'Dominican Republic', 'PRI': 'Puerto Rico',
    'TTO': 'Trinidad and Tobago', 'BRB': 'Barbados', 'GRD': 'Grenada', 'LCA': 'Saint Lucia',
    'VCT': 'Saint Vincent and the Grenadines', 'ATG': 'Antigua and Barbuda', 'DMA': 'Dominica',
    'KNA': 'Saint Kitts and Nevis', 'BHS': 'Bahamas',
    
    # Asia
    'CHN': 'China', 'JPN': 'Japan', 'KOR': 'South Korea', 'IND': 'India',
    'IDN': 'Indonesia', 'THA': 'Thailand', 'VNM': 'Vietnam', 'PHL': 'Philippines',
    'MYS': 'Malaysia', 'SGP': 'Singapore', 'MMR': 'Myanmar', 'KHM': 'Cambodia',
    'LAO': 'Laos', 'NPL': 'Nepal', 'BTN': 'Bhutan', 'LKA': 'Sri Lanka',
    'MDV': 'Maldives', 'BGD': 'Bangladesh', 'PAK': 'Pakistan', 'AFG': 'Afghanistan',
    'IRN': 'Iran', 'IRQ': 'Iraq', 'SYR': 'Syria', 'LBN': 'Lebanon', 'JOR': 'Jordan',
    'ISR': 'Israel', 'PSE': 'Palestine', 'SAU': 'Saudi Arabia', 'ARE': 'UAE',
    'QAT': 'Qatar', 'KWT': 'Kuwait', 'BHR': 'Bahrain', 'OMN': 'Oman',
    'YEM': 'Yemen', 'GEO': 'Georgia', 'ARM': 'Armenia', 'AZE': 'Azerbaijan',
    'KAZ': 'Kazakhstan', 'UZB': 'Uzbekistan', 'TKM': 'Turkmenistan', 'TJK': 'Tajikistan',
    'KGZ': 'Kyrgyzstan', 'MNG': 'Mongolia', 'PRK': 'North Korea', 'TWN': 'Taiwan',
    'HKG': 'Hong Kong', 'MAC': 'Macau',
    
    # Africa
    'ZAF': 'South Africa', 'EGY': 'Egypt', 'MAR': 'Morocco', 'TUN': 'Tunisia',
    'DZA': 'Algeria', 'LBY': 'Libya', 'SDN': 'Sudan', 'SSD': 'South Sudan',
    'ETH': 'Ethiopia', 'KEN': 'Kenya', 'TZA': 'Tanzania', 'UGA': 'Uganda',
    'RWA': 'Rwanda', 'BDI': 'Burundi', 'SOM': 'Somalia', 'DJI': 'Djibouti',
    'ERI': 'Eritrea', 'NGA': 'Nigeria', 'GHA': 'Ghana', 'CIV': "Côte d'Ivoire",
    'SEN': 'Senegal', 'MLI': 'Mali', 'BFA': 'Burkina Faso', 'NER': 'Niger',
    'TCD': 'Chad', 'CAF': 'Central African Republic', 'CMR': 'Cameroon', 'GNQ': 'Equatorial Guinea',
    'GAB': 'Gabon', 'COG': 'Republic of the Congo', 'COD': 'Democratic Republic of the Congo',
    'AGO': 'Angola', 'ZMB': 'Zambia', 'ZWE': 'Zimbabwe', 'BWA': 'Botswana',
    'NAM': 'Namibia', 'LSO': 'Lesotho', 'SWZ': 'Eswatini', 'MWI': 'Malawi',
    'MOZ': 'Mozambique', 'MDG': 'Madagascar', 'MUS': 'Mauritius', 'SYC': 'Seychelles',
    'COM': 'Comoros', 'CPV': 'Cape Verde', 'STP': 'São Tomé and Príncipe', 'GIN': 'Guinea',
    'GNB': 'Guinea-Bissau', 'GMB': 'Gambia', 'SLE': 'Sierra Leone', 'LBR': 'Liberia',
    'MRT': 'Mauritania', 'ESH': 'Western Sahara', 'BEN': 'Benin', 'TGO': 'Togo',
    
    # Oceania
    'AUS': 'Australia', 'NZL': 'New Zealand', 'FJI': 'Fiji', 'PNG': 'Papua New Guinea',
    'NCL': 'New Caledonia', 'VUT': 'Vanuatu', 'SLB': 'Solomon Islands', 'TON': 'Tonga',
    'WSM': 'Samoa', 'KIR': 'Kiribati', 'TUV': 'Tuvalu', 'NRU': 'Nauru',
    'MHL': 'Marshall Islands', 'FSM': 'Micronesia', 'PLW': 'Palau', 'COK': 'Cook Islands',
    'NUE': 'Niue', 'TKL': 'Tokelau', 'PYF': 'French Polynesia', 'WLF': 'Wallis and Futuna',
    'GUM': 'Guam', 'ASM': 'American Samoa', 'MNP': 'Northern Mariana Islands',
    
    # Other territories and special cases
    'GRL': 'Greenland', 'FRO': 'Faroe Islands', 'ATF': 'French Southern Territories',
    'BVT': 'Bouvet Island', 'HMD': 'Heard Island and McDonald Islands', 'IOT': 'British Indian Ocean Territory',
    'SGS': 'South Georgia and the South Sandwich Islands', 'ATA': 'Antarctica',
    'CCK': 'Cocos Islands', 'CXR': 'Christmas Island', 'NFK': 'Norfolk Island',
    'PCN': 'Pitcairn Islands', 'SHN': 'Saint Helena', 'TCA': 'Turks and Caicos Islands',
    'VGB': 'British Virgin Islands', 'VIR': 'U.S. Virgin Islands', 'CYM': 'Cayman Islands',
    'MSR': 'Montserrat', 'AIA': 'Anguilla', 'BMU': 'Bermuda', 'SPM': 'Saint Pierre and Miquelon',
    'REU': 'Réunion', 'MYT': 'Mayotte', 'BLM': 'Saint Barthélemy', 'MAF': 'Saint Martin',
    'ABW': 'Aruba', 'CUW': 'Curaçao', 'SXM': 'Sint Maarten', 'BES': 'Caribbean Netherlands',
    'GIB': 'Gibraltar', 'IMN': 'Isle of Man', 'JEY': 'Jersey', 'GGY': 'Guernsey',
    'ALA': 'Åland Islands', 'SJM': 'Svalbard and Jan Mayen',
    
    # Additional codes that might appear
    'Unknown': 'Unknown', 'NULL': 'Unknown', 'nan': 'Unknown', '': 'Unknown'
}

# Large countries that could benefit from sub-regional simulation
LARGE_COUNTRIES = {
    'United States': ['East Coast', 'West Coast', 'Midwest', 'South', 'Southwest'],
    'Brazil': ['Southeast', 'Northeast', 'South', 'North', 'Central-West'],
    'China': ['Eastern', 'Northern', 'Southern', 'Western', 'Central'],
    'Russia': ['European Russia', 'Siberia', 'Far East', 'Urals', 'South'],
    'Canada': ['Eastern', 'Western', 'Central', 'Atlantic', 'Northern'],
    'Australia': ['New South Wales', 'Victoria', 'Queensland', 'Western Australia', 'South Australia'],
    'Germany': ['North', 'South', 'East', 'West'],
    'France': ['North', 'South', 'East', 'West', 'Central'],
    'Spain': ['North', 'South', 'East', 'Central'],
    'Italy': ['North', 'South', 'Central']
}

def load_logo():
    """Load logo image if available"""
    logo_path = "hoteloptix_logo.png"  # Your logo file name
    if os.path.exists(logo_path):
        with open(logo_path, "rb") as f:
            logo_data = base64.b64encode(f.read()).decode()
        return f"data:image/png;base64,{logo_data}"
    return None

@st.cache_data
def load_and_prepare_data():
    """Load and prepare hotel booking data with disaster response features"""
    try:
        df = pd.read_csv('hotel_bookings.csv')
    except FileNotFoundError:
        st.error("❌ hotel_bookings.csv file not found. Please upload the dataset.")
        return pd.DataFrame()
    
    # Clean data
    df = df.drop(columns=['company', 'agent', 'reservation_status_date'], errors='ignore')
    df['children'] = df['children'].fillna(0).astype(int)
    
    # Handle missing countries
    if df['country'].isna().any():
        modes = df.groupby('market_segment')['country'].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else 'Unknown')
        df['country'] = df.apply(lambda r: modes[r['market_segment']] if pd.isna(r['country']) else r['country'], axis=1)
    
    # Add readable country names - handle all cases including NaN
    df['country_name'] = df['country'].fillna('Unknown').astype(str).map(COUNTRY_MAPPING).fillna(df['country'].fillna('Unknown').astype(str))
    
    # Simulate sub-regions for large countries using market segments as proxy
    df['sub_region'] = df['country_name']  # Default to country name
    
    for country, regions in LARGE_COUNTRIES.items():
        country_mask = df['country_name'] == country
        if country_mask.sum() > 100:  # Only if enough data points
            # Use market segment + hotel type to create sub-regions
            country_data = df[country_mask]
            segments = country_data['market_segment'].unique()
            
            # Map segments to regions (simplified approach)
            for i, segment in enumerate(segments[:len(regions)]):
                segment_mask = (df['country_name'] == country) & (df['market_segment'] == segment)
                region_name = f"{country} - {regions[i % len(regions)]}"
                df.loc[segment_mask, 'sub_region'] = region_name
    
    # Create disaster response features
    df['total_nights'] = df['stays_in_week_nights'] + df['stays_in_weekend_nights']
    df['total_guests'] = df[['adults','children','babies']].sum(axis=1)
    df['emergency_booking'] = (df['lead_time'] <= 7).astype(int)
    
    # Date features
    df['arrival_date'] = pd.to_datetime(
        df['arrival_date_year'].astype(str) + '-' + 
        df['arrival_date_month'].str[:3] + '-' + 
        df['arrival_date_day_of_month'].astype(str),
        format='%Y-%b-%d', errors='coerce'
    )
    df['arrival_month'] = df['arrival_date'].dt.month
    
    # Disaster season risk
    df['disaster_season_risk'] = df['arrival_month'].map({
        6: 3, 7: 3, 8: 3, 9: 3, 10: 2,  # Hurricane season
        3: 2, 4: 2, 5: 2,  # Flood season
        11: 1, 12: 1, 1: 1, 2: 1  # Low risk
    })
    
    # Emergency priority score
    df['emergency_priority'] = (
        (df['total_guests'] >= 4) * 3 +  # Families get priority
        (df['babies'] > 0) * 2 +         # Families with babies
        (df['total_nights'] >= 7) * 1 +  # Long stays
        (df['adr'] >= df['adr'].quantile(0.8)) * 1  # High-value guests
    )
    
    return df

def simulate_disaster_impact(df, disaster_location, disaster_type, severity):
    """Simulate disaster impact based on user inputs - DYNAMIC VERSION"""
    df = df.copy()
    
    # Base impact rates by disaster type
    disaster_impact_rates = {
        'Hurricane': {'base': 0.15, 'season_months': [6, 7, 8, 9, 10]},
        'Flood': {'base': 0.12, 'season_months': [3, 4, 5, 6]},
        'Earthquake': {'base': 0.08, 'season_months': []},  # No seasonal pattern
        'Wildfire': {'base': 0.10, 'season_months': [7, 8, 9]},
        'Tornado': {'base': 0.06, 'season_months': [4, 5, 6]}
    }
    
    # Get impact rate for selected disaster type
    base_rate = disaster_impact_rates.get(disaster_type, {'base': 0.10, 'season_months': []})['base']
    season_months = disaster_impact_rates.get(disaster_type, {'base': 0.10, 'season_months': []})['season_months']
    
    # Severity multiplier (1-5 scale)
    severity_multiplier = severity / 3.0  # Scale so 3 = normal, 5 = extreme
    
    # Calculate impact probability
    df['disaster_affected'] = 0
    
    # Check if location is a sub-region or country
    if ' - ' in disaster_location:
        # Sub-region selected
        location_mask = df['sub_region'] == disaster_location
    else:
        # Country selected
        location_mask = df['country_name'] == disaster_location
    
    if season_months:
        # Higher impact during disaster season
        season_mask = df['arrival_month'].isin(season_months)
        high_season_rate = base_rate * severity_multiplier * 1.5
        low_season_rate = base_rate * severity_multiplier * 0.5
        
        df.loc[location_mask & season_mask, 'disaster_affected'] = np.random.binomial(
            1, min(high_season_rate, 0.8), sum(location_mask & season_mask)
        )
        df.loc[location_mask & ~season_mask, 'disaster_affected'] = np.random.binomial(
            1, min(low_season_rate, 0.4), sum(location_mask & ~season_mask)
        )
    else:
        # No seasonal pattern
        impact_rate = base_rate * severity_multiplier
        df.loc[location_mask, 'disaster_affected'] = np.random.binomial(
            1, min(impact_rate, 0.7), sum(location_mask)
        )
    
    # Secondary impact: neighboring regions (reduced impact)
    # For sub-regions, affect other sub-regions in same country
    if ' - ' in disaster_location:
        country_name = disaster_location.split(' - ')[0]
        other_regions_mask = df['sub_region'].str.startswith(country_name) & (df['sub_region'] != disaster_location)
        secondary_rate = base_rate * severity_multiplier * 0.4
        df.loc[other_regions_mask, 'disaster_affected'] = np.random.binomial(
            1, min(secondary_rate, 0.3), sum(other_regions_mask)
        )
    
    return df

def find_emergency_alternatives(df, disaster_location, guest_requirements, ml_predictor=None):
    """Find alternative hotels for emergency rebooking with ML-powered availability prediction"""
    
    # Determine if we're dealing with a sub-region or country
    if ' - ' in disaster_location:
        # Sub-region: exclude the specific sub-region but allow other regions in same country
        country_name = disaster_location.split(' - ')[0]
        safe_hotels = df[df['sub_region'] != disaster_location].copy()
        # Prioritize other regions in same country for easier logistics
        same_country_mask = df['sub_region'].str.startswith(country_name) & (df['sub_region'] != disaster_location)
        safe_hotels['same_country_priority'] = same_country_mask.astype(int)
    else:
        # Country: exclude entire country
        safe_hotels = df[df['country_name'] != disaster_location].copy()
        safe_hotels['same_country_priority'] = 0  # All are different countries
    
    # Filter by guest requirements
    if guest_requirements.get('total_guests'):
        safe_hotels = safe_hotels[safe_hotels['total_guests'] >= guest_requirements['total_guests']]
    
    if guest_requirements.get('total_nights'):
        safe_hotels = safe_hotels[safe_hotels['total_nights'] >= guest_requirements['total_nights']]
    
    # Calculate availability score
    hotel_availability = safe_hotels.groupby(['country_name', 'sub_region', 'hotel']).agg({
        'is_canceled': 'mean',
        'adr': 'mean',
        'total_guests': 'count',
        'same_country_priority': 'first'
    }).reset_index()
    
    # Traditional availability score (fallback)
    hotel_availability['traditional_availability'] = (
        hotel_availability['is_canceled'] * 0.6 +  # Higher cancellations = more availability
        (1 - hotel_availability['total_guests'] / hotel_availability['total_guests'].max()) * 0.3 +
        hotel_availability['same_country_priority'] * 0.1  # Slight preference for same country
    )
    
    # ML-enhanced availability prediction
    if ml_predictor and ml_predictor.rf_model is not None:
        hotel_availability['ml_availability'] = 0.5  # Default
        hotel_availability['ml_confidence'] = 'low'
        hotel_availability['model_predictions'] = None
        
        for idx, hotel_row in hotel_availability.iterrows():
            # Get hotel-specific data for ML prediction
            hotel_data = safe_hotels[
                (safe_hotels['hotel'] == hotel_row['hotel']) & 
                (safe_hotels['country_name'] == hotel_row['country_name'])
            ]
            
            if len(hotel_data) > 0:
                try:
                    prediction = ml_predictor.predict_availability(hotel_data)
                    hotel_availability.at[idx, 'ml_availability'] = prediction['availability_score']
                    hotel_availability.at[idx, 'ml_confidence'] = prediction['confidence']
                    hotel_availability.at[idx, 'model_predictions'] = str(prediction['model_predictions'])
                except Exception as e:
                    # Fallback to traditional method
                    hotel_availability.at[idx, 'ml_availability'] = hotel_availability.at[idx, 'traditional_availability']
        
        # Combine traditional and ML scores
        hotel_availability['availability_score'] = (
            hotel_availability['traditional_availability'] * 0.3 +
            hotel_availability['ml_availability'] * 0.7
        )
        
        # Boost score for high confidence predictions
        confidence_boost = hotel_availability['ml_confidence'].map({
            'high': 0.1, 'medium': 0.05, 'low': 0.0
        })
        hotel_availability['availability_score'] += confidence_boost
    else:
        # Use traditional scoring if ML models not available
        hotel_availability['availability_score'] = hotel_availability['traditional_availability']
        hotel_availability['ml_availability'] = hotel_availability['traditional_availability']
        hotel_availability['ml_confidence'] = 'traditional'
    
    # Emergency pricing
    emergency_hotels = hotel_availability[hotel_availability['availability_score'] > 0.3].copy()
    emergency_hotels['emergency_price'] = emergency_hotels['adr'] * CONFIG['price_surge_factor']
    
    return emergency_hotels.sort_values(['same_country_priority', 'availability_score'], ascending=[False, False])

def relocate_service_workers(df, disaster_location, worker_requirements):
    """Find accommodation for displaced service workers - CENTRALIZED APPROACH"""
    
    # Determine if we're dealing with a sub-region or country
    if ' - ' in disaster_location:
        # Sub-region: look for nearby regions within same country first (centralized approach)
        country_name = disaster_location.split(' - ')[0]
        # Priority 1: Other regions in same country (easier logistics, centralized)
        nearby_areas = df[df['sub_region'].str.startswith(country_name) & (df['sub_region'] != disaster_location)].copy()
        nearby_areas['proximity_score'] = 2  # High proximity
        
        # Priority 2: Neighboring countries if needed
        other_areas = df[~df['sub_region'].str.startswith(country_name)].copy()
        other_areas['proximity_score'] = 1  # Lower proximity
        
        safe_areas = pd.concat([nearby_areas, other_areas])
    else:
        # Country: exclude entire country, find closest alternatives
        safe_areas = df[df['country_name'] != disaster_location].copy()
        safe_areas['proximity_score'] = 1
    
    # Filter for worker-suitable options (budget-friendly, longer stays)
    worker_suitable = safe_areas[
        (safe_areas['adr'] <= safe_areas['adr'].quantile(0.4)) &  # Budget-friendly (bottom 40%)
        (safe_areas['total_nights'] >= 7)  # Longer stays suitable for workers
    ]
    
    # Group by location and calculate suitability
    worker_accommodations = worker_suitable.groupby(['country_name', 'sub_region', 'hotel']).agg({
        'adr': 'mean',
        'is_canceled': 'mean',
        'total_guests': 'count',
        'proximity_score': 'first'
    }).reset_index()
    
    # Enhanced suitability score emphasizing proximity to disaster area for crew efficiency
    worker_accommodations['worker_suitability'] = (
        (worker_accommodations['adr'] <= worker_accommodations['adr'].quantile(0.5)) * 0.3 +  # Cost efficiency
        worker_accommodations['is_canceled'] * 0.3 +  # Availability
        (worker_accommodations['total_guests'] / worker_accommodations['total_guests'].max()) * 0.2 +  # Capacity
        worker_accommodations['proximity_score'] * 0.2  # Proximity for crew efficiency
    )
    
    return worker_accommodations.sort_values(['proximity_score', 'worker_suitability'], ascending=[False, False])

def calculate_evacuation_priority(bookings_df):
    """Calculate evacuation priority for affected bookings"""
    
    priority_bookings = bookings_df[bookings_df['disaster_affected'] == 1].copy()
    
    if len(priority_bookings) == 0:
        return pd.DataFrame()
    
    # Priority scoring
    priority_bookings['evacuation_priority'] = (
        priority_bookings['emergency_priority'] * 0.4 +
        (priority_bookings['babies'] > 0) * 3 +
        (priority_bookings['total_guests'] >= 4) * 2 +
        (priority_bookings['adr'] >= priority_bookings['adr'].quantile(0.8)) * 1
    )
    
    return priority_bookings.sort_values('evacuation_priority', ascending=False)

def main():
    st.set_page_config(
        page_title="HotelOptix Disaster Response Tool",
        page_icon="🏨",
        layout="wide",
        initial_sidebar_state="expanded"
    )
    
    # HotelOptix Brand Styling
    st.markdown("""
    <style>
    /* HotelOptix Brand Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700;800&display=swap');
    @import url('https://fonts.googleapis.com/css2?family=Source+Sans+Pro:wght@400;500;600&display=swap');
    
    /* Try to load custom fonts - you can add font files to your repository */
    @font-face {
        font-family: 'Futura Maxi CG';
        src: url('fonts/FuturaMaxiCGBold.woff2') format('woff2'),
             url('fonts/FuturaMaxiCGBold.woff') format('woff');
        font-weight: bold;
        font-display: swap;
    }
    
    @font-face {
        font-family: 'Telegraf';
        src: url('fonts/Telegraf-Regular.woff2') format('woff2'),
             url('fonts/Telegraf-Regular.woff') format('woff');
        font-weight: normal;
        font-display: swap;
    }
    
    /* Custom styling */
    .main-header {
        display: flex;
        align-items: center;
        gap: 20px;
        margin-bottom: 30px;
        padding: 20px 0;
        border-bottom: 2px solid #FACC00;
    }
    
    .logo-container {
        flex-shrink: 0;
    }
    
    .title-container {
        flex-grow: 1;
    }
    
    /* HotelOptix Typography */
    .custom-title {
        font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Exact brand font with fallback */
        font-size: 2.5rem;
        font-weight: bold;
        color: #000053;  /* OPTIX BLUE */
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .custom-subtitle {
        font-family: 'Telegraf', 'Source Sans Pro', sans-serif;  /* Exact brand font with fallback */
        font-size: 1.1rem;
        color: #666666;
        margin: 5px 0;
        font-weight: 500;
    }
    
    /* HotelOptix Brand Colors */
    :root {
        --optix-blue: #000053;      /* Primary - OPTIX BLUE */
        --sunset-yellow: #FACC00;   /* Secondary - SUNSET */
        --burnt-orange: #F08301;    /* Accent - BURNT ORANGE */
        --white: #FFFFFF;           /* White */
        --text-color: #000053;      /* Use brand blue for text */
        --bg-color: #FFFFFF;        /* Clean white background */
        --light-gray: #F5F5F5;      /* For cards */
    }
    
    /* HotelOptix Metric Cards */
    .metrics-container {
        margin: 20px 0;
    }
    
    .metric-card {
        background: var(--white);
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 83, 0.1);
        border-left: 5px solid var(--sunset-yellow);
        border-top: 1px solid var(--burnt-orange);
        margin-bottom: 15px;
        transition: all 0.3s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 83, 0.15);
        border-left-color: var(--burnt-orange);
    }
    
    .metric-title {
        font-family: 'Telegraf', 'Source Sans Pro', sans-serif;  /* Exact brand font */
        font-size: 0.9rem;
        color: #666666;
        margin-bottom: 8px;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-value {
        font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Exact brand font */
        font-size: 2.2rem;
        font-weight: bold;
        color: var(--optix-blue);
        line-height: 1;
        margin-bottom: 5px;
    }
    
    .metric-delta {
        font-family: 'Telegraf', 'Source Sans Pro', sans-serif;  /* Exact brand font */
        font-size: 0.8rem;
        color: var(--burnt-orange);
        font-weight: 500;
    }
    
    /* HotelOptix Tab Styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
    }
    
    .stTabs [data-baseweb="tab"] {
        font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Headers use Futura */
        font-weight: bold;
        color: var(--optix-blue);
        background-color: var(--light-gray);
        border-radius: 8px 8px 0 0;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: var(--sunset-yellow) !important;
        color: var(--optix-blue) !important;
    }
    
    /* HotelOptix Button Styling */
    .stButton > button {
        font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Headers use Futura */
        font-weight: bold;
        background-color: var(--optix-blue);
        color: var(--white);
        border: none;
        border-radius: 8px;
        transition: all 0.3s ease;
    }
    
         .stButton > button:hover {
         background-color: var(--burnt-orange);
         transform: translateY(-1px);
         box-shadow: 0 4px 8px rgba(240, 131, 1, 0.3);
     }
     
     /* HotelOptix Sidebar Styling */
     .css-1d391kg {
         background-color: var(--light-gray);
     }
     
     .stSelectbox label, .stSlider label {
         font-family: 'Telegraf', 'Source Sans Pro', sans-serif;  /* Subheaders use Telegraf */
         font-weight: 600;
         color: var(--optix-blue);
     }
     
     /* HotelOptix Headers */
     h1, h2, h3 {
         font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Headers use Futura */
         color: var(--optix-blue);
         font-weight: bold;
     }
     
     /* HotelOptix Paragraphs and Body Text */
     p, .stMarkdown, .stText, .streamlit-expanderContent {
         font-family: 'Telegraf', 'Source Sans Pro', sans-serif;  /* Body text uses Telegraf */
     }
     
     /* HotelOptix Dataframes */
     .stDataFrame {
         border: 2px solid var(--sunset-yellow);
         border-radius: 10px;
         font-family: 'Telegraf', 'Source Sans Pro', sans-serif;
     }
     
     /* HotelOptix Expanders */
     .streamlit-expanderHeader {
         font-family: 'Futura Maxi CG', 'Montserrat', sans-serif;  /* Headers use Futura */
         font-weight: bold;
         color: var(--optix-blue);
         background-color: var(--light-gray);
     }
     </style>
     """, unsafe_allow_html=True)
    
    # Header with logo and title
    logo_data = load_logo()
    
    if logo_data:
        # Use actual logo - Updated to larger size
        st.markdown(f"""
        <div class="main-header">
            <div class="logo-container">
                <img src="{logo_data}" style="width: 180px; height: 180px; object-fit: contain;">
            </div>
            <div class="title-container">
                <h1 class="custom-title">HotelOptix Disaster Response Tool</h1>
                <p class="custom-subtitle">Professional Emergency Rebooking & Service Worker Relocation Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        # HotelOptix Branded Placeholder Logo
        st.markdown("""
        <div class="main-header">
            <div class="logo-container">
                <div style="width: 180px; height: 180px; background: white; border: 3px solid #000053; border-radius: 20px; display: flex; align-items: center; justify-content: center; color: #000053; font-size: 48px; font-weight: bold; font-family: 'Futura Maxi CG', 'Montserrat', sans-serif; box-shadow: 0 8px 20px rgba(0, 0, 83, 0.3);">
                    <div style="text-align: center;">
                        <div style="font-size: 48px; line-height: 1;">H</div>
                        <div style="font-size: 12px; font-weight: normal; margin-top: -5px; letter-spacing: 2px;">OPTIX</div>
                    </div>
                </div>
            </div>
            <div class="title-container">
                <h1 class="custom-title">HotelOptix Disaster Response Tool</h1>
                <p class="custom-subtitle">Professional Emergency Rebooking & Service Worker Relocation Platform</p>
            </div>
        </div>
        """, unsafe_allow_html=True)
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_and_prepare_data()
        
        if df.empty:
            st.stop()
    
    # Initialize ML predictor
    if 'ml_predictor' not in st.session_state:
        st.session_state.ml_predictor = HotelAvailabilityPredictor()
        # Try to load pre-trained models
        try:
            st.session_state.ml_predictor.load_models()
            if st.session_state.ml_predictor.rf_model is not None:
                st.success("✅ Pre-trained ML models loaded successfully!")
        except:
            pass
    
    # Sidebar for disaster simulation
    st.sidebar.header("🚨 Disaster Scenario Setup")
    
    disaster_type = st.sidebar.selectbox(
        "Disaster Type",
        ["Hurricane", "Flood", "Earthquake", "Wildfire", "Tornado"]
    )
    
    # Create location options with sub-regions for large countries
    location_options = []
    countries_with_regions = set()
    
    # Add sub-regions for large countries first
    for sub_region in sorted(df['sub_region'].unique()):
        if ' - ' in sub_region:  # This is a sub-region
            location_options.append(sub_region)
            countries_with_regions.add(sub_region.split(' - ')[0])
    
    # Add countries that don't have sub-regions
    for country in sorted(df['country_name'].unique()):
        if country not in countries_with_regions:
            location_options.append(country)
    
    affected_location = st.sidebar.selectbox(
        "Affected Location",
        location_options,
        help="Select a country or specific region within large countries"
    )
    
    severity = st.sidebar.slider("Disaster Severity (1-5)", 1, 5, 3,
                                help="1=Minor, 3=Moderate, 5=Catastrophic")

    # Show location info
    if ' - ' in affected_location:
        country_name = affected_location.split(' - ')[0]
        region_name = affected_location.split(' - ')[1]
        st.sidebar.info(f"📍 **Region**: {region_name}\n\n🌍 **Country**: {country_name}")
    else:
        bookings_count = len(df[df['country_name'] == affected_location])
        st.sidebar.info(f"🌍 **Country**: {affected_location}\n\n📊 **Total Bookings**: {bookings_count:,}")
    
    # Simulate disaster impact based on user inputs
    df = simulate_disaster_impact(df, affected_location, disaster_type, severity)
    
    # Main tabs
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "📊 Dashboard Overview", 
        "🏨 Emergency Rebooking", 
        "👷 Worker Relocation",
        "📈 Analytics",
        "🤖 ML Models"
    ])
    
    with tab1:
        st.header("Disaster Response Dashboard Overview")
        
        # Key metrics with custom styling
        st.markdown('<div class="metrics-container">', unsafe_allow_html=True)
        col1, col2, col3, col4 = st.columns(4)
        
        affected_bookings = df[df['disaster_affected'] == 1]
        
        with col1:
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Affected Bookings</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">Severity {} in {}</div>
            </div>
            """.format(len(affected_bookings), severity, affected_location), unsafe_allow_html=True)
        
        with col2:
            emergency_bookings = affected_bookings[affected_bookings['emergency_booking'] == 1]
            percentage = f"{len(emergency_bookings)/len(affected_bookings)*100:.1f}%" if len(affected_bookings) > 0 else "0%"
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Emergency Bookings</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">{} of affected</div>
            </div>
            """.format(len(emergency_bookings), percentage), unsafe_allow_html=True)
        
        with col3:
            high_priority = affected_bookings[affected_bookings['emergency_priority'] >= 3]
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">High Priority Cases</div>
                <div class="metric-value">{}</div>
                <div class="metric-delta">Families & VIP guests</div>
            </div>
            """.format(len(high_priority)), unsafe_allow_html=True)
        
        with col4:
            avg_nights = affected_bookings['total_nights'].mean() if len(affected_bookings) > 0 else 0
            st.markdown("""
            <div class="metric-card">
                <div class="metric-title">Avg. Nights Affected</div>
                <div class="metric-value">{:.1f}</div>
                <div class="metric-delta">Per booking</div>
            </div>
            """.format(avg_nights), unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        # Impact analysis
        st.subheader("Impact by Country")
        
        impact_by_country = df.groupby('country_name').agg({
            'disaster_affected': 'sum',
            'adr': 'mean',
            'total_guests': 'sum'
        }).reset_index()
        
        st.dataframe(
            impact_by_country.sort_values('disaster_affected', ascending=False).head(10),
            use_container_width=True
        )
    
    with tab2:
        st.header("🏨 Emergency Rebooking System")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Guest Requirements")
            
            guest_total = st.number_input("Total Guests", min_value=1, max_value=10, value=2)
            guest_nights = st.number_input("Total Nights", min_value=1, max_value=30, value=3)
            budget_max = st.number_input("Maximum Budget per Night", min_value=50, max_value=500, value=150)
            
            guest_requirements = {
                'total_guests': guest_total,
                'total_nights': guest_nights,
                'budget_max': budget_max
            }
            
            if st.button("🔍 Find Emergency Alternatives", type="primary"):
                with st.spinner("Searching for available alternatives..."):
                    alternatives = find_emergency_alternatives(df, affected_location, guest_requirements, st.session_state.ml_predictor)
                    st.session_state['alternatives'] = alternatives
        
        with col1:
            if 'alternatives' in st.session_state:
                st.subheader("Available Emergency Accommodations")
                
                alternatives = st.session_state['alternatives'].head(10)
                
                if len(alternatives) == 0:
                    st.warning("No suitable alternatives found for the given criteria.")
                else:
                    for idx, hotel in alternatives.iterrows():
                        # Create header with ML confidence indicator
                        confidence_emoji = {"high": "🟢", "medium": "🟡", "low": "🔴", "traditional": "⚪"}
                        confidence_indicator = confidence_emoji.get(hotel.get('ml_confidence', 'traditional'), "⚪")
                        
                        with st.expander(f"🏨 {hotel['hotel']} in {hotel['country_name']} - Availability: {hotel['availability_score']:.2f} {confidence_indicator}"):
                            col_a, col_b, col_c, col_d = st.columns(4)
                            
                            with col_a:
                                st.metric("Emergency Price", f"${hotel['emergency_price']:.0f}/night")
                            
                            with col_b:
                                st.metric("Availability Score", f"{hotel['availability_score']:.2f}")
                            
                            with col_c:
                                st.metric("Historical Cancellations", f"{hotel['is_canceled']*100:.1f}%")
                            
                            with col_d:
                                confidence = hotel.get('ml_confidence', 'traditional')
                                st.metric("ML Confidence", confidence.title())
                            
                            # Show ML model predictions if available
                            if hotel.get('model_predictions') and hotel['model_predictions'] != 'None':
                                st.info(f"🤖 **ML Predictions**: {hotel['model_predictions']}")
                            
                            if st.button(f"📞 Contact Hotel {hotel['hotel']}", key=f"contact_{idx}"):
                                st.success(f"Emergency booking request sent to {hotel['hotel']} in {hotel['country_name']}")
    
    with tab3:
        st.header("👷 Service Worker Relocation")
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Worker Requirements")
            
            num_workers = st.number_input("Number of Workers", min_value=1, max_value=100, value=10)
            duration_days = st.number_input("Relocation Duration (days)", min_value=7, max_value=180, value=30)
            budget_per_worker = st.number_input("Budget per Worker per Night", min_value=30, max_value=150, value=60)
            
            worker_requirements = {
                'num_workers': num_workers,
                'duration_days': duration_days,
                'budget_per_worker': budget_per_worker
            }
            
            if st.button("🔍 Find Worker Accommodations", type="primary"):
                with st.spinner("Finding suitable worker accommodations..."):
                    worker_options = relocate_service_workers(df, affected_location, worker_requirements)
                    st.session_state['worker_options'] = worker_options
        
        with col1:
            if 'worker_options' in st.session_state:
                st.subheader("Worker Accommodation Options")
                
                worker_options = st.session_state['worker_options'].head(8)
                
                if len(worker_options) == 0:
                    st.warning("No suitable worker accommodations found.")
                else:
                    for idx, accommodation in worker_options.iterrows():
                        with st.expander(f"🏠 {accommodation['hotel']} in {accommodation['country_name']} - Suitability: {accommodation['worker_suitability']:.2f}"):
                            col_a, col_b, col_c = st.columns(3)
                            
                            with col_a:
                                st.metric("Rate per Night", f"${accommodation['adr']:.0f}")
                            
                            with col_b:
                                total_cost = accommodation['adr'] * duration_days * num_workers
                                st.metric("Total Cost", f"${total_cost:,.0f}")
                            
                            with col_c:
                                st.metric("Suitability Score", f"{accommodation['worker_suitability']:.2f}")
                            
                            if st.button(f"📋 Reserve for Workers", key=f"worker_{idx}"):
                                st.success(f"Reservation request sent for {num_workers} workers at {accommodation['hotel']}")
    
    with tab4:
        st.header("📈 Disaster Response Analytics")
        
        # Evacuation priority analysis
        priority_bookings = calculate_evacuation_priority(df)
        
        if len(priority_bookings) > 0:
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("High Priority Evacuations")
                high_priority = priority_bookings.head(10)[['country_name', 'sub_region', 'hotel', 'total_guests', 'babies', 'evacuation_priority']]
                st.dataframe(high_priority, use_container_width=True)
            
            with col2:
                st.subheader("Priority Distribution")
                priority_counts = priority_bookings['evacuation_priority'].value_counts().sort_index()
                
                # Simple bar chart using dataframe
                priority_df = pd.DataFrame({
                    'Priority Score': priority_counts.index,
                    'Number of Bookings': priority_counts.values
                })
                st.bar_chart(priority_df.set_index('Priority Score'))
        
        # Financial impact analysis
        st.subheader("Financial Impact Analysis")
        
        affected_bookings = df[df['disaster_affected'] == 1]
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            revenue_loss = affected_bookings['adr'].sum()
            st.metric("Potential Revenue Loss", f"${revenue_loss:,.0f}")
        
        with col2:
            rebooking_cost = revenue_loss * CONFIG['price_surge_factor']
            st.metric("Emergency Rebooking Cost", f"${rebooking_cost:,.0f}")
        
        with col3:
            net_impact = rebooking_cost - revenue_loss
            st.metric("Net Financial Impact", f"${net_impact:,.0f}", 
                     delta=f"{(net_impact/revenue_loss)*100:.1f}%" if revenue_loss > 0 else "0%")
        
        # Summary statistics
        st.subheader("Disaster Impact Summary")
        
        summary_stats = pd.DataFrame({
            'Metric': [
                'Total Affected Bookings',
                'Average ADR (Affected)',
                'Total Guests Affected',
                'Emergency Bookings',
                'High Priority Cases',
                'Affected Location',
                'Geographic Scope'
            ],
            'Value': [
                len(affected_bookings),
                f"${affected_bookings['adr'].mean():.2f}" if len(affected_bookings) > 0 else "$0",
                affected_bookings['total_guests'].sum(),
                len(affected_bookings[affected_bookings['emergency_booking'] == 1]),
                len(affected_bookings[affected_bookings['emergency_priority'] >= 3]),
                affected_location,
                "Sub-Regional" if ' - ' in affected_location else "National"
            ]
        })
        
        st.dataframe(summary_stats, use_container_width=True)
        
        # Model information
        st.subheader("🤖 Analysis Methodology")
        st.info("""
        **HotelOptix Disaster Response Analytics**
        
        - **Geographic Precision**: Sub-regional analysis for large countries (US, Brazil, China, etc.)
        - **Emergency Priority Scoring**: Families with children, VIP guests, and long-stay bookings receive priority
        - **Availability Prediction**: Based on historical cancellation patterns and booking flexibility
        - **Pricing Strategy**: 20% surge pricing during emergency rebooking scenarios
        - **Centralized Worker Relocation**: Budget-optimized accommodation search prioritizing proximity to disaster zone
        - **Dynamic Impact Modeling**: Severity-adjusted disaster simulation with seasonal patterns
        
        *This tool provides data-driven insights for hotel emergency response planning with real-time scenario modeling.*
        """)
    
    with tab5:
        st.header("🤖 Machine Learning Models")
        
        ml_predictor = st.session_state.ml_predictor
        
        col1, col2 = st.columns([2, 1])
        
        with col2:
            st.subheader("Model Management")
            
            # Model training section
            st.markdown("### Train New Models")
            if st.button("🚀 Train ML Models", type="primary"):
                with st.spinner("Training machine learning models... This may take several minutes."):
                    try:
                        results = ml_predictor.train_models(df)
                        st.session_state['training_results'] = results
                        
                        # Save models after training
                        ml_predictor.save_models()
                        st.success("✅ Models trained and saved successfully!")
                        
                    except Exception as e:
                        st.error(f"❌ Training failed: {str(e)}")
            
            # Model status
            st.markdown("### Model Status")
            models_status = {
                "Random Forest": "✅ Loaded" if ml_predictor.rf_model is not None else "❌ Not Available",
                "LightGBM": "✅ Loaded" if ml_predictor.lightgbm_model is not None else "❌ Not Available", 
                "LSTM": "✅ Loaded" if ml_predictor.lstm_model is not None else "❌ Not Available"
            }
            
            for model, status in models_status.items():
                st.write(f"**{model}**: {status}")
            
            # Load existing models
            if st.button("📂 Load Pre-trained Models"):
                try:
                    ml_predictor.load_models()
                    st.success("✅ Models loaded successfully!")
                    st.experimental_rerun()
                except Exception as e:
                    st.error(f"❌ Loading failed: {str(e)}")
        
        with col1:
            st.subheader("Model Performance")
            
            # Show training results if available
            if 'training_results' in st.session_state:
                results = st.session_state['training_results']
                
                st.markdown("### Model Evaluation Metrics")
                
                # Create performance comparison table
                if results:
                    performance_df = pd.DataFrame(results).T
                    st.dataframe(performance_df, use_container_width=True)
                    
                    # Best model selection
                    if len(performance_df) > 0:
                        best_model = performance_df['R²'].idxmax()
                        st.success(f"🏆 **Best performing model**: {best_model} (R² = {performance_df.loc[best_model, 'R²']:.3f})")
            
            # Feature importance
            st.markdown("### Feature Importance")
            
            if ml_predictor.rf_model is not None or ml_predictor.lightgbm_model is not None:
                importance_data = ml_predictor.get_feature_importance()
                
                if importance_data:
                    for model_name, importances in importance_data.items():
                        st.markdown(f"#### {model_name}")
                        
                        # Convert to DataFrame and sort
                        importance_df = pd.DataFrame(
                            list(importances.items()), 
                            columns=['Feature', 'Importance']
                        ).sort_values('Importance', ascending=False).head(10)
                        
                        st.bar_chart(importance_df.set_index('Feature'))
            else:
                st.info("Train models to see feature importance analysis.")
        
        # Model explanation
        st.markdown("---")
        st.subheader("🔍 How the ML Models Work")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **🌳 Random Forest**
            - **Purpose**: Baseline ensemble model
            - **Strengths**: Robust, interpretable
            - **Use Case**: Feature importance analysis
            - **Input**: Tabular hotel features
            """)
        
        with col2:
            st.markdown("""
            **🚀 LightGBM**
            - **Purpose**: Advanced gradient boosting
            - **Strengths**: High accuracy, fast training
            - **Use Case**: Primary availability prediction
            - **Input**: Engineered features
            """)
        
        with col3:
            st.markdown("""
            **🔗 LSTM Neural Network**
            - **Purpose**: Time series patterns
            - **Strengths**: Temporal dependencies
            - **Use Case**: Seasonal availability trends
            - **Input**: 14-day sequences
            """)
        
        st.markdown("""
        ### 🎯 Prediction Target: Hotel Availability Score (0-1)
        
        The models predict a **real-time availability score** for each hotel:
        - **1.0** = Fully available (high cancellation rates, low occupancy)
        - **0.5** = Moderate availability 
        - **0.0** = No availability (fully booked, low cancellations)
        
        **Key Features Used**:
        - 📅 Temporal patterns (seasonality, day of week, holidays)
        - 🏨 Hotel characteristics (location, pricing, capacity)
        - 📊 Historical booking patterns (cancellations, lead times)
        - 👥 Guest demographics (family size, stay duration)
        - 💰 Pricing dynamics (ADR, revenue patterns)
        """)

if __name__ == "__main__":
    main() 