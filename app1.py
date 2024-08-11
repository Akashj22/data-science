import streamlit as st
import pandas as pd
import numpy as np
import joblib
from datetime import datetime

# Load the model
model = joblib.load('model.pkl')

# Function to extract date features
def date_to_features(input_date):
    date = pd.to_datetime(input_date)
    
    features = {}
    features['year'] = date.year
    features['month'] = date.month
    features['day'] = date.day
    features['day_of_week'] = date.dayofweek
    features['month_sin'] = np.sin(2 * np.pi * date.month / 12)
    features['month_cos'] = np.cos(2 * np.pi * date.month / 12)
    features['day_of_week_sin'] = np.sin(2 * np.pi * date.dayofweek / 7)
    features['day_of_week_cos'] = np.cos(2 * np.pi * date.dayofweek / 7)
    features['is_weekend'] = 1 if date.dayofweek >= 5 else 0
    
    return features

# Streamlit app
st.title("Predictive Model")

# Input fields for the features
deviceCategory = st.selectbox('Device Category', ['desktop', 'mobile', 'tablet'])
operatingSystem = st.selectbox('Operating System', [
    'Windows', 'iOS', 'Linux', 'Android', 'Chrome OS', 'Macintosh', 'others'
])
browser = st.selectbox('Browser', [
    'Internet Explorer', 'Chrome', 'Safari', 'Firefox', 'others',
    'Safari (in-app)', 'Edge', 'Android Webview', 'Opera', 'UC Browser',
    'Opera Mini', 'YaBrowser'
])
country = st.selectbox('Country', [
    'United States', 'Australia', 'Italy', 'Malaysia', 'Canada',
    'Japan', 'Philippines', 'United Kingdom', 'Brazil', 'Ireland',
    'others', 'Singapore', 'India', 'Germany', 'Indonesia', 'Mexico',
    'Netherlands', 'France', 'Spain', 'Turkey', 'Poland', 'Russia',
    'Taiwan', 'Thailand', 'Vietnam', 'Romania'
])
trafficSource = st.selectbox('Traffic Source', [
    'others', 'google', '(direct)', 'Partners', 'youtube.com'
])
trafficMedium = st.selectbox('Traffic Medium', [
    'organic', 'cpm', 'referral', 'cpc', '(none)', 'affiliate'
])
isFirstVisit = st.selectbox('Is First Visit', ['no', 'yes'])  # Changed from 0/1 to no/yes
totalVisits = st.number_input('Total Visits', value=10)
totalHits = st.number_input('Total Hits', value=100)
totalPageviews = st.number_input('Total Pageviews', value=50)
totalTimeOnSite = st.number_input('Total Time on Site', value=2000)
productPagesViewed = st.number_input('Product Pages Viewed', value=5)
input_date = st.date_input('Date', datetime.now())

# Convert date input to features
date_features = date_to_features(input_date)

# Combined feature set
combined_features = {
    'deviceCategory': deviceCategory,
    'operatingSystem': operatingSystem,
    'browser': browser,
    'country': country,
    'trafficSource': trafficSource,
    'trafficMedium': trafficMedium,
    'isFirstVisit': isFirstVisit,
    'totalVisits': totalVisits,
    'totalHits': totalHits,
    'totalPageviews': totalPageviews,
    'totalTimeOnSite': totalTimeOnSite,
    'productPagesViewed': productPagesViewed,
    'year': date_features['year'],
    'month': date_features['month'],
    'day': date_features['day'],
    'day_of_week': date_features['day_of_week'],
    'month_sin': date_features['month_sin'],
    'month_cos': date_features['month_cos'],
    'day_of_week_sin': date_features['day_of_week_sin'],
    'day_of_week_cos': date_features['day_of_week_cos'],
    'is_weekend': date_features['is_weekend']
}

# Ensure the features are in the correct order
feature_order = [
    'deviceCategory', 'operatingSystem', 'browser', 'country',
    'trafficSource', 'trafficMedium', 'isFirstVisit', 'totalVisits',
    'totalHits', 'totalPageviews', 'totalTimeOnSite', 'productPagesViewed',
    'year', 'month', 'day', 'day_of_week', 'month_sin',
    'month_cos', 'day_of_week_sin', 'day_of_week_cos', 'is_weekend'
]

# Mapping categorical features to numeric values
def map_values(value, mapping):
    return mapping[value]

# Define mappings for each categorical feature
device_category_mapping = {
    'desktop': 0, 'mobile': 1, 'tablet': 2
}

operating_system_mapping = {
    'Android': 0, 'Chrome OS': 1, 'Linux': 2, 'Macintosh': 3, 'Windows': 4, 'iOS': 5, 'others': 6
}

browser_mapping = {
    'Android Webview': 0, 'Chrome': 1, 'Edge': 2, 'Firefox': 3, 'Internet Explorer': 4, 'Opera': 5,
    'Opera Mini': 6, 'Safari': 7, 'Safari (in-app)': 8, 'UC Browser': 9, 'YaBrowser': 10, 'others': 11
}

country_mapping = {
    'Australia': 0, 'Brazil': 1, 'Canada': 2, 'France': 3, 'Germany': 4, 'India': 5, 'Indonesia': 6,
    'Ireland': 7, 'Italy': 8, 'Japan': 9, 'Malaysia': 10, 'Mexico': 11, 'Netherlands': 12, 'Philippines': 13,
    'Poland': 14, 'Romania': 15, 'Russia': 16, 'Singapore': 17, 'Spain': 18, 'Taiwan': 19, 'Thailand': 20,
    'Turkey': 21, 'United Kingdom': 22, 'United States': 23, 'Vietnam': 24, 'others': 25
}

traffic_source_mapping = {
    '(direct)': 0, 'Partners': 1, 'google': 2, 'others': 3, 'youtube.com': 4
}

traffic_medium_mapping = {
    '(none)': 0, 'affiliate': 1, 'cpc': 2, 'cpm': 3, 'organic': 4, 'referral': 5
}

# Convert 'no'/'yes' to 0/1
combined_features['isFirstVisit'] = 1 if combined_features['isFirstVisit'] == 'yes' else 0

# Map the categorical values
combined_features['deviceCategory'] = map_values(combined_features['deviceCategory'], device_category_mapping)
combined_features['operatingSystem'] = map_values(combined_features['operatingSystem'], operating_system_mapping)
combined_features['browser'] = map_values(combined_features['browser'], browser_mapping)
combined_features['country'] = map_values(combined_features['country'], country_mapping)
combined_features['trafficSource'] = map_values(combined_features['trafficSource'], traffic_source_mapping)
combined_features['trafficMedium'] = map_values(combined_features['trafficMedium'], traffic_medium_mapping)

feature_list = [combined_features[feature] for feature in feature_order]

# Predict button
if st.button('Predict'):
    prediction = model.predict([feature_list])[0]
    
    if prediction == 0:
        st.write("The product is not added to the cart")
    else:
        st.write("The product is added to the cart")
