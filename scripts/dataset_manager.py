# -*- coding: utf-8 -*-
'''
python file to extract all datasets and combine them into a single dataframe for analysis.
'''

# import libraries
import numpy as np
import pandas as pd

# import datasets
open_brewery_df = pd.read_csv('../data/open-brewery-db.csv')
metro_df = pd.read_csv('../data/metropolianCities.csv')
national_park_df = pd.read_csv('../data/national_parks.csv')
ski_resorts_df = pd.read_csv('../data/ski_resorts.csv')
tech_df = pd.read_csv('../data/techHubs.csv')

'''
Open Brewery DB
'''
# get number of breweries per city
breweries_per_city = pd.DataFrame(open_brewery_df['city'].value_counts()).reset_index()
breweries_per_city.columns = ['city', 'city_count']

# get number of breweries per state
breweries_per_state = pd.DataFrame(open_brewery_df['state'].value_counts()).reset_index()
breweries_per_state.columns = ['state', 'state_count']

# merge cities with states
city_state = open_brewery_df[['city', 'state']]
city_state = city_state.drop_duplicates(subset='city')
city_level_df = pd.merge(breweries_per_city, city_state, how='inner', on='city')
city_level_df = pd.merge(city_level_df, breweries_per_state, how='inner', on='state')

# brewery types per city
brewery_type = open_brewery_df.pivot_table(index='city', columns='brewery_type', aggfunc='size', fill_value=0)
city_level_df = pd.merge(city_level_df, brewery_type, how='inner', on='city')

'''
Major Metropolitan
'''
# extract and clean major cities
metro_df['City Names'] = metro_df['City Names'].str.lower()
metro_df['State Name'] = metro_df['State Name'].str.lower()

# remove (balance)
metro_df['City Names'] = metro_df['City Names'].str.replace('\(balance\)', '')
metro_df['State Name'] = metro_df['State Name'].str.replace('\(balance\)', '')

# trim if needed
metro_df['City Names'] = metro_df['City Names'].str.strip()
metro_df['State Name'] = metro_df['State Name'].str.strip()

# major cities
major_cities = metro_df[['City Names', 'State Name']]
major_cities.columns = ['city', 'state']
major_cities['major_city'] = 1

# find in city_level_df
city_level_df = pd.merge(city_level_df, major_cities, how = 'left', on=['city', 'state'])
city_level_df['major_city'] = city_level_df['major_city'].fillna(0)

'''
National Parks
'''
# lower case and trim
national_park_df['State'] = national_park_df['State'].str.lower()
national_park_df['State'] = national_park_df['State'].str.strip()

# national parks by state
national_park_df['State'].value_counts()
national_park_counts = pd.DataFrame(national_park_df['State'].value_counts()).reset_index()
national_park_counts.columns = ['state', 'national_park_count']

# find in city_level_df
city_level_df = pd.merge(city_level_df, national_park_counts, how = 'left', on=['state'])
city_level_df['national_park_count'] = city_level_df['national_park_count'].fillna(0)

'''
Ski Resorts
'''
# lower case and trim
# city
ski_resorts_df['City'] = ski_resorts_df['City'].str.lower()
ski_resorts_df['City'] = ski_resorts_df['City'].str.strip()
# state
ski_resorts_df['State'] = ski_resorts_df['State'].str.lower()
ski_resorts_df['State'] = ski_resorts_df['State'].str.strip()

# ski resort cities
ski_resort_cities = ski_resorts_df[['City', 'State']]
ski_resort_cities.columns = ['city', 'state']
ski_resort_cities['ski_resort'] = 1

# find in city_level_df
city_level_df = pd.merge(city_level_df, ski_resort_cities, how = 'left', on=['city', 'state'])
city_level_df['ski_resort'] = city_level_df['ski_resort'].fillna(0)
city_level_df = city_level_df.drop_duplicates()

'''
Tech Hubs
'''
# lower case and trim
# city
tech_df['City'] = tech_df['City'].str.lower()
tech_df['City'] = tech_df['City'].str.strip()
# state
tech_df['State Name'] = tech_df['State Name'].str.lower()
tech_df['State Name'] = tech_df['State Name'].str.strip()

# ski resort cities
tech_cities = tech_df[['City', 'State Name']]
tech_cities.columns = ['city', 'state']
tech_cities['tech_city'] = 1

# find in city_level_df
city_level_df = pd.merge(city_level_df, tech_cities, how = 'left', on=['city', 'state'])
city_level_df['tech_city'] = city_level_df['tech_city'].fillna(0)

'''
One Last Data Review
'''
# check nulls - following should return 0 if good
# city_level_df.isnull().sum().sum() # comment out for scripting

# check duplicates - following should return False if good
# city_level_df.duplicated().any() # comment out for scripting

'''
export data
'''
# export file
city_level_df.to_csv('../data/city_level.csv', index = False)