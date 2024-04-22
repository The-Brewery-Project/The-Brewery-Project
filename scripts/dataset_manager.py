# -*- coding: utf-8 -*-
'''
python file to extract all datasets and combine them into a single dataframe for analysis.
'''

# import libraries
import numpy as np
import pandas as pd
import regex as re

# import datasets
open_brewery_df = pd.read_csv('../data/open-brewery-db.csv')
college_df = pd.read_csv('../data/top_colleges.csv')
metro_df = pd.read_csv('../data/metropolianCities.csv')
national_park_df = pd.read_csv('../data/national_parks.csv')
ski_resorts_df = pd.read_csv('../data/ski_resorts.csv')
tech_df = pd.read_csv('../data/techHubs.csv')
census_df = pd.read_csv('../data/censusData.csv').drop(['Unnamed: 0'], axis = 1)

'''
Open Brewery DB
'''
# get number of breweries per city
breweries_per_city = pd.DataFrame(open_brewery_df['city'].value_counts()).reset_index()
breweries_per_city.columns = ['city', 'city_brewery_count']

# get number of breweries per state
breweries_per_state = pd.DataFrame(open_brewery_df['state'].value_counts()).reset_index()
breweries_per_state.columns = ['state', 'state_brewery_count']

# merge cities with states
city_state = open_brewery_df[['city', 'state']]
city_state = city_state.drop_duplicates(subset='city')
city_level_df = pd.merge(breweries_per_city, city_state, how='inner', on='city')
city_level_df = pd.merge(city_level_df, breweries_per_state, how='inner', on='state')

# brewery types per city
brewery_type = open_brewery_df.pivot_table(index='city', columns='brewery_type', aggfunc='size', fill_value=0)
city_level_df = pd.merge(city_level_df, brewery_type, how='inner', on='city')

'''
College Towns
'''
# college town boolean
college_towns = college_df[['city', 'state']]
college_towns['college_town'] = 1

# find in city_level_df
city_level_df = pd.merge(city_level_df, college_towns, how = 'left', on=['city', 'state'])
city_level_df['college_town'] = city_level_df['college_town'].fillna(0)

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

# state names from abbreviated
us_state_abbrev = {
            'AL': 'Alabama',
            'AK': 'Alaska',
            'AZ': 'Arizona',
            'AR': 'Arkansas',
            'CA': 'California',
            'CO': 'Colorado',
            'CT': 'Connecticut',
            'DE': 'Delaware',
            'FL': 'Florida',
            'GA': 'Georgia',
            'HI': 'Hawaii',
            'ID': 'Idaho',
            'IL': 'Illinois',
            'IN': 'Indiana',
            'IA': 'Iowa',
            'KS': 'Kansas',
            'KY': 'Kentucky',
            'LA': 'Louisiana',
            'ME': 'Maine',
            'MD': 'Maryland',
            'MA': 'Massachusetts',
            'MI': 'Michigan',
            'MN': 'Minnesota',
            'MS': 'Mississippi',
            'MO': 'Missouri',
            'MT': 'Montana',
            'NE': 'Nebraska',
            'NV': 'Nevada',
            'NH': 'New Hampshire',
            'NJ': 'New Jersey',
            'NM': 'New Mexico',
            'NY': 'New York',
            'NC': 'North Carolina',
            'ND': 'North Dakota',
            'OH': 'Ohio',
            'OK': 'Oklahoma',
            'OR': 'Oregon',
            'PA': 'Pennsylvania',
            'RI': 'Rhode Island',
            'SC': 'South Carolina',
            'SD': 'South Dakota',
            'TN': 'Tennessee',
            'TX': 'Texas',
            'UT': 'Utah',
            'VT': 'Vermont',
            'VA': 'Virginia',
            'WA': 'Washington',
            'WV': 'West Virginia',
            'WI': 'Wisconsin',
            'WY': 'Wyoming',
            'DC': 'District of Columbia',
            'MP': 'Northern Mariana Islands',
            'PW': 'Palau',
            'PR': 'Puerto Rico',
            'VI': 'Virgin Islands',
            'AA': 'Armed Forces Americas (Except Canada)',
            'AE': 'Armed Forces Africa/Canada/Europe/Middle East',
            'AP': 'Armed Forces Pacific',
            'AS': 'America Samoa'
        }

# lowercase dictionary
us_state_abbrev = {k.lower():us_state_abbrev[k].lower() for k in us_state_abbrev}

# apply dictionary to nation park dataframe
national_park_df['State'] = national_park_df['State'].apply(lambda state: us_state_abbrev[state])

# national park count per state
national_parks_per_state = pd.DataFrame(national_park_df['State'].value_counts()).reset_index()
national_parks_per_state.columns = ['state', 'state_national_park_count']
city_level_df = pd.merge(city_level_df, national_parks_per_state, how='left', on='state')
city_level_df['state_national_park_count'] = city_level_df['state_national_park_count'].fillna(0)

# drop non pertinent columns
national_park_df.drop(['National Park', 'Zip Code'], axis = 1, inplace=True)

# rename columns
national_park_df.columns = ['national_park_vistors', 'city', 'state']
national_park_df = national_park_df[['city', 'state', 'national_park_vistors']]

# sum duplicates
national_park_df = national_park_df.groupby(['city','state'], as_index=False).sum()

# find in city_level_df
city_level_df = pd.merge(city_level_df, national_park_df, how = 'left', on=['city', 'state'])
city_level_df['national_park_vistors'] = city_level_df['national_park_vistors'].fillna(0)


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

# ski resorts per state
ski_resorts_per_state = pd.DataFrame(ski_resorts_df['State'].value_counts()).reset_index()
ski_resorts_per_state.columns = ['state', 'ski_resort_count']
city_level_df = pd.merge(city_level_df, ski_resorts_per_state, how='left', on='state')
city_level_df['ski_resort_count'] = city_level_df['ski_resort_count'].fillna(0)

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
Census Data
'''
# rename city and state columns
census_df.columns = census_df.columns.str.lower()

# strip just in case
census_df['city'] = census_df['city'].str.strip()
census_df['state'] = census_df['state'].str.strip()

# unload regions (groupby removes them)
state_regions = census_df[['state', 'region']]
state_regions.drop_duplicates(inplace = True)

# average (to deal with percents) duplicates
census_df = census_df.groupby(['city','state'], as_index=False).mean()

# find in city_level_df
city_level_df = pd.merge(city_level_df, census_df, how = 'left', on=['city', 'state'])

# add regions back in
city_level_df = pd.merge(city_level_df, state_regions, how = 'left', on = 'state')

'''
brewery concentration / hotspot creation
'''

# create new column with brewery concentration per capita (per 1000)
city_level_df['brewery_concentration'] = 1000*(city_level_df['city_brewery_count'] / city_level_df['total population'])
# create ranks: 1 - 10
# labels as false and the + 1 return ints instead of category type data
city_level_df['per_capita_ranked'] = pd.qcut(city_level_df['brewery_concentration'], q=10, labels=False) + 1

# classes from 1-6
city_level_df['custom_ranked'] = 0
for index in city_level_df.index:
    if city_level_df['city_brewery_count'][index]<3:
        city_level_df['custom_ranked'][index] = 1
    else:
        city_level_df['custom_ranked'][index] = pd.qcut(city_level_df['city_brewery_count'][city_level_df['city_brewery_count']>=3],q=10,duplicates='drop',labels=False)[index] + 1

# nulls = 723 brewery cities not matched with census
# city_nulls = city_level_df[city_level_df['total population'].isnull()]
# city_nulls_info = city_nulls.describe() # commented out for scripting

# the below functions are commented out for script running purposes
# remove multistring comment before using functions
'''
# function to remove matched values
def remove_matched_census(df1 = census_df, df2 = city_level_df):
    # Create a boolean mask where 'city' and 'state' match
    mask = df1.apply(lambda row: (row['city'], row['state']) in zip(df2['city'], df2['state']), axis=1)
    
    # Get the indices where the values match
    matching_indices = df1.index[mask].tolist()
    
    # drop indices
    df1 = census_df.drop(matching_indices)
    df1.reset_index(drop = True, inplace = True)
    return df1

# function to check indices for matching 
def find_matched_indices(df1 = census_df, df2 = city_level_df):
    # Create a boolean mask where 'city' and 'state' match
    mask = df1.apply(lambda row: (row['city'], row['state']) in zip(df2['city'], df2['state']), axis=1)
    
    # Get the indices where the values match
    matching_indices = df1.index[mask].tolist()
    return matching_indices
'''

'''
One Last Data Review
'''
# check nulls - following should return 0 if good (currently with 723 from census)
# city_level_df.isnull().sum().sum() # comment out for scripting

# check duplicates - following should return False if good
# city_level_df.duplicated().any() # comment out for scripting

'''
export data
'''
# export file
city_level_df.to_csv('../data/city_level.csv', index = False)
