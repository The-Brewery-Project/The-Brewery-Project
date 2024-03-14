# -*- coding: utf-8 -*-
'''
python script to extract data from open-brewery-db
'''

# libraries
import requests
import json
import pandas as pd
import numpy as np

# pull data with while loop (extracts all database data)
data_bool = True
page = 1
while data_bool == True:
    url = f'https://api.openbrewerydb.org/v1/breweries?page={page}&per_page=200'
    response = requests.get(url)
    data = response.json()
    if len(data) == 0:
        data_bool = False
        break
    if page == 1:
        data_df = pd.DataFrame(data)
    else:
        data_df = pd.concat([data_df, pd.DataFrame(data)])
    page += 1

# reset index
data_df.reset_index(inplace = True, drop = True)

# lambda function to make lower case, remove trailing or leading spaces
lower_strip = lambda text: text.lower().strip() if text is not None else None


# clean country column and filter for "united states"
data_df['country'] = data_df['country'].apply(func=lower_strip)
data_us = data_df[data_df['country'] == 'united states']

# drop all locations without addy 1, and remove addy 2 and addy 3 columns
data_us.dropna(subset = ['address_1'], axis = 0, inplace = True)
data_us.drop(['address_2', 'address_3'], axis = 1, inplace = True)

# remove address_1 and state_province columns (these are equivalents of street and state)
data_us.drop(['address_1', 'state_province'], axis = 1, inplace = True)

# clean name, brewery_type, city, state, street, and website_url columns
data_us['name'] = data_us['name'].apply(func=lower_strip)
data_us['brewery_type'] = data_us['brewery_type'].apply(func=lower_strip)
data_us['city'] = data_us['city'].apply(func=lower_strip)
data_us['state'] = data_us['state'].apply(func=lower_strip)
data_us['street'] = data_us['street'].apply(func=lower_strip)
data_us['website_url'] = data_us['website_url'].apply(func=lower_strip)

# export file
data_us.to_csv('../data/open-brewery-db.csv')