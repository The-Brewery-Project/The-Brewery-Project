# -*- coding: utf-8 -*-
"""The Brewery Project Web Scraping (Ski Resorts).ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1yaig1X__jt9hoGL_feEgc-VfGUQo_nht

**Web Scraping**
"""

import pandas as pd
import requests
from bs4 import BeautifulSoup
from IPython.display import FileLink
from IPython.display import display
import re

# Build scraper

url = "https://en.wikipedia.org/wiki/List_of_ski_areas_and_resorts_in_the_United_States"

response = requests.get(url)

soup = BeautifulSoup(response.text, 'html.parser')

state_headings = soup.find_all("h3")

ski_resorts_data = []

for heading in state_headings:
  state = heading.text.strip().replace("[edit]", "")
  ul = heading.find_next_sibling("ul")
  if ul:
    ski_resorts = ul.find_all("li")
    for resort in ski_resorts:
      resort_info = resort.text.split("—")
      if len(resort_info) == 2:
        ski_resort = resort_info[0].strip()
        city = resort_info[1].strip()
        ski_resorts_data.append({"Ski Resort": ski_resort, "City": city, "State": state})

ski_resorts_df = pd.DataFrame(ski_resorts_data)

# Data cleaning

# Removes "near " before city name
ski_resorts_df['City'] = ski_resorts_df['City'].str.replace(r'^near\s*', '', regex=True)

# Removes unnecessary text in parentheses and brackets
ski_resorts_df['City'] = ski_resorts_df['City'].str.replace(r'[\[\(].*[\]\)]', '', regex=True)

# Some resorts have multiple cities listed. Pulls first city only.
ski_resorts_df['City'] = ski_resorts_df['City'].str.split('\n').str[0]

#display(ski_resorts_df)

ski_resorts_df.to_csv('ski_resorts.csv', index=False)

FileLink('ski_resorts.csv')