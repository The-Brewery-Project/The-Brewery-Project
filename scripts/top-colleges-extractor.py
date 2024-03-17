# -*- coding: utf-8 -*-
'''
America's 150 Best College Towns (2021 Ranking)
'''

import requests
from bs4 import BeautifulSoup
import numpy as np
import pandas as pd

url = 'https://listwithclever.com/research/best-college-towns-2021/'
url_headers = {'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_11_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/50.0.2661.102 Safari/537.36'}
# response of 403 without headers, 200 with
page = requests.get(url, headers=url_headers)
soup = BeautifulSoup(page.text, 'lxml')

# get table with top 150
college_table = soup.find('div', class_ = 'clever-table border-top border-bot col-1-center col-2-left col-3-left')

# identify table rows
college_rows = college_table.find_all('tr')

# header
col_headers = [header.text for header in college_rows[0].find_all('th')]

# data
top_colleges_dict = {'ranking': [], 'city': [], 'colleges': []}
for row in college_rows:
    ranked_row = row.find_all('td')
    if len(ranked_row) == 3:
        top_colleges_dict['ranking'].append(ranked_row[0].text)
        top_colleges_dict['city'].append(ranked_row[1].text)
        top_colleges_dict['colleges'].append(ranked_row[2].text)

# put data into pandas dataframe
colleges_df = pd.DataFrame(top_colleges_dict)        

# split city and state
colleges_df[['city', 'state']] = colleges_df['city'].str.split(', ', expand=True)

# rearrange order
colleges_df = colleges_df[['city', 'state', 'colleges']]

# lowercase city. state. and colleges
colleges_df['city'] = colleges_df['city'].str.lower()
colleges_df['state'] = colleges_df['state'].str.lower()
colleges_df['colleges'] = colleges_df['colleges'].str.lower()

# unabbreviate state process
# remove punctuation
colleges_df['state'] = colleges_df['state'].str.replace('.', '')

# abbreviated states
abbreviated = ['calif','va','nj','mich','mass','ill','fla','ind','wva','nc','pa',
               'miss','texas','colo','ny','ga','ore','iowa','del','wis','md','okla',
               'mo','utah','minn','ariz','la','sc','ky','wash','ark','ala','wyo',
               'conn','kan','nm','tenn','ohio','mont','nev','ri','maine','vt','idaho']

# full states
full = ['california','virginia','new jersey','michigan','massachusetts','illinois',
        'florida','indiana','west virginia','north carolina','pennsylvania',
        'mississippi','texas','colorado','new york','georgia','oregon','iowa',
        'delaware','wisconsin','maryland','oklahoma','missouri','utah','minnesota',
        'arizona','louisiana','south carolina','kentucky','washington','arkansas',
        'alabama','wyoming','connecticut','kansas','new mexico','tennessee','ohio',
        'montana','nevada','rhode island','maine','vermont','idaho']

# dictionary to switch between abbreviation to full state names
# table article used non-standard abbreviations, so needed a custom dictionary
abbreviated_states_custom = {abbr_st: full_st for abbr_st, full_st in zip(abbreviated, full)}

# apply the dictionary
colleges_df['state'] = colleges_df['state'].apply(lambda state: abbreviated_states_custom[state])

colleges_df.to_csv('../data/top_colleges.csv', index=False)
