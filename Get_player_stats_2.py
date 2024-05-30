import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
from urllib.error import HTTPError




'''
I think I want to organize the player statistics by position. I will void any defensive positions and special teams because they are not important in fantasy.
I will keep offensive linemen positions because they might be important later down the line when I want to add features involving the offensive line.
'''


def clean_year_column():
    
    return 

def is_offensive_player(position):
    offensive_positions = ['QB', 'RB', 'WR', 'TE', 'G', 'T', 'C', 'FB']
    return any(item in position.split('-') for item in offensive_positions)


def is_offensive_lineman(position):
    O_lineman_positions = ['G', 'T', 'C', 'FB']
    return any(item in position.split('-') for item in O_lineman_positions)

'''gets the player_table given the url of the player's career stats.'''
def get_player_table(url, retries=3, delay=2):
    time.sleep(delay) #need to delay so that i dont get a "too many requests" error.
    for attempt in range(retries):
        try:
            player_table = pd.read_html(url, header=1)
            player_table = player_table[0]
            player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
            return player_table
        except:
            player_table = pd.read_html(url, header=0)
            player_table = player_table[0]
            player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
            return player_table
        finally:
            print('couldnt get player table...')

    





all_players = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/All_players_and_ids.csv', index_col=0) #players, their player ID, and the website to get their career stats on.
print(all_players)

#go through each player in the player list, and get their yearly statistics.
for player in all_players.iterrows():
    player = player[1]
    start_year = player['start year']
    
    if(start_year < 1960):
        #we will limit our dataset to players that started their careers after 1960
        print('player is too old')
        continue
    
    position = player['position']
    print(position)
    if not is_offensive_player(position):
        # Here we are checking to see if they were ever an offensive player. If not, then we skip them.
        print('not an offensive player')
        continue
    

        
        
    name = player['Name']
    player_id = player['Player ID']
    url = player['href']
    
    player_table = get_player_table(url)
    #clean up year column
    # player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)

    #remove the total career stats and anything after that.
    career_index = player_table[player_table['Year'] == 'Career'].index[0]
    player_table = player_table.iloc[:career_index]
    
    #remove nan values in the year column
    player_table = player_table[player_table['Year'].notna()]

    
    player_table['Year'] = player_table['Year'].str.replace('*', '', regex=False)
    player_table['Year'] = player_table['Year'].replace("", pd.NA)
    player_table['Year'] = player_table['Year'].astype(str)    # Convert the "Year" column to string to handle <NA> properly
    player_table = player_table[~player_table['Year'].str.contains('Career|yrs|<NA>', na=False)]  # Filter out rows where "Year" is 'Career', contains 'yrs', or is <NA>
    player_table['Year'] = player_table['Year'].astype(int)    # Convert the "Year" column to an int
    print('\n')
    print(player)
    print(player_table)
    
    
    print('here')