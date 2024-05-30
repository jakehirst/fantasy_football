import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
from urllib.error import HTTPError
import random




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



'''gets the player_table given the url of the player's career stats.
I also implemented exponential backoff (increasing delay every time request fails)
and rotating user agents. This should help avoid the "too many requests" error.'''
def get_player_table(url, retries=3, delay=2, backoff_factor=1.5, timeout=10):
    # List of user agents for rotation
    USER_AGENTS = [
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
    ]
    attempt = 0
    while attempt < retries:
        try:
            # Rotate user agents
            user_agent = random.choice(USER_AGENTS)
            headers = {'User-Agent': user_agent}

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the HTML content using pandas
            player_table = pd.read_html(response.text, header=1)[0]
            player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
            return player_table
        except requests.RequestException as e:
            print(f"Primary attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            delay *= backoff_factor  # Exponential backoff
        except Exception as e:
            print(f"Primary attempt {attempt + 1} failed to parse: {e}")
            time.sleep(delay)
            delay *= backoff_factor  # Exponential backoff

        try:
            # Rotate user agents again for the second attempt
            user_agent = random.choice(USER_AGENTS)
            headers = {'User-Agent': user_agent}

            response = requests.get(url, headers=headers, timeout=timeout)
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the HTML content using pandas
            player_table = pd.read_html(response.text, header=0)[0]
            player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
            return player_table
        except requests.RequestException as e:
            print(f"Secondary attempt {attempt + 1} failed: {e}")
            time.sleep(delay)
            delay *= backoff_factor  # Exponential backoff
        except Exception as e:
            print(f"Secondary attempt {attempt + 1} failed to parse: {e}")
            time.sleep(delay)
            delay *= backoff_factor  # Exponential backoff

        attempt += 1

    print('Couldn\'t get player table after multiple attempts.')
    return None







all_players = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/All_players_and_ids.csv', index_col=0) #players, their player ID, and the website to get their career stats on.
print(all_players)

#go through each player in the player list, and get their yearly statistics.
for player in all_players.iterrows(): #TODO write this such that it adds to the existing saved player stats. For example, if it stopped working after DK metcalf, it picks up the all_players list from there and continues.
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