import pandas as pd
import numpy as np
from bs4 import BeautifulSoup
import requests
import time
from urllib.error import HTTPError
import random
# from requests_ip_rotator import ApiGateway, EXTRA_REGIONS
from fp.fp import FreeProxy
# from proxy_requests import ProxyRequests
import math as m



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

'''adds the player_table to the appropriate .csv file in the database. Also adds the player's name and ID to the data.'''
def add_player_stats_to_table(player_table, player):
    yrly_stats_path = '/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics/'
    #Add the player name and ID to the player_table
    player_table.insert(0, 'Name', player['Name'])
    player_table.insert(0, 'Player ID', player['Player ID'])
    try:
        most_frequent_pos = player_table['Pos'].mode()[0] #COMMENT we will consider the player's position to be whatever position they were most frequently in their career.
    except:
        return
    #append the player_table to the right position table and save it.
    if(most_frequent_pos == 'RB' or most_frequent_pos == 'FB'or most_frequent_pos == 'HB'):
        table = pd.read_csv(f'{yrly_stats_path}RB_table.csv', index_col = 0)
        pos = 'RB'
    elif(most_frequent_pos == 'QB'):
        table = pd.read_csv(f'{yrly_stats_path}QB_table.csv', index_col = 0)
        pos = 'QB'
    elif(most_frequent_pos == 'WR' or most_frequent_pos == 'FL'):
        table = pd.read_csv(f'{yrly_stats_path}WR_table.csv', index_col = 0)
        pos = 'WR'
    elif(most_frequent_pos == 'TE'):    
        table = pd.read_csv(f'{yrly_stats_path}TE_table.csv', index_col = 0)
        pos = 'TE'
    elif(is_offensive_lineman(player['position'])):
        table = pd.read_csv(f'{yrly_stats_path}O_LINE_table.csv', index_col = 0)
        pos = 'O_LINE'
    else:
        print('IDK the position')
        return
    table = pd.concat([table, player_table], ignore_index=True)
    table.to_csv(f'{yrly_stats_path}{pos}_table.csv')
    return


'''helper to rotate proxies... hopefully helps the "too many requests error."'''
def get_free_proxy():
    proxy = FreeProxy(rand=True, timeout=1).get()
    return proxy

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
            proxy = get_free_proxy()
            
            response = requests.get(url, timeout=timeout)   #rotating nothing
            # response = requests.get(url, headers=headers, timeout=timeout)     #rotating user agents
            # response = requests.get(url, headers=headers, timeout=timeout, proxies={"http": proxy, "https": proxy})    #rotating user agents and proxies
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the HTML content using pandas
            player_table = pd.read_html(response.text, header=1)[0]
            player_table['Year'] = player_table['Year'].astype(str).str.replace('+', '', regex=False) #removing the + from the year column      
            #COMMENT need the above line here because it checks to see if "Year" is even a column... it will not be if the player is an offensive lineman. if its a lineman, then it has to go to the second try/except.
            return player_table
        except requests.RequestException as e:
            print(f"Primary attempt {attempt + 1} failed: {e}")
            print('its probably an offensive lineman... or we have had too many requests.')
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
            proxy = get_free_proxy()
            response = requests.get(url, timeout=timeout)   #rotating nothing
            # response = requests.get(url, headers=headers, timeout=timeout)     #rotating user agents
            # response = requests.get(url, headers=headers, timeout=timeout, proxies={"http": proxy, "https": proxy})    #rotating user agents and proxies
            response.raise_for_status()  # Raises an HTTPError for bad responses

            # Read the HTML content using pandas
            player_table = pd.read_html(response.text, header=0)[0]
            player_table['Year'] = player_table['Year'].astype(str).str.replace('+', '', regex=False) #removing the + from the year column #COMMENT this will now work for offensive linemen

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





i = -1
all_players_path = '/Users/jakehirst/Desktop/fantasy_football_predictors/All_players_and_ids.csv'
all_players = pd.read_csv(all_players_path, index_col=0) #players, their player ID, and the website to get their career stats on.

#go through each player in the player list, and get their yearly statistics.
while i+1 < len(all_players):
    i+=1 #go to next player
    all_players = pd.read_csv(all_players_path, index_col=0) #players, their player ID, and the website to get their career stats on.
    player = all_players.iloc[i]
    start_year = player['start year']
    
    #we will limit our dataset to players that started their careers after 1960
    if(start_year < 1960):
        print('player is too old')
        continue
    #have gotten some nan positions
    elif(type(player['position']) == float and m.isnan(player['position'])):
        print('no position reported')
        continue
    #only doing offensive positions for now
    elif not is_offensive_player(player['position']):
        print('not an offensive player')
        continue
    # try to prevent duplicate rows
    elif(player['statistics recieved?'] == 1.0):
        print('statistics already gotten')
        continue

    
        
    name = player['Name']
    player_id = player['Player ID']
    url = player['href']
    player_table = get_player_table(url)
    
    #clean up year column
    # player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
    # player_table['Year'] = player_table['Year'].astype(str).str.replace('+', '', regex=False) #removing the + from the year column
    #remove the total career stats and anything after that.
    try:
        career_index = player_table[player_table['Year'] == 'Career'].index[0]
        player_table = player_table.iloc[:career_index]
    except:
        print('\n No total career stats column?')
    #remove nan values in the year column
    player_table = player_table[player_table['Year'].notna()]

    player_table['Year'] = player_table['Year'].str.replace('*', '', regex=False)
    player_table['Year'] = player_table['Year'].replace("", pd.NA)
    player_table['Year'] = player_table['Year'].astype(str)    # Convert the "Year" column to string to handle <NA> properly
    player_table = player_table[~player_table['Year'].str.contains('Career|yrs|<NA>|nan|1 yr', na=False)]  # Filter out rows where "Year" is 'Career', contains 'yrs', or is <NA>, or is nan
    player_table['Year'] = player_table['Year'].astype(int)    # Convert the "Year" column to an int
    print('\n')
    print(player)
    print(player_table)
    
    
    add_player_stats_to_table(player_table, player) # add the player stats to the appropriate position table
    #mark that player as collected and save it in all_players
    all_players.loc[all_players['Player ID'] == player['Player ID'], 'statistics recieved?'] = 1.0
    all_players.to_csv(all_players_path)
    
    print('here')
    # player['statistics recieved'] += 1.0