'''
Here we make a list of all players in history of football and put them in a .csv. Each player needs to have a unique ID number.

We get this information from https://www.pro-football-reference.com/players/A/ , which contains the player name, position, and 
years played for each player whose last name starts with A. (obviously we repeat for the whole alphabet)
'''
import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
from Get_player_stats_2 import is_offensive_lineman


def get_player_fantasy_points(href):
    player_letter = href.split('/')[-2]
    player_code = href.split('/')[-1].split('.')[0]
    fantasy_url = f'https://www.pro-football-reference.com/players/{player_letter}/{player_code}/fantasy/'
    response = requests.get(fantasy_url)
    
    player_table = pd.read_html(response.text, header=2)[0]
    fantasy_pts_per_year_table = player_table.copy()
    fantasy_pts_per_year_table = fantasy_pts_per_year_table[['Unnamed: 0', 'FantPt']]
    #TODO add name and player ID columns... 
    #TODO  then delete the last row which is a "totals" row...
    #TODO  and then save them to a table somewhere...
    #TODO when you add them to your og table, you need to add the receptions to the points... cause we do ppr.
    return

alphabet = 'A B C D E F G H I J K L M N O P Q R S T U V W X Y Z'
alphabet = alphabet.split(' ')

players = []
player_id = 0
for last_name_letter in alphabet:
    url = f'https://www.pro-football-reference.com/players/{last_name_letter}/'
    # Fetch the HTML content from the URL
    response = requests.get(url)
    html_content = response.text
    # Parse the HTML content
    soup = BeautifulSoup(html_content, 'html.parser')
    
    # Find the div with the specified class and id
    div_players = soup.find('div', class_='section_content', id='div_players')

    # Extract player information
    for p in div_players.find_all('p'):
        player_info = {}
        
        # Extract player name and href
        a_tag = p.find('a')
        if a_tag:
            print(a_tag.text)
            player_info['Player ID'] = player_id
            player_info['Name'] = a_tag.text
            player_info['href'] = f"https://www.pro-football-reference.com{a_tag['href']}"
            fantasy_points = get_player_fantasy_points(player_info['href'])
        # Extract player position and years
        text_parts = p.get_text().split(' ')
        if len(text_parts) >= 2:
            player_info['position'] = text_parts[-2].strip('()')
            player_info['start year'] = int(text_parts[-1].split('-')[0])
            player_info['end year'] = int(text_parts[-1].split('-')[1])
        
        players.append(player_info)
        player_id += 1

df_players = pd.DataFrame(players, columns=["Player ID", "Name", "href", "position", "start year", "end year"])
df_players['statistics recieved?'] = np.zeros(len(df_players))# adding a column to record if we have their statistics or not.

df_players.to_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/All_players_and_ids.csv')
    
