import pandas as pd
import requests
from bs4 import BeautifulSoup


'''scrapes data from a given year (say 2019 = https://www.pro-football-reference.com/years/2019/fantasy.htm) '''

predicting_year = 2019
website = f'https://www.pro-football-reference.com/years/{predicting_year}/fantasy.htm'
all_player_table = pd.read_html(website, header=1)
all_player_table = all_player_table[0]

#player data for each player in the predicting year
skills_pos_table = all_player_table[all_player_table['FantPos'].isin(['TE', 'RB', 'WR'])]
skills_pos_table = skills_pos_table.reset_index(drop=True)

#get the previous years stats for each of the players for their entire career
for row in skills_pos_table.iterrows():
    player = row[1]['Player'].replace('+', '').replace('*', '') #remove the + and * from the player name.
    response = requests.get(website)
    soup = BeautifulSoup(response.content, 'html.parser')
    link = soup.find('a', text=player)
    
    player_href = link.get('href')
    player_website = f'https://www.pro-football-reference.com{player_href}'
    player_table = pd.read_html(player_website, header=1)
    player_table = player_table[0]

    #clean up year column
    player_table['Year'] = player_table['Year'].str.replace('+', '', regex=False)
    print('\n')
    print(player)
    print(player_table)

    player_table['Year'] = player_table['Year'].str.replace('*', '', regex=False)
    player_table['Year'] = player_table['Year'].replace("", pd.NA)
    player_table['Year'] = player_table['Year'].astype(str)    # Convert the "Year" column to string to handle <NA> properly
    player_table = player_table[~player_table['Year'].str.contains('Career|yrs|<NA>', na=False)]  # Filter out rows where "Year" is 'Career', contains 'yrs', or is <NA>
    player_table['Year'] = player_table['Year'].astype(int)    # Convert the "Year" column to an int
    
    #now I want to redefine the column names to make them more informative.
    player_table = player_table.rename(columns={"G": "Gms plyd", "GS": "Gms Strtd", "Yds.1":"Rec Yds", "Yds":"Rush Yds", "1D":"1 Dns (Rsh)", "1D.1":"1 Dns (Rec)", "Lng":"Lng (Rsh)", "Lng.1":"Lng (Rec)"})

    
    print('\n filtered player table:')
    print(player_table)
    print('here')    
    
print('here')