import pandas as pd
import numpy as np
'''
rules for PPR scoring:
TD pass = 4 pts
25 passing yds = 1pt
2pt passing conversion = 2pts


'''


wr_data = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/WR_table_clean.csv',index_col=0)
wr_data.columns = ['Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'Tgt', 'Rec',
       'Rec Yds', 'Yds/Rec', 'Rec TD', 'Rec 1D', 'Rec Succ%', 'Rec Lng', 'Rec/G', 'Rec Y/G', 'Ctch%',
       'Y/Tgt', 'Rush Att', 'Rush Yds', 'Rush TD', 'Rush 1D', 'Rush Succ%', 'Rush Lng', 'Y/A',
       'Rush Y/G', 'A/G', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV',
       'Player ID', 'Awards', 'injured']
