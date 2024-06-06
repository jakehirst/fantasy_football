import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



'''only keeps the columns in the table with the column names in columns_to_keep'''
def remove_unecessary_columns(df, columns_to_keep):
    df = df[columns_to_keep]    
    return df

'''adds a column to the table that indicates if the player was injured during that year. '''
def add_injured_column(df):
    # Add the 'injured' column with 1 if 'Tm' contains 'Injured', otherwise 0
    df['injured'] = df['Tm'].apply(lambda x: 1 if 'Injured' in x else 0)
    return df

'''fills nan values of the age with the previous row's value plus 1 (they are most likely one year older in the next season)'''
def fill_nan_age_values(group):
    group['Age'] = group['Age'].fillna(method='ffill') + group['Age'].isna().cumsum()
    return group

'''replaces any values in the given column that contains 'missed season' with "N/A". also does this for nan values.'''
def replace_missed_season_with_NA(df, column_names):
    # Replace values containing 'missed season' with "N/A"
    for column_name in column_names:
        df[column_name] = df[column_name].apply(lambda x: "N/A" if pd.notna(x) and 'Missed season' in x else x)
        df[column_name] = df[column_name].apply(lambda x: "N/A" if pd.isna(x) else x)
    return df

'''replaces any values in the given column that contains 'missed season' with 0. also does this for nan values.'''
def replace_missed_season_with_ZERO(df, column_names):
    # Replace values containing 'missed season' with "N/A"
    for column_name in column_names:
        df[column_name] = pd.to_numeric(df[column_name], errors='coerce') #cast all values to numeric, which the ones with 'missed season' will be Nan
        df[column_name] = df[column_name].apply(lambda x: 0 if pd.isna(x) else x) #replace all Nan values (missed season) with 0
        df[column_name] = df[column_name].astype(float)#cast to float to make sure theyre all the same type.
    return df

'''only keep rows taht have positions that are in the valid_positions array.'''
def get_rid_of_invalid_positions(df, column_name, valid_positions):
    df = df[df[column_name].isin(valid_positions)]
    return df

def main():
        

    '''cleaning WR table'''
    table = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics/WR_table.csv', index_col=0)
    table = table[table['Player ID'].notna()] #get rid of empty rows 
    columns_to_keep = ['Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'Tgt', 'Rec',
       'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'Lng', 'R/G', 'Y/G', 'Ctch%',
       'Y/Tgt', 'Att', 'Yds.1', 'TD.1', '1D.1', 'Succ%.1', 'Lng.1', 'Y/A',
       'Y/G.1', 'A/G', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV', 'Player ID', 'Awards']
    table = remove_unecessary_columns(table, columns_to_keep)    
    table = add_injured_column(table)
    table = table.groupby('Player ID').apply(fill_nan_age_values) #grouping by Player ID, fill the nan value of the age column
    table = table.reset_index(drop=True) #need to do this to get rid of the playerID group by...

    table = replace_missed_season_with_NA(table, ['Tm', 'Pos', 'No.', 'Awards'])

    table['Ctch%'] = table['Ctch%'].str.rstrip('%')  #get rid of % sign in Ctch% column
    # table['AV'] = pd.to_numeric(table['AV'], errors='coerce') #need to cast all AV to numeric that replace_missed_season_with_ZERO() can do its work.
    table = replace_missed_season_with_ZERO(table, ['G', 'GS', 'AV', 'Att', 'Yds', 'TD', '1D', 'Succ%', 'Lng', 'Y/A', 'Y/G', 'A/G', 'Tgt', 'Rec', 'Yds.1',
                                                    'Y/R', 'TD.1', '1D.1', 'Succ%.1', 'Lng.1', 'R/G', 'Y/G.1', 'Ctch%',
                                                    'Y/Tgt', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV'])#replace missed seasons with 0 for all the columns in the array
    
    
    #get rid of rows where the player wasnt an WR
    table = get_rid_of_invalid_positions(table, 'Pos', ['WR', 'FL', 'N/A'])
    
    table.to_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/WR_table_clean.csv')

    print('here')

if __name__ == "__main__":
    main()
