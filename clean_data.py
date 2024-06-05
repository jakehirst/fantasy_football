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
def replace_missed_season_with_NA(df, column_name):
    # Replace values containing 'missed season' with "N/A"
    df[column_name] = df[column_name].apply(lambda x: "N/A" if pd.notna(x) and 'Missed season' in x else x)
    df[column_name] = df[column_name].apply(lambda x: "N/A" if pd.isna(x) else x)
    return df

'''replaces any values in the given column that contains 'missed season' with 0. also does this for nan values.'''
def replace_missed_season_with_ZERO(df, column_name):
    # Replace values containing 'missed season' with "N/A"
    df[column_name] = df[column_name].apply(lambda x: 0 if pd.notna(x) and 'Missed season' in x else x)
    df[column_name] = df[column_name].apply(lambda x: 0 if pd.isna(x) else x)
    return df

'''only keep rows taht have positions that are in the valid_positions array.'''
def get_rid_of_invalid_positions(df, column_name, valid_positions):
    df = df[df[column_name].isin(valid_positions)]
    return df

def main():
    '''cleaning oline table'''
    # table = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics/O_LINE_table.csv')
    # columns_to_keep = ['Name', 'Year','Age', 'Tm', 'Pos', 'No.','G','GS','AV', 'Awards', 'Player ID']
    # table = remove_unecessary_columns(table, columns_to_keep)
    # table = add_injured_column(table)
    # table = table.groupby('Player ID').apply(fill_nan_age_values) #grouping by Player ID, fill the nan value of the age column
    # table = table.reset_index(drop=True) #need to do this to get rid of the playerID group by...

    # table = replace_missed_season_with_NA(table, 'Tm')
    # table = replace_missed_season_with_NA(table, 'Pos')
    # table = replace_missed_season_with_NA(table, 'No.')
    # table = replace_missed_season_with_NA(table, 'Awards')


    # table = replace_missed_season_with_ZERO(table, 'G')#replaced missed seasons with 0 games played
    # table['G'] = table['G'].astype(int)#cast Games played to int
    
    # table = replace_missed_season_with_ZERO(table, 'GS')#replaced missed seasons with 0 games started
    # table['GS'] = table['GS'].astype(float)#cast Games started to int
    
    # table = replace_missed_season_with_ZERO(table, 'AV')#replaced missed seasons with 0 games started
    # table['AV'] = table['AV'].astype(float)#cast Games started to int
    
    # #get rid of rows where the player wasnt an offensive lineman
    # table = get_rid_of_invalid_positions(table, 'Pos',  ['T', 'C', 'LG', 'G', 'LT', 'RG', 'RT', 'LS', 'RT/RG', 
    #                                                     'LT/RT', 'N/A', 'RT-T', 'RG/C', 'LT/LG', 'LG/RG', 'RT/LG', 'LG/RT',
    #                                                     'C/LT', 'RG/LT', 'LT/RG', 'RG/LG', 'LT-T', 'C/RG', 'RG/RT', 'RT/LT', 
    #                                                     'G-RG', 'LG/C', 'C/LG', 'RG-T', 'G-LG', 'T/TE', 'G-LT', 'LG/LT',
    #                                                     'C-LG', 'G/T', 'C/T', 'G/C', 'LT/C', 'LG-RG', 'G-T', 'C/G', 'TOG', 'RT/LT-T', 'C/RT',
    #                                                     'C-LS', 'LG-T', 'C-G', 'RT/RDE', 'G-G/C', 'G-LG/RG-OL', 'RT/C', 'G-RT'])

    # table.to_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/O_Line_table_clean.csv')
        
    '''cleaning RB table'''
    table = pd.read_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics/RB_table.csv', index_col=0)
    table = table[table['Player ID'].notna()] #get rid of empty rows 
    columns_to_keep = ['Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'Att', 'Yds',
       'TD', '1D', 'Succ%', 'Lng', 'Y/A', 'Y/G', 'A/G', 'Tgt', 'Rec', 'Yds.1',
       'Y/R', 'TD.1', '1D.1', 'Succ%.1', 'Lng.1', 'R/G', 'Y/G.1', 'Ctch%',
       'Y/Tgt', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV', 'Player ID', 'Awards']
    table = remove_unecessary_columns(table, columns_to_keep)
    # table = add_injured_column(table)
    # table = table.groupby('Player ID').apply(fill_nan_age_values) #grouping by Player ID, fill the nan value of the age column
    # table = table.reset_index(drop=True) #need to do this to get rid of the playerID group by...

    # table = replace_missed_season_with_NA(table, 'Tm')
    # table = replace_missed_season_with_NA(table, 'Pos')
    # table = replace_missed_season_with_NA(table, 'No.')
    # table = replace_missed_season_with_NA(table, 'Awards')


    # table = replace_missed_season_with_ZERO(table, 'G')#replaced missed seasons with 0 games played
    # table['G'] = table['G'].astype(int)#cast Games played to int
    
    # table = replace_missed_season_with_ZERO(table, 'GS')#replaced missed seasons with 0 games started
    # table['GS'] = table['GS'].astype(float)#cast Games started to int
    
    # table = replace_missed_season_with_ZERO(table, 'AV')#replaced missed seasons with 0 games started
    # table['AV'] = table['AV'].astype(float)#cast Games started to int
    
    # #get rid of rows where the player wasnt an offensive lineman
    # table = get_rid_of_invalid_positions(table, 'Pos',  ['T', 'C', 'LG', 'G', 'LT', 'RG', 'RT', 'LS', 'RT/RG', 
    #                                                     'LT/RT', 'N/A', 'RT-T', 'RG/C', 'LT/LG', 'LG/RG', 'RT/LG', 'LG/RT',
    #                                                     'C/LT', 'RG/LT', 'LT/RG', 'RG/LG', 'LT-T', 'C/RG', 'RG/RT', 'RT/LT', 
    #                                                     'G-RG', 'LG/C', 'C/LG', 'RG-T', 'G-LG', 'T/TE', 'G-LT', 'LG/LT',
    #                                                     'C-LG', 'G/T', 'C/T', 'G/C', 'LT/C', 'LG-RG', 'G-T', 'C/G', 'TOG', 'RT/LT-T', 'C/RT',
    #                                                     'C-LS', 'LG-T', 'C-G', 'RT/RDE', 'G-G/C', 'G-LG/RG-OL', 'RT/C', 'G-RT'])

    # table.to_csv('/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/O_Line_table_clean.csv')
    
    print('here')

if __name__ == "__main__":
    main()
