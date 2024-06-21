import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



'''only keeps the columns in the table with the column names in columns_to_keep'''
def remove_unecessary_columns(df, position):
    if(position in ['WR', 'RB', 'TE']):
        columns_to_keep = ['Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'Tgt', 'Rec',
        'Yds', 'Y/R', 'TD', '1D', 'Succ%', 'Lng', 'R/G', 'Y/G', 'Ctch%',
        'Y/Tgt', 'Att', 'Yds.1', 'TD.1', '1D.1', 'Succ%.1', 'Lng.1', 'Y/A',
        'Y/G.1', 'A/G', 'Touch', 'Y/Tch', 'YScm', 'RRTD', 'Fmb', 'AV', 'Player ID', 'Awards']
    elif(position == "QB"):
        columns_to_keep = ['Player ID', 'Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS',
       'QBrec', 'Cmp', 'Att', 'Cmp%', 'Yds', 'TD', 'TD%', 'Int', 'Int%', 'Lng',
       'Y/A', 'AY/A', 'Y/C', 'Y/G', 'Rate', 'Sk', 'Yds.1', 'Sk%', 'NY/A',
       'ANY/A', '4QC', 'GWD', 'AV', '1D', 'Succ%', 'Awards', 'QBR']
    elif(position == "O_LINE"):
        columns_to_keep = ['Player ID', 'Name', 'Year', 'Age', 'Tm', 'Pos', 'No.', 'G', 'GS', 'AV', 'Awards']
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
def get_rid_of_invalid_positions(df, column_name, position):
    if(position == 'QB'):
        valid_positions = ['QB', 'N/A']
    elif(position == 'RB'):
        valid_positions = ['RB', 'FB', 'HB', 'HB/FB', 'HB-RB', 'RHB', 'RB/FB', 'FB-RB', 'FB/HB', 'RH', 'LH', 'LHB', 'RHB',  'N/A']
    elif(position == 'WR'):
        valid_positions = ['WR', 'N/A']
    elif(position == 'TE'):
        valid_positions = ['TE', 'N/A']
    elif(position == 'O_LINE'):
        valid_positions = ['T', 'C', 'LG', 'G', 'LT', 'RG', 'RT', 'RT/RG', 'LT/RT', 'N/A', 'RT-T', 'RG/C',
       'LT/LG', 'LG/RG', 'RT/LG', 'LG/RT', 'C/LT', 'RG/LT', 'LT/RG', 'RG/LG', 'LT-T', 'C/RG', 'RG/RT', 'RT/LT', 
       'G-RG', 'LG/C', 'C/LG', 'RG-T', 'G-LG', 'G-LT', 'LG/LT', 'C-LG', 'G/T', 'C/T', 'G/C', 'LT/C', 'LG-RG', 
       'G-T', 'C/G', 'RT/LT-T', 'C/RT', 'LG-T', 'C-G', 'G-G/C', 'G-LG/RG-OL', 'RT/C', 'G-RT']
    else:
        print('what position is this?')
        
    df = df[df[column_name].isin(valid_positions)]
    return df

'''identifies the columns which should contain numeric values like yards, games played, etc. So that we can replace "Missed Season" rows in the column with 0 later.'''
def identify_numeric_columns(table):
    # Function to check if a string can be converted to a float
    def is_convertible_to_float(value):
        try:
            float(value)
            return True
        except ValueError:
            if(value is np.nan): return True #nan values could be true...
            return False

    columns_to_convert = []
    for column in table.columns:
        # Check a sample of values to see if they should be numeric
        if table[column].apply(is_convertible_to_float).sum() > 0.75 * len(table):  # More than 50% convertible
            columns_to_convert.append(column)
    return columns_to_convert

'''Calculates the fantasy points for each row, and adds them as a separate column'''
def add_fantasy_points(table, position):
    #TODO DONT DO THIS YET UNTIL YOU HAVE MORE COMPLETE DATA... 
    # QB's FOR EXAMPLE DO NOT HAVE RUSHING YDS AND REC YARDS YET...
    # RB's FOR EXAMPLE DO NOT HAVE 2PT CONVERSIONS YET...
    '''
    1 pt per 25 passing yards
    4 pts per passing TD
    -2 pts per passing INT
    1 pt per 10 RUSH YDS
    6 pts per RUSH TD
    1 pt per REC
    1 point per 10 REC YDS
    6 pts per REC TD
    2 pts per 2-point conv
    -2 pts for fumble
    6 pts for fumble returned for TD
    '''
    if(position == 'RB'):
        pts_from_rush_yds = table['Yds'] / 10.0
        pts_from_RushandRec_TD = table['RRTD'] * 6.0
        pts_from_REC = table['Rec']
        pts_from_REC_YDS = table['Yds.1'] / 10
        pts_from_Fmb = table['Fmb'] * -2
        
    elif(position == 'WR'):
        pts_from_rush_yds = table['Yds.1'] / 10.0
        pts_from_RushandRec_TD = table['RRTD'] * 6.0
        pts_from_REC = table['Rec']
        pts_from_REC_YDS = table['Yds'] / 10
        pts_from_Fmb = table['Fmb'] * -2
    elif(position == 'O_LINE'):
        return table
        
    
    
    return table


def main():
    positions = ['QB', 'TE', 'WR', 'RB', 'O_LINE']
    positions = ['RB']
    for position in positions:
        # position = 'QB'
        table = pd.read_csv(f'/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics/{position}_table.csv', index_col=0)
        table = table[table['Player ID'].notna()] #get rid of empty rows 

        table = remove_unecessary_columns(table, position)    
        table = add_injured_column(table)
        table = table.groupby('Player ID').apply(fill_nan_age_values) #grouping by Player ID, fill the nan value of the age column
        table = table.reset_index(drop=True) #need to do this to get rid of the playerID group by...

        table = replace_missed_season_with_NA(table, ['Tm', 'Pos', 'No.', 'Awards'])

        numeric_columns = identify_numeric_columns(table)
        # table['Ctch%'] = table['Ctch%'].str.rstrip('%')  #get rid of % sign in Ctch% column
        # table['AV'] = pd.to_numeric(table['AV'], errors='coerce') #need to cast all AV to numeric that replace_missed_season_with_ZERO() can do its work.
        table = replace_missed_season_with_ZERO(table, numeric_columns)#replace missed seasons with 0 for all the columns in the array
        
        
        #get rid of rows where the player wasnt an WR
        table = get_rid_of_invalid_positions(table, 'Pos', position)
        print('here')
        
        #add fantasy_points column
        table = add_fantasy_points(table, position)
        
        table.to_csv(f'/Users/jakehirst/Desktop/fantasy_football_predictors/Yearly_statistics_clean/{position}_table_clean.csv')


if __name__ == "__main__":
    main()
