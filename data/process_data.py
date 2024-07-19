import sys

import pandas as pd
from sqlalchemy import create_engine


def load_data(messages_filepath, categories_filepath) -> pd.DataFrame:
    """Loads datasets into dataframe.

    :param messages_filepath: str path to messages file
    :param categories_filepath: str path to categories file
    :return: dataframe
    """
    messages_df = pd.read_csv(messages_filepath)
    categories_df = pd.read_csv(categories_filepath)
    df = pd.merge(messages_df, categories_df, on='id')
    return df


def clean_data(df):
    """Cleans dataframe for duplicates and category names.

    :param df: dataframe
    :return: cleaned dataframe
    """
    categories = df['categories'].str.split(';', expand=True)
    # select the first row of the categories dataframe
    row = categories.iloc[0]

    # use this row to extract a list of new column names for categories.
    # one way is to apply a lambda function that takes everything
    # up to the second to last character of each string with slicing
    category_colnames = [r.split('-')[0] for r in row]
    categories.columns = category_colnames

    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].apply(lambda x: x.split('-')[-1])

        # convert column from string to numeric
        categories[column] = categories[column].astype(int)

    df = df.drop('categories', axis=1)
    df = pd.concat([df, categories], axis=1)
    df = df.drop_duplicates(keep='first')
    df = df[(df != 2).all(axis=1)]
    return df


def save_data(df, database_filename):
    """Saves dataframe to database.

    :param df: dataframe
    :param database_filename: name of sqlite database file
    :return: None
    """
    engine = create_engine(f'sqlite:///{database_filename}')
    df.to_sql('disaster_response_cleaned', engine, index=False, if_exists='replace')


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()
