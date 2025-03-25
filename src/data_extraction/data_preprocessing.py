'''This file contains functions for processing wikidata dump files, cleaning the articles and creating a final clean dataset
'''
import re

import pandas as pd
from wiki_dump_reader import Cleaner, iterate


def create_clean_dataset(input_file: str, output_file: str):
    '''Creates a cleaned dataset from an input file, cleans the text, extracts links, removes invalid articles
    and writes it to an output file.

    Args:
        input_file (str): The path to the input file containing the raw data.
        output_file (str): The path to the output file where the cleaned data will be saved.

    Returns:
        None

    Example:
        create_clean_dataset('raw_data.json', 'cleaned_data.json')

    '''
    titles = []
    texts = []
    urls = []

    # wiki dump reader used for reading the input file and clean  of links, images etc.
    cleaner = Cleaner()
    for title, text in iterate(input_file):
        text = cleaner.clean_text(text)
        cleaned_text, links = cleaner.build_links(text)
        titles.append(title)
        texts.append(cleaned_text)
        urls.append(links)
    df = pd.DataFrame({'title': titles, 'text': texts, 'urls': urls})
    # removing articles only containing 'redirect'
    df = remove_invalid_articles(df)
    # removing titles with '==', non-english char  etc. and write to file
    df = clean_articles(df)
    df.to_json('../data/' + output_file, orient='records')


def remove_invalid_articles(data: pd.DataFrame) -> pd.DataFrame:
    '''Removes invalid articles from the dataset, i.e. articles that contain the word 'redirect'

    Args:
        data : The DataFrame containing the articles to be filtered.

    Returns:
        The filtered DataFrame with invalid articles removed.

    Example:
        cleaned_data = remove_invalid_articles(data)
    '''
    filter = data['text'].str.contains('redirect', case=False, na=False)
    filtered_df = data[~filter]
    return filtered_df


def clean_articles(data: pd.DataFrame) -> pd.DataFrame:
    '''Cleans the articles in the dataset by extracting text units and removing non-English characters.

    Args:
        data : The DataFrame containing the articles to be cleaned.

    Returns:
        The DataFrame with an additional column 'cleaned_text' containing the cleaned articles.

    Example:
        cleaned_data = clean_articles(data)
    '''
    data['cleaned_text'] = data['text'].apply(lambda x: _get_text_unit(
        str(x))).apply(lambda x: _remove_non_english_char(str(x)))
    return data


def _get_text_unit(text: str) -> str:
    '''Removes sections of text enclosed in '==' characters and joins non-empty lines into a single string.

    Args:
        text : The input string to be processed.

    Returns:
        The cleaned text with sections removed and lines joined.

    Example:
        cleaned_text = _get_text_unit(
            '==History==\nParagraph about the history\n of a topic.')
        # Returns 'Paragraph about the history of a topic.'
    '''
    if '==' in text:
        text = text.replace(text[text.index('='):text.rindex('=')+1], '')
    text_lst = [line for line in text.split('\n') if line.strip()]
    text = ' '.join(text_lst)
    return text


def _remove_non_english_char(text: str) -> str:
    '''Removes non-English characters from a given string.

    Args:
        text: The input string to be processed.

    Returns:
        The cleaned text with non-English characters removed.

    Example:
        cleaned_text = _remove_non_english_char('English, 日本語!')
        # Returns 'English, !'
    '''
    text = re.sub(r'[^\u0000-\u05C0\u2100-\u214F]+', '', text)

    # if ( ) left over
    text = re.sub(r'\s\(\s\)\s', ' ', text)
    return text


def sample_dataset(file: str, n: int = 1000, random_state: int = 25):
    '''Samples a subset of the dataset from a JSON file and saves it to a new JSON file.

    Args:
        file: The path to the input JSON file containing the dataset.
        n (optional): The number of samples to extract from the dataset. Defaults to 1000.
        random_state (optional): The seed for random number generation to ensure reproducibility. Defaults to 25.

    Returns:
        None

    '''
    df = pd.read_json(file)
    df_sample = df.sample(n, random_state=random_state)
    df_sample.to_json(f'{file.split('.'[0])}_sample.json', orient='records')
