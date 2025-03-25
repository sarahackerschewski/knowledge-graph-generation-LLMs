'''This file contains functions to open files
'''
import json
from typing import Dict, List


def read_json(json_file: str) -> List[Dict]:
    '''Reads a JSON file and returns its contents as a list of dictionaries.

    Args:
        json_file : The path to the JSON file to be read.

    Returns:
        List[Dict]: The contents of the JSON file
    '''
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)

    return data


def read_batch_json(batch_file: str) -> List[Dict]:
    '''Reads a JSON file, which was created in batches

    Args:
        json_file : The path to the JSON file to be read.

    Returns:
        List[Dict]: The contents of the JSON file
    '''
    with open(batch_file, 'r') as f:
        file_data = f.read()
        file_data = file_data.replace('}{', ',')
        json_content = json.loads(file_data)

    return json_content
