'''This file contains functions to make calls to the OpenAI assistant API
'''
import json
import re
from typing import Dict, List

from openai import OpenAI

from src.utils.file_utils import read_json
from src.utils.openai_utils import call_assistant, delete_assistant_thread


def run_ontology_extraction_assistant(data: List[Dict], openai_client: OpenAI, model: str,  n: int = 10):
    '''Runs an OpenAI assistant to extract the ontology from text.

    This function processes the input data in batches, sends each batch to an OpenAI assistant for ontology extraction,
    and writes the responses to a JSON file.

    Args:
        data: A list of dictionaries containing the articles to be processed. Each dictionary should have a 'cleaned_text' key.
        openai_client: An instance of the OpenAI client used to interact with the OpenAI API.
        model: OpenAI model used by the assistant
        n (optional): The number of items to include in each batch. Defaults to 10.

    Returns:
        None
    '''

    batches = [data[i:i + n]for i in range(0, len(data), n)]
    for i, batch in enumerate(batches):
        # create list of strings as input
        batch = [item['cleaned_text'] for item in batch]
        print(f'Batch {i+1} of {len(batches)}')
        # create and/or write json objects to csv file
        outfile = f'data/ontologies/batch_responses/batches_ontology_{
            model}.json'
        with open(outfile, 'a') as f:
            batch_response, thread_id = call_assistant(openai_client, str(
                batch), assistant_id='asst_LuwkLqMtfZFkMxmBLvTfwVVq')
            # thread is deleted after each batch to prevent overlap of articles
            delete_assistant_thread(openai_client, thread_id)
            print(batch_response)
            # remove \n and whitespaces
            batch_response = re.sub(r'\s*\n\s*', '', batch_response)
            json_batch = json.loads(batch_response)
            json_obj = json.dumps({i: json_batch})
            f.write(json_obj)


def run_hierarchy_extraction_assistant(ontology: str, openai_client: OpenAI, model: str, n: int = 100):
    '''Runs an OpenAI assistant to extract the hierarchy for the entities in the ontology..

    This function processes the input data in batches, sends each batch to an OpenAI assistant for hierarchy extraction,
    and writes the responses to a JSON file.

    Args:
        ontology: The file containing the already extracted ontology
        openai_client: An instance of the OpenAI client used to interact with the OpenAI API.
        model: OpenAI model used by the assistant
        n (optional): The number of items to include in each batch. Defaults to 100.

    Returns:
        None
    '''
    data = read_json(ontology)
    batches = [data['entities'][i:i + 100]
               for i in range(0, len(data['entities']), 100)]
    tmp_hierarchies = {}

    for i, batch in enumerate(batches):
        print(f'---------Batch {i+1} of {len(batches)}---------')
        with open(f'data/ontologies/batch_responses/batches_hierarchy_{model}.json', 'a', encoding='utf-8') as f:
            if i == 0:
                # use prompt for initialization
                response, thread_id = call_assistant(
                    openai_client, str(batch), assistant_id='asst_NkfsaFqZ1GeGqxuZ3GaSieFH')
                # thread is deleted after each batch to prevent overlap of articles
                delete_assistant_thread(openai_client, thread_id)
                print(response)
            else:
                # use prompt for batches
                input = f'''Here is the list {
                    batch}.\nHere is the hierarchy {tmp_hierarchies[i-1]}'''
                response, thread_id = call_assistant(
                    openai_client, input, assistant_id='asst_IusmlEOBwlsRuQbZEMmgl3g6')
                print(response)
            json_batch = json.loads(response)
            tmp_hierarchies[i] = json_batch
            json_obj = json.dumps({i: json_batch})
            f.write(json_obj)
    # delete thread for batch hierarchy extraction later so that hierarchy is `remembered` by model
    delete_assistant_thread(openai_client, thread_id)


def run_property_extraction_assistant(hierarchy: str, openai_client: OpenAI, model: str):
    '''Runs an OpenAI assistant to extract the properites for the entities in the hierarchical ontology structure.

    This function processes the input data in batches, sends each batch to an OpenAI assistant for property extraction,
    and writes the responses to a JSON file.

    Args:
        hierarchy: The file containing the already extracted hierarchy
        openai_client: An instance of the OpenAI client used to interact with the OpenAI API.
        model: OpenAI model used by the assistant

    Returns:
        None
    '''
    data = read_json(hierarchy)

    # create batches from the hierarchy dictionary
    hierarchy_list = list(data.items())
    batches = [dict(hierarchy_list[i:i + 1])
               for i in range(0, len(hierarchy_list), 1)]

    for i, batch in enumerate(batches):
        print(f'---------Batch {i+1} of {len(batches)}---------')
        with open(f'data/ontologies/batch_responses/batches_properties_{model}.json', 'a', encoding='utf-8') as f:
            response, thread_id = call_assistant(
                openai_client, str(batch), assistant_id='asst_8e4SDjWK2sseBf8VqvVOs9he')
            print(response)
            json_batch = json.loads(response)
            json_obj = json.dumps({i: json_batch})
            f.write(json_obj)
    # delete thread for batch hierarchy extraction later so that hierarchy & properties (for inheritance) are `remembered` by model
    delete_assistant_thread(openai_client, thread_id)


def run_kg_generation_assistant(ontology_file: str, openai_client: OpenAI, article_file: str, model: str, n: int = 1):
    '''Runs an OpenAI assistant to extract the a knowledge graph from text with the ontology as additional input.

    This function processes the input data in batches, sends each batch with the ontology to an OpenAI assistant for kg extraction,
    and writes the responses to a JSON file.

    Args:
        ontology_file: The file containing the ontology
        openai_client: An instance of the OpenAI client used to interact with the OpenAI API.
        article_file: The file containing the articles to be processed. 
        model: OpenAI model used by the assistant
        n (optional): The number of items to include in each batch. Defaults to 1.

    Returns:
        None
    '''
    ontology = read_json(ontology_file)
    articles = read_json(article_file)

    batches = [articles[i:i + n]for i in range(0, len(articles), n)]
    for i, batch in enumerate(batches):
        # create list of strings as input
        batch = [item['cleaned_text'] for item in batch]
        print(f'Batch {i+1} of {len(batches)}')
        # create and/or write json objects to csv file
        outfile = f'data/knowledge_graphs/batch_responses/batches_kg_{
            model}.json'
        with open(outfile, 'a', encoding='utf-8') as f:
            input = f'''Here is the list:{batch}
                \nHere is the ontology:{ontology}'''
            response, thread_id = call_assistant(
                openai_client, input, assistant_id='asst_pXaNvG0P4vQ3dEK87ngdix3x')
            print(response)
            json_batch = json.loads(response)
            json_obj = json.dumps({i: json_batch})
            f.write(json_obj)
            # thread is deleted after each batch to prevent overlap of articles
            delete_assistant_thread(openai_client, thread_id)
