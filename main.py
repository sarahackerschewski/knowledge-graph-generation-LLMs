'''This script is the main script for automatic knowledge graph (KG) generation from Wikipedia articles.

This main script contains:
    - a newly introduced two-step KG generation approach, generating an ontology and using that ontology along the articles for KG generation
    - an existing KG generation approach using LangChainsLLMGraphTransformer
    - calls to create a dataset, run the KG generation approaches, run gold triple extraction from Wikidata & DBpedia, and run the evaluation of these graphs
Usage:
    Run the script with:
    'python main.py -[method to use] -[model to use] -[eval_only] -[evaluation methods to use]'
    [method to use] -> auto for two-step approach, langchain for langchain
    [model to use] -> gpt-4o, gpt-4-turbo, gpt-4o-mini
    [eval_only] -> (optional) if only evaluation should be performed
    [evaluation methods to use] -> (optional: Defaults to all are used) otherwise any of the methods to use [content, structural, statistics]
'''

import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

from src.data_extraction.data_preprocessing import create_clean_dataset, sample_dataset
from src.data_extraction.dbpedia_extraction import run_triple_extraction_dbpedia
from src.data_extraction.wikidata_extraction import run_triple_extraction_wikidata
from src.evaluate import run_evaluation
from src.kg_generation.generate_kg import (
    generate_graph,
    generate_graph_langchain,
    generate_ontology,
    generate_triples,
)
from src.utils.file_utils import read_json
from src.utils.openai_utils import init_openai_client

WIKIDATA_PATH = Path('data/gold_triples/wikidata/gold_triples.json')
DBPEDIA_PATH = Path('data/gold_triples/dbpedia/gold_triples.json')
# not in repo due to size, download from GoogleDrive (see ReadME)
WIKIDUMP_PATH = Path(
    'data/enwiki-20240520-pages-articles-multistream23.xml-p49288942p50564553')
# not in repo due to size, download from GoogleDrive (see ReadME)
CLEAN_DATASET_PATH = Path('data/20240520_wikipedia_cleaned.json')
SAMPLE_PATH = Path('data/20240520_wikipedia_cleaned_sample.json')
ONTO_PATH_INIT = 'data/ontologies/merged_batches/final_ontology_'
KG_PATH_INIT = 'data/knowledge_graphs/merged_batches/kg_'


def create_sample(n: int = 1000, random_state: int = 25):
    '''Create a sample dataset from a Wikipedia dump. 

    Args:
        n (optional): The number of samples to create. Defaults to 1000.
        random_state (optional): Seed for randomizing and sampling the data. Defaults to 25 to replicate the dataset of this thesis

    Returns:
        None
    '''
    # create a dataset JSON file from the Wikipedia dump if it was not created before
    if not CLEAN_DATASET_PATH.exists():
        create_clean_dataset(WIKIDUMP_PATH, CLEAN_DATASET_PATH)

    # sample dataset if it was not created before
    if not SAMPLE_PATH.exists():
        sample_dataset(CLEAN_DATASET_PATH, n, random_state)
    print('Created Sample for the Wikipedia articles')


def run_automatic_kg_generation_process(input_file: str, model: str = 'gpt-4o', n_onto: int = 100, n_kg: int = 1):
    '''Automatically generate an ontology and a knowledge graph using the specified model.

    Args:
        input_file: The path to the JSON file containing the input data.
        model (optional): The model to use for generation. Defaults to 'gpt-4o'.
        n_onto (optional): The number of ontology generations to perform. Defaults to 100.
        n_kg (optional): The number of knowledge graph generations to perform. Defaults to 1.

    Returns:
        None
    '''
    load_dotenv('data/config/.env')
    data = read_json(input_file)

    # initialize openai client to use
    openai_client = init_openai_client()
    print('Starting two-step KG Generation')
    # generate ontology
    print('-Starting Step 1: Ontology Generation-')
    generate_ontology(data, openai_client, input_file, model, n_onto)
    print('-Finished Step 1: Ontology Generation-')
    # generate KG
    print('-Starting Step 2: Knowledge Graph Generation-')
    onto_file = f'data/ontologies/merged_batches/final_ontology_{model}.json'
    generate_graph(onto_file, openai_client, input_file, model, n_kg)
    print('-Finished Step 2: Knowledge Graph Generation-')

    print('Finished two-step KG Generation')


def run_langchain_kg_generation_process(input_file: str, model: str = 'gpt-4o', temperature: int = 0.2, n: int = 1):
    '''Generate a knowledge graph using LangChain based on the input data.

    Args:
        input_file: The path to the JSON file containing the input data.
        model (optional): The model to use for LangChain generation. Defaults to 'gpt-4o'.
        temperature (optional): The temperature setting for the model. Defaults to 0.2.
        n (optional): The number of generations to perform. Defaults to 1.

    Returns:
        None
    '''
    data = read_json(input_file)
    print('Starting LangChain KG Generation')
    generate_graph_langchain(data, model, temperature, n)
    print('Finished LangChain KG Generation')


def run_evaluation_process(article_file: str, ontology_file: str, kg_file: str, langchain: bool = False, content: bool = True, structural: bool = True, statistics: bool = True):
    '''Run the evaluation process for an ontology and a knowledge graph.

    Args:
        article_file: The path to the JSON file containing the articles.
        ontology_file: The path to the JSON file containing the ontology.
        kg_file: The path to the JSON file containing the knowledge graph.
        langchain (optional): Flag indicating whether to use LangChain-specific processing. Defaults to False.
        content (optional): Flag indicating whether to run content evaluation. Defaults to True.
        structural (optional): Flag indicating whether to run structural evaluation. Defaults to True.
        statistics (optional): Flag indicating whether to run statistical evaluation. Defaults to True.

    Returns:
        None
    '''
    articles = read_json(article_file)
    # Scrape Triples from Wikidata and DBpedia if triples do not exist already
    if not WIKIDATA_PATH.exists():
        if langchain:
            kg_triples = generate_triples(kg_file, langchain=langchain)
        else:
            kg_triples = generate_triples(kg_file)
        run_triple_extraction_wikidata(kg_triples)
    if not DBPEDIA_PATH.exists():
        if langchain:
            kg_triples = generate_triples(kg_file, langchain=langchain)
        else:
            kg_triples = generate_triples(kg_file)
        run_triple_extraction_dbpedia(kg_triples)

    run_evaluation(articles, ontology_file, kg_file, content=content,
                   structural=structural, statistics=statistics, langchain=langchain)


if __name__ == '__main__':

    if len(sys.argv) < 2:
        print('''Arguments are missing.\n
        Usage:python main.py -[method to use] -[model to use] -[eval_only] -[evaluation methods to use]\n'
        [method to use] -> auto for two-step approach, langchain for langchain\n
        [model to use] -> gpt-4o, gpt-4-turbo, gpt-4o-mini\n
        [eval_only] -> if only evaluation should be performed
        The following are optional. If none are added, all evaluations will be performed
        [statistics] -> (optional) if statistical evaluation should be run
        [structural] -> (optional) if structual evaluation should be run
        [content] -> (optional) if content/knowledge evaluation should be run
        ''')
        sys.exit(1)

    method = sys.argv[1].strip('-')  # '-auto' or '-langchain'
    model = sys.argv[2].strip('-')  # 'gpt-4o', 'gpt-4-turbo' or 'gpt-4o-mini'

    # set the parameters for the experiment
    langchain = False
    if method == 'langchain':
        langchain = True
    evaluation_only = False
    # if evaluation are given, set these to true, else all evaluation methods will be applied
    if len(sys.argv) >= 3:
        if 'eval_only' in sys.argv[3]:
            evaluation_only = True
        if len(sys.argv) >4: 
            evaluation = [arg.strip('-') for arg in sys.argv[3:]]
            content = False
            structural = False
            statistics = False
            if 'content' in evaluation:
                print('here')
                content = True
            if 'structural' in evaluation:
                structural = True
            if 'statistics' in evaluation:
                statistics = True
        else:
            content = True
            structural = True
            statistics = True

    if not evaluation_only:
        # Create the dataset (if it does not exist already)
        print('Create dataset')
        create_sample()

        # Generate the knowledge graph
        print('Knowlegde Graph Generation')
        if langchain:
            run_langchain_kg_generation_process(SAMPLE_PATH, model=model)
        else:
            run_automatic_kg_generation_process(SAMPLE_PATH, model=model)

    # Evaluate the graph
    onto_file = ONTO_PATH_INIT + f'{model}.json'
    kg_file = KG_PATH_INIT + f'{model}.json' if not langchain else KG_PATH_INIT.replace(
        'kg_', 'langchain_kg_') + f'{model}.json'
    run_evaluation_process(SAMPLE_PATH, onto_file, kg_file, langchain=langchain,
                           content=content, structural=structural, statistics=statistics)
