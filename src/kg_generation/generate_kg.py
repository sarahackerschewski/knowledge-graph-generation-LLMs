'''This file contains functions to create an ontology, a knowledge graph either in the two-step way or using LangChain, and extracting triples from the KG.
'''
import json
import re
from datetime import datetime
from typing import Dict, List, Tuple

from dotenv import load_dotenv
from langchain_community.graphs.graph_document import Node
from langchain_core.documents import Document
from langchain_experimental.graph_transformers import LLMGraphTransformer
from langchain_neo4j import Neo4jGraph
from langchain_openai import ChatOpenAI
from openai import OpenAI

from src.kg_generation.assistant_calls import (
    run_hierarchy_extraction_assistant,
    run_kg_generation_assistant,
    run_ontology_extraction_assistant,
    run_property_extraction_assistant,
)
from src.kg_generation.postprocess import (
    clean_graph,
    clean_ontology,
    merge_ontology_with_entities,
    merge_property_batches,
    remove_duplicates_in_kg,
)
from src.utils.file_utils import read_json

load_dotenv('data/config/.env')

# specify/connect to OpenAI model
client = OpenAI()
# create Neo4J graph
graph = Neo4jGraph()


def generate_triples(kg_file: str, include_properties: bool = False, langchain: bool = False) -> List[Tuple]:
    '''Generates triples from a knowledge graph JSON file, optionally including properties.

    Args:
        kg_file: The path to the knowledge graph JSON file.
        include_properties (optional): Flag to include properties in the triples. Defaults to False.
        langchain (optional): Flag indicating if kg was created with LangChain. Defaults to False.

    Returns:
        List[Tuple]: A list of triples, where each triple is a tuple containing the head, relation type, and tail.
    '''
    def convert_date(day: str = None, month: str = None, year: str = None, date: str = None) -> str:
        '''Converts various date formats into a standardized date string format.

        Args:
            day (optional): The day component of the date.
            month (optional): The month component of the date.
            year (optional): The year component of the date.
            date (optional): A string representing the date in various formats.

        Returns:
            str: The standardized date string in the format 'dd/mm/yyyy', 'mm/yyyy', or 'yyyy'.
        '''
        if date:
            try:
                if "AD" in date:
                    year = int(date.split()[1])
                    return str(year)
                elif len(date) == 4 and date.isdigit():
                    return date
                else:
                    date_obj = datetime.strptime(date, "%d %B %Y")
                    return date_obj.strftime("%d/%m/%Y")
            except ValueError:
                try:
                    # Try another format
                    date_obj = datetime.strptime(date, "%Y-%m-%d")
                    return date_obj.strftime("%d/%m/%Y")
                except ValueError:
                    patterns = [
                        (r'(\d{4})-(\d{2})$', '%Y-%m',
                         'mm/yyyy'),  # matches yyyy-mm
                        # matches month yyyy
                        (r'(\w+) (\d{4})', '%B %Y', 'mm/yyyy'),
                        (r'(\w+) (\d{1,2}), (\d{4})', '%B %d, %Y',
                         'dd/mm/yyyy'),  # matches month dd, yyyy
                        (r'(\d{2})\.(\d{2})\.(\d{4})', '%d.%m.%Y',
                         'dd/mm/yyyy'),  # matches dd.mm.yyyy
                    ]

                    for pattern, date_format, output_format in patterns:
                        match = re.match(pattern, date)
                        if match:
                            date_obj = datetime.strptime(date, date_format)
                            if output_format == 'dd/mm/yyyy':
                                return date_obj.strftime("%d/%m/%Y")
                            elif output_format == 'mm/yyyy':
                                return date_obj.strftime("%m/%Y")
                            elif output_format == 'yyyy':
                                return date_obj.strftime("%Y")
                    return date
        try:
            # handle not date string but day, month and year that are added separately in the Date node
            day = int(day) if day else 0
            month = int(month) if month else 0
            year = int(year) if year else 0
            if day and month and year:
                date = datetime(year, month, day)
                return date.strftime('%d/%m/%Y')
            elif month and year:
                return f'{month:02d}/{year}'
            elif year:
                return f'{year}'
        except ValueError:
            try:
                date = datetime.strptime(f'{day} {month} {year}', '%d %B %Y')
                return date.strftime('%d/%m/%Y')
            except:
                return f'{day}/{month}/{year}'

    kg = read_json(kg_file)
    triples = []
    if langchain:
        # handle langchain graph differently because of different KG structure
        for relation in kg['relationships']:
            if include_properties:
                triples.append(
                    (relation['source'], relation['type'], relation['target']))
            else:
                triples.append(
                    (relation['source']['id'], relation['type'], relation['target']['id']))
        return triples
    else:
        for relation in kg['relationships']:
            start_id = relation['startNode']
            end_id = relation['endNode']

            start = [node for node in kg['nodes'] if node['id'] == start_id]
            end = [node for node in kg['nodes'] if node['id'] == end_id]
            # get the node for the id
            if include_properties:
                head = start[0] if start else start_id
                tail = end[0] if end else end_id
            else:
                # use the corresponding property as the representive label of the node
                head = start[0]['properties'].get('title') or start[0]['properties'].get('name') or start[0]['properties'].get('type') or start[0]['properties'].get('scientificName') or start[0]['properties'].get(
                    'species') or start[0]['properties'].get('model') or start[0]['properties'].get('surName') or str(start[0]['properties'].get('value')) or start[0]['properties'].get('location') or start[0]['properties'].get('branch')
                if not head:
                    head = ''
                if end:
                    tail = end[0]['properties'].get('title') or end[0]['properties'].get('name') or end[0]['properties'].get('type') or end[0]['properties'].get('scientificName') or end[0]['properties'].get(
                        'species') or end[0]['properties'].get('model') or end[0]['properties'].get('surName') or str(end[0]['properties'].get('value')) or end[0]['properties'].get('location') or end[0]['properties'].get('branch')
                    # handle dates separately
                    if 'Date' in end[0]['labels']:
                        if 'year' in end[0]['properties'] and 'day' in end[0]['properties'] and 'month' in end[0]['properties']:
                            day = end[0]['properties'].get('day')
                            month = end[0]['properties'].get('month')
                            year = end[0]['properties'].get('year')
                            tail = convert_date(day, month, year)
                        elif 'year' in end[0]['properties']:
                            year = end[0]['properties'].get('year')
                            tail = convert_date(None, None, year)
                        elif 'month' in end[0]['properties'] and 'year' in end[0]['properties']:
                            month = end[0]['properties'].get('month')
                            year = end[0]['properties'].get('year')
                            tail = convert_date(None, month, year)
                        elif 'startYear' in end[0]['properties'] and 'endYear' in end[0]['properties']:
                            tail = f'''{end[0]['properties'].get(
                                'startYear')} - {end[0]['properties'].get('endYear')}'''
                        elif 'dateValue' in end[0]['properties']:
                            tail = convert_date(
                                date=end[0]['properties'].get('dateValue'))
                        elif not tail:
                            tail = ''

                        if isinstance(tail, str):
                            # Check if tail is a date string
                            match = re.match(
                                r'(\d{1,2}) (\w+) (\d{4})|(\d{1,2}) (\d{1,2}) (\d{4})|(\w+) (\d{4})|(\d{1,2}) (\d{4})|(\d{4})', tail)
                            if match:
                                groups = match.groups()
                                day, month, year = groups[:3] if groups[:3] != (
                                    None, None, None) else (None, None, groups[-1])
                                tail = convert_date(day, month, year)
                else:
                    tail = end_id

            triples.append((head, relation['type'], tail))
        return triples


def generate_ontology(data: List[str], openai_client: OpenAI, input_file: str, model: str = 'gpt-4o', n_onto=10):
    '''Generates an ontology by extracting entities and relations from articles, creating a hierarchy, and adding properties using OpenAI's assistant.

    Args:
        data : The list of data strings to be used for ontology extraction.
        openai_client: The OpenAI client for interacting with the assistant.
        input_file: The path to the input file containing the articles.
        model (optional): The model to be used for ontology extraction. Defaults to 'gpt-4o'.
        n_onto (optional): The number of ontology batches to extract. Defaults to 10.

    Returns:
        None
    '''
    # 'basic' ontology extraction with assistant
    print(f'---Starting basic ontology extraction for {input_file}---')
    run_ontology_extraction_assistant(data, openai_client, input_file, n_onto)
    print('---Basic Ontology Extraction done---')

    # clean ontology, i.e. combine all batches with little duplicates as possible
    batch_ontology_file = f'data/ontologies/batch_responses/batches_ontology_{
        model}.json'
    output_ontology_file = f'data/ontologies/merged_batches/ontology_{
        model}.json'
    clean_ontology(batch_ontology_file, output_ontology_file, openai_client)

    # create a hierarchy for the 'basic' ontology
    print(f'---Starting Hierarchy extraction for {basic_onto_file}---')
    basic_onto_file = f'data/ontologies/merged_batches/ontology_{model}.json'
    run_hierarchy_extraction_assistant(basic_onto_file)

    # use last batch as final hierarchy
    hierarchy = read_json(
        f'data/ontologies/batch_responses/batches_hierarchy_{model}.json')
    index = str(len(hierarchy)-1)
    hierarchy_file = f'data/ontologies/merged_batches/hierarchy_{model}.json'
    with open(hierarchy_file, 'w', encoding='utf-8') as f:
        json.dump(hierarchy[index], f)
    print('---Hierarchy Extraction done---')

    # add properties to the entities
    print(f'---Starting Property extraction for {hierarchy_file}---')
    run_property_extraction_assistant(hierarchy_file)
    print('---Property Extraction done---')

    # merging property batches
    property_file = f'data/ontologies/batch_responses/batches_properties_{
        model}.json'
    merge_property_batches(
        property_file, model)

    # replace the old entities in the basic ontology with the hierarchy with properties
    hierarchy_property_file = f'data/ontologies/merged_batches/hierarchy_property_{
        model}.json'
    final_onto_file = f'data/ontologies/merged_batches/final_ontology_{
        model}.json'
    merge_ontology_with_entities(basic_onto_file,
                                 hierarchy_property_file, f'data/ontologies/merged_batches/final_ontology_{model}.json')
    print(f'---Final ontology can be found at {final_onto_file}---')


def generate_graph(ontology_file: str, openai_client: OpenAI, article_file: str, model: str = 'gpt-4o', n: int = 1):
    '''Generates a knowledge graph by extracting data from articles using an ontology and OpenAI's assistant, and cleans the graph to remove duplicates.

    Args:
        ontology_file: The path to the ontology file.
        openai_client: The OpenAI client for interacting with the assistant.
        article_file: The path to the file containing articles.
        model (optional): The model to be used for knowledge graph generation. Defaults to 'gpt-4o'.
        n (optional): The number of batches to process. Defaults to 1.

    Returns:
        None
    '''
    run_kg_generation_assistant(
        ontology_file, openai_client, article_file, model, n)
    batch_file = f'data/knowledge_graphs/batch_responses/batches_kg_{
        model}.json'
    kg_file = f'data/knowledge_graphs/merged_batches/kg_{model}.json'
    # merge batches
    clean_graph(batch_file, kg_file)
    # remove duplicate nodes
    remove_duplicates_in_kg(kg_file, kg_file)


def generate_graph_langchain(articles: List,  model: str = 'gpt-4o', temperature: int = 0.2, n: int = 1):
    '''Generates a knowledge graph using LangChain from a list of articles, and saves the nodes and relationships to JSON files.

    Args:
        articles: A list of articles to be processed.
        model (optional): The model to be used for knowledge graph generation. Defaults to 'gpt-4o'.
        temperature (optional): The temperature setting for the language model. Defaults to 0.2.
        n (optional): The number of articles to process in each batch. Defaults to 1.

    Returns:
        None
    '''
    llm = ChatOpenAI(temperature=temperature, model_name=model)
    llm_transformer = LLMGraphTransformer(
        llm=llm, node_properties=True, relationship_properties=True)
    batches = [articles[i:i + n]for i in range(0, len(articles), n)]
    nodes = []
    relations = []
    for i, batch in enumerate(batches):
        print(f'Batch {i+1} of {len(batches)}')
        documents = convert_to_document(batch)
        # extract nodes and relations from the documents
        graph_documents = llm_transformer.convert_to_graph_documents(documents)
        # Store information into a graph
        graph.add_graph_documents(graph_documents)
        # graph needs to be refreshed after batch was added
        graph.refresh_schema()
        # save nodes and relations in same structure like auto KGs
        for node in graph_documents[0].nodes:
            nodes.append(dict(node))
        for rel in graph_documents[0].relationships:
            rel = dict(rel)
            relations.append(
                {
                    'source': node_to_dict(rel['source']),
                    'target': node_to_dict(rel['target']),
                    'type': rel['type'],
                    'properties': rel['properties']
                }
            )
        with open(f'data/knowledge_graphs/batch_responses/batches_langchain_kg_{model}.json', 'a', encoding='utf-8') as f:
            json.dump({i: {'nodes': nodes, 'relationships': relations}}, f)
    print(f'Nodes:{nodes}\n')
    print(f'Relationships:{relations}')
    with open(f'data/ontologies/langchain_onto_{model}.json', 'w', encoding='utf-8') as f:
        json.dump(graph.get_structured_schema, f)


def convert_to_document(articles: List) -> List[Document]:
    '''Converts a list of articles into a list of Document objects.

    Args:
        articles: A list of dictionaries, where each dictionary represents an article with a 'cleaned_text' key.

    Returns:
        List[Document]: A list of Document objects created from the cleaned text of the articles.
    '''
    return [Document(page_content=article['cleaned_text']) for article in articles]


def node_to_dict(node: Node) -> Dict:
    '''Converts a node object into a dictionary representation.

    Args:
        node: The node object to be converted.

    Returns:
        Dict: A dictionary representation of the node, containing its id, type, and properties.
    '''
    return {
        'id': node.id,
        'type': node.type,
        'properties': node.properties
    }
