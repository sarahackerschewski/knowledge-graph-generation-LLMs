'''This file contains functions to query Wikidata to return the triples for the articles in the dataset, and to postprocess the extracted triples
'''
import csv
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
import requests
from dateutil.parser import parse
from lxml import html
from rdflib import Graph, URIRef
from SPARQLWrapper import JSON, SPARQLWrapper
from wikidata.client import Client

from src.utils.file_utils import read_batch_json, read_json


def extract_triples_wikidata(entity) -> List[Tuple[str, str, str]]:
    '''Extracts triples for a given entity from Wikidata using SPARQL queries and returns them.

    Args:
        entity: The entity for which triples are to be extracted.

    Returns:
       List[Tuple[str, str, str]]: A list of processed and mapped triples (subject, predicate, object).
    '''
    endpoint_url = 'https://query.wikidata.org/sparql'
    client = Client()

    # sparql query for retrieving English triples for the entity from Wikidata
    query = f"""
        SELECT ?subject ?subjectLabel ?predicate ?predicateLabel ?object ?objectLabel
        WHERE {{
        ?subject rdfs:label "{entity}"@en .
        ?subject ?predicate ?object .
        FILTER (lang(?object) = 'en' || !isLiteral(?object))
        SERVICE wikibase:label {{ bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". }}
        }}
    """
    # necessary headers for retrieving data from wikidata
    headers = {
        'User-Agent': 'KG_Construction',
        'Accept': 'application/sparql-results+json'
    }
    time.sleep(1)
    response = requests.get(endpoint_url, params={
                            'query': query}, headers=headers)
    # check that no error occured while querying
    if response.status_code != 200:
        return []

    data = response.json()
    triples = []
    seen_triples = set()

    if not data['results']['bindings']:
        # get triples from entity id if the other query did not work
        with open('data/gold_triples/wikidata/id_entity_mapping.csv', 'r', encoding='utf-8') as f:
            csvreader = csv.reader(f, delimiter=';')
            articles = dict((rows[0], rows[1]) for rows in csvreader)
            if entity in articles:
                data = _extract_triples_sparql(articles[entity])

    # get subject, predicate, object from the query result
    for i, item in enumerate(data['results']['bindings']):
        subject = item['subjectLabel']['value'] if 'subjectLabel' in item else item['subject']['value']
        predicate = item['predicateLabel']['value'] if 'predicateLabel' in item else item['predicate']['value']
        obj = item['objectLabel']['value'] if 'objectLabel' in item else item['object']['value']
        tmp_triple = (subject, predicate, obj)

        triple = []
        # handle different link cases to get a triple only consisting of sbj, pred, & obj labels not links
        for i, link in enumerate(tmp_triple):
            if 'wikidata.org/prop/' in link:
                # handle a specific Wikidata property
                label = _get_wikidata_property_label(link)
            elif 'wikidata.org/entity/Q' in link:
                # handle a specific Wikidata entity
                time.sleep(1)
                entity = client.get(link.split('/')[-1], load=True)
                label = str(entity.label)
            elif 'statement' in link:
                # handle a Wikidata `statement`, which often refers to a property value for an entity
                base_url = tmp_triple[i-2].replace(
                    'entity', 'wiki') if 'wikidata.org/entity/Q' in tmp_triple[i-2] else _get_wikidata_link(triple[i-2])
                prev_link = tmp_triple[i-1].split('/')[-1]
                value = _get_inner_html(base_url, prev_link, link)
                label = str(value)
            elif 'w3' in link:
                # handle links from the w3 schema
                label = _get_w3label(link)
            elif 'schema' in link:
                # handle schema links which already contain the label in the link
                label = link.split(
                    '#')[-1] if '#' in link else link.split('/')[-1]
            elif '//' not in link:
                # if not a link, label was already extracted
                label = link

            # image and freebase ID not relevant for comparison
            if label.endswith('ID') or label == 'image':
                break
            else:
                triple.append(label)
            time.sleep(1)
        if len(triple) == 3:
            triple_tuple = tuple(triple)
            if triple_tuple not in seen_triples:
                triples.append(triple_tuple)
                seen_triples.add(triple_tuple)
    return triples


def _get_all_article_title(filename: str) -> List[str]:
    '''Retrieves all article titles from a JSON file.

    Args:
        filename: The path to the JSON file containing the article data.

    Returns:
        List: A list of article titles.
    '''
    data = read_json(filename)
    return [article['title'] for article in data]


def _get_inner_html(link: str, prop_id: str, statement: str) -> str:
    '''Extracts the inner HTML content of a specific div from a webpage.

    Args:
        link (str): The URL of the webpage to fetch.
        prop_id (str): The id of the div to locate.
        statement (str): A statement used to modify the link if the div is not found.

    Returns:
        str: The extracted content from the div, or an error message if the div is not found.
    '''
    # get the html structure for the link
    response = requests.get(link)
    response.raise_for_status()
    tree = html.fromstring(response.content)
    # find the div with the specific id
    div = tree.xpath(f"//div[@id='{prop_id}']")
    # if div not in the properties, try with other entity link
    if not div:
        new_q_id = statement.split('/')[-1].split('-')[0]
        pattern = r'Q\d+'
        link = re.sub(pattern, new_q_id, link)
        response = requests.get(link)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        div = tree.xpath(f"//div[@id='{prop_id}']")
    try:
        # extract and return the inner HTML of the div and its sub-divs
        inner_html = html.tostring(div[0], pretty_print=True).decode('utf-8')
        # specific divs that contain the values for the property div
        start = inner_html.find('<div class="wikibase-snakview-value wikibase-snakview-variation-valuesnak">') + \
            len('<div class="wikibase-snakview-value wikibase-snakview-variation-valuesnak">')
        end = inner_html.find('</div>', start)
        value = inner_html[start:end].strip()

        tree = html.fromstring(value)

        # extract text content from <a> tags to get the label
        content_list = tree.xpath('//a/text()')
        if content_list:
            return content_list[0]

        # extract href attribute from <a> tags to get the label
        content_list = tree.xpath('//a/@href')
        if content_list:
            if 'Special:Map' in content_list[0]:
                return content_list[0].split('/')[-3] + '|' + content_list[0].split('/')[-2]
            return content_list[0]

        # extract dates
        if 'Gregorian' in value:
            return value.split('<')[0]
        elif 'wb-monolingualtext-value' in value:
            clean_text = re.sub(r'<.*?>', '', value)
            return clean_text.strip().split('(')[0]
        return value
    except:
        return ''


def _get_w3label(uri: str) -> str:
    '''Retrieves the label from a w3 URI

    Args:
        uri: The URI of the resource to query for.

    Returns:
        str: The label of the resource, or None if no label is found.
    '''
    # initialize and parse an RDF graph to obtain the w3 schema rdf label via querying
    g = Graph()
    g.parse(uri)

    property_uri = URIRef(uri)
    query = f"""
        SELECT ?label
        WHERE {{
            <{property_uri}> rdfs:label ?label .
        }}
    """
    results = g.query(query)
    for row in results:
        return str(row.label)


def _get_wikidata_property_label(uri: str) -> str:
    '''Retrieves the English label for a given Wikidata property URI.

    Args:
        uri (str): The URI of the Wikidata property.

    Returns:
        str: The English label of the Wikidata property.
    '''
    time.sleep(1)
    # search the prop id in the link
    property_id = re.search(r'P\d+', uri).group()

    params = {
        'action': 'wbgetentities',
        'ids': property_id,
        'format': 'json',
        'props': 'labels',
        'languages': 'en'
    }
    # extract the label for that prop id
    response = requests.get(
        'https://www.wikidata.org/w/api.php', params=params)
    data = response.json()

    return data['entities'][property_id]['labels']['en']['value']


def _get_wikidata_link(article_name: str) -> str:
    '''Retrieves the Wikidata link for a given article name.

    This function searches Wikidata for an entity matching the provided article name and
    returns the corresponding Wikidata link.

    Args:
        article_name (str): The name of the article to search for.

    Returns:
        str: The Wikidata link for the article, or a message if no link is found.
    '''
    url = 'https://www.wikidata.org/w/api.php'
    params = {
        'action': 'wbsearchentities',
        'search': article_name,
        'language': 'en',
        'format': 'json'
    }
    # request the wikidata entity id for an article
    response = requests.get(url, params=params)
    data = response.json()

    if data['search']:
        entity_id = data['search'][0]['id']
        # add id to the link to obtain wikidata link for the article
        wikidata_link = f'https://www.wikidata.org/wiki/{entity_id}'
        return wikidata_link
    else:
        return ''


def _extract_triples_sparql(entity_id) -> Dict:
    '''Extracts triples for a given entity from Wikidata using SPARQL queries and returns them in JSON format.

    Args:
        entity_id (str): The Wikidata entity ID for which triples are to be extracted.

    Returns:
        Dict: A dictionary containing the extracted triples in JSON format.

    '''
    sparql = SPARQLWrapper('https://query.wikidata.org/sparql')
    # query to extract English triples using the entity id
    query = f"""
    CONSTRUCT {{?s ?p ?o}}
    WHERE {{
      BIND(wd:{entity_id} AS ?s)
      ?s ?p ?o .
      FILTER (lang(?o) = 'en' || !isLiteral(?o))
    }}"""

    time.sleep(1)
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)

    return sparql.query().convert()


def postprocessing_triples_wikidata(triple_file: str):
    '''Postprocesses triples extracted from Wikidata, maps relations, nationalities, & dates, and saves the final triples to a JSON file.

    Args:
        triple_file (str): The path to the JSON file containing the extracted triples.

    Returns:
        List[Tuple[str, str, str]]: A list of processed and mapped triples (subject, predicate, object).
    '''
    triple_dict = read_json(triple_file)
    final_triples = []
    # open the file containing the relation mappings
    df = pd.read_excel('data/gold_triples/wikidata/relation_mapping.xlsx',
                       header=None, index_col=None, engine='openpyxl')
    mapping = dict(zip(df[0], df[1]))
    # open the file containing the nationality mappings
    df2 = pd.read_excel('data/gold_triples/country_nationality_mapping.xlsx',
                        header=None, index_col=None, engine='openpyxl')
    country_mapping = dict(zip(df2[0], df2[1]))

    for entity, triples in triple_dict.items():
        for triple in triples:
            # ignore image, link or template triples
            if 'image' in triple[1] or 'username' in triple[1] or 'symbol' in triple[1] or 'WikicatWarsInvolving' in triple[-1] or triple in final_triples:
                continue
            if len(triple) == 3:
                # ignore incomplete triples
                if triple[-1] == '':
                    continue
                try:
                    # handle dates to bring them into the format dd/mm/yyyy
                    parse(triple[2], fuzzy=True)
                    obj = datetime.strptime(
                        triple[2], '%d %B %Y').strftime('%d/%m/%Y')
                except:
                    obj = triple[2]
                # map the relations from Wikidata to the relations of the automatically generated graphs
                for wikidata_rel, onto_rel in mapping.items():
                    if wikidata_rel == triple[1]:
                        # map nationalities
                        if onto_rel == 'HAS_NATIONALITY':
                            if obj in country_mapping:
                                obj = country_mapping[obj]
                        final_triples.append((triple[0], onto_rel, obj))
                    # handle cases where relations with `located in` contain more than that
                    elif wikidata_rel == 'located in' and wikidata_rel in triple[1]:
                        final_triples.append((triple[0], onto_rel, obj))

                if triple[-1] == 'World War I':
                    final_triples.append(
                        (triple[0], triple[1], 'First World War'))

                elif triple[-1] == 'World War II':
                    final_triples.append(
                        (triple[0], triple[1], 'Second World War'))
                else:
                    final_triples.append(tuple(triple)
                                         )
    with open('data/gold_triples/wikidata/postprocessed_triples.json', 'w', encoding='utf-8') as f:
        json.dump(final_triples, f)
    return final_triples


def get_wikidata_entity_id(title: str) -> str:
    '''Retrieves the Wikidata entity ID for a given Wikipedia page title.

    Args:
        title : The title of the Wikipedia page.

    Returns:
        The Wikidata entity ID if found, otherwise None.

    Example:
        entity_id = get_wikidata_entity_id('Douglas Adams')
    '''
    url = f'https://en.wikipedia.org/w/api.php'

    params = {
        'action': 'query',
        'titles': title,
        'prop': 'pageprops',
        'format': 'json'
    }
    # request the entity id for an article title for later extraction of the triples
    response = requests.get(url, params=params, headers={
                            'User-agent': 'KG Generation'})
    data = response.json()

    pages = data.get('query', {}).get('pages', {})
    for page_id, page_data in pages.items():

        entity_id = page_data.get('pageprops', {}).get('wikibase_item')
        if entity_id:
            return entity_id

    return None


def create_entity_mapping(enitites: List[str]):
    '''Creates a mapping of entities to their Wikidata entity IDs and writes it to a CSV file.

    Args:
        entities : A list of entity names to be mapped.

    Returns:
        None

    Example:
        create_entity_mapping(['Douglas Adams', 'Guillermo del Toro Gomez'])
    '''
    mapping = {}
    for entity in enitites:
        entity_id = get_wikidata_entity_id(entity)
        mapping[entity] = entity_id

    outfile = '../data/gold_triples/wikidata/entity_mapping2.csv'
    with open(outfile, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        for key, value in mapping.items():
            csvwriter.writerow([key, value])


def run_triple_extraction_wikidata(kg_triples: List):
    '''Extracts triples for the Wikipedia articles and appends them to a Wikidata gold triples JSON file.

    Args:
        kg_triples: A list of knowledge graph triples where each triple is a tuple containing subject, predicate, and object.

    Returns:
        None
    '''
    # get the article titles for the triple extraction
    filename = 'data/20240520_wikipedia_cleaned_sample.json'
    articles = _get_all_article_title(filename)
    subjects = articles
    # additionally get the triples for subjects found in the LLM-generated KG triples
    for triple in kg_triples:
        if triple[0] not in subjects:
            subjects.append(triple[0])
    # read the tmp_triples in case the extraction breaks off
    tmp_triples = read_batch_json(
        'data/gold_triples/wikidata/gold_triples.json')
    # extract triples for each article & sbj from the triples
    with open('data/gold_triples/wikidata/gold_triples.json', 'a', encoding='utf-8') as f:
        for i, subject in enumerate(subjects):
            if subject not in tmp_triples:
                print(
                    f'-----Processing entity {i+1} of {len(subjects)}: {subject}-----')
                triples = extract_triples_wikidata(subject)
                print(triples)
                print('----------------------------------------------------------------')
                json.dump({subject: triples}, f)
                time.sleep(3)
