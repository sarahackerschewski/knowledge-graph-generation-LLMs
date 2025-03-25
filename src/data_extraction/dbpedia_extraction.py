'''This file contains functions to query DBpedia to return the triples for the articles in the dataset, and to postprocess the extracted triples
'''
import json
import re
import time
from datetime import datetime
from typing import Dict, List, Tuple

import pandas as pd
from dateutil.parser import parse
from SPARQLWrapper import JSON, SPARQLWrapper
from wikidata.client import Client

from src.data_extraction.wikidata_extraction import _get_all_article_title
from src.utils.file_utils import read_batch_json

sparql = SPARQLWrapper("http://dbpedia.org/sparql")
client = Client()


def extract_triples_dbpedia(entity: str) -> List[Tuple[str, str, str]]:
    '''Extracts RDF triples for a given entity from a SPARQL endpoint from DBpedia.
    Args:
        entity (str): The label of the entity to query for.

    Returns:
        List[Tuple[str, str, str]]: A list of triples (subject, predicate, object) where
        the subject has a label matching the provided entity.
    '''
    # query extracting English triples for the entity in DBpedia
    query = f"""
    SELECT ?subject ?predicate ?object
    WHERE {{
        ?subject ?predicate ?object .
        ?subject rdfs:label "{entity}"@en .
        FILTER (lang(?object) = 'en' || !isLiteral(?object))
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    try:
        results = sparql.query().convert()
    except:
        return []

    triples = []
    # get subject, predicate, object from the query result
    for i, result in enumerate(results["results"]["bindings"]):
        subject = result["subject"]["value"].split('/')[-1].replace('_', ' ')
        predicate = result["predicate"]["value"].split(
            '/')[-1].replace('_', ' ')
        object_ = result["object"]["value"]
        if not object_.startswith("http://"):
            object_ = object_.split('/')[-1].replace('_', ' ')
        tmp_triple = (subject, predicate, object_)
        triple = []
        seen_triples = set()
        # handle different link cases to get a triple only consisting of sbj, pred, & obj labels not links
        for i, link in enumerate(tmp_triple):
            # ignore templates and image triples
            if tmp_triple[1] == 'wikiPageExternalLink' or tmp_triple[1] == 'wikiPageUsesTemplate' or tmp_triple[1] == 'thumbnail':
                break
            if 'http://' in link:
                try:
                    link = _get_label(link)
                except:
                    if 'espn' in link:
                        break
            if '#' in link:
                triple.append(link.split('#')[-1])
            else:
                triple.append(link)
            time.sleep(3)
        if len(triple) == 3:
            triple_tuple = tuple(triple)
            # only add unique triples
            if triple_tuple not in seen_triples:
                triples.append(triple_tuple)
                seen_triples.add(triple_tuple)
    return triples


def _get_label(uri: str) -> str:
    '''Retrieves the English label for a given URI from a SPARQL endpoint.

    Args:
        uri (str): The URI of the entity to query for.

    Returns:
        str: The English label of the entity, or the last part of the URI if no label is found
    '''
    sparql = SPARQLWrapper("http://dbpedia.org/sparql")

    # query to obtain the label from a link
    query = f"""
    PREFIX rdfs: <http://www.w3.org/2000/01/rdf-schema#>

    SELECT ?label WHERE {{
      <{uri}> rdfs:label ?label .
      FILTER (lang(?label) = 'en')
    }}
    """
    sparql.setQuery(query)
    sparql.setReturnFormat(JSON)
    results = sparql.query().convert()

    if results["results"]["bindings"]:
        return results["results"]["bindings"][0]["label"]["value"]
    else:
        # handle wikidata links with the wikidata client
        if 'wikidata' in uri:
            entity = client.get(uri.split('/')[-1], load=True)
            return str(entity.label)
        return uri.split('/')[-1]


def postprocessing_triples_dbpedia(triple_file: str) -> List[Tuple[str, str, str]]:
    '''Postprocesses triples extracted from DBpedia, maps relations, nationalities, & dates, and saves the final triples to a JSON file.

    Args:
        triple_file (str): The path to the file containing extracted triples.

    Returns:
        List[Tuple[str, str, str]]: A list of processed and mapped triples (subject, predicate, object).
    '''
    triple_dict = read_batch_json(triple_file)
    final_triples = []
    # open the file containing the relation mappings
    df = pd.read_excel('data/gold_triples/dbpedia/relation_mapping.xlsx',
                       header=None, index_col=None, engine='openpyxl')
    mapping = dict(zip(df[0], df[1]))
    # open the file containing the nationality mappings
    df2 = pd.read_excel('data/gold_triples/country_nationality_mapping.xlsx',
                        header=None, index_col=None, engine='openpyxl')
    country_mapping = dict(zip(df2[0], df2[1]))

    for entity, triples in triple_dict.items():
        for triple in triples:
            # ignoring image triples
            if 'depiction' in triple[1] or 'username' in triple[1] or 'logo' in triple[1] or 'caption' in triple[1] or 'image' in triple[1] or triple in final_triples:
                continue
            if len(triple) == 3:
                # ignore incomplete triples
                if triple[-1] == '':
                    continue
                triple[-1] = triple[-1].replace('_', ' ')
                try:
                    # handle dates to bring them into the format dd/mm/yyyy
                    parse(triple[2], fuzzy=True)
                    obj = datetime.strptime(
                        triple[2], '%d %B %Y').strftime('%d/%m/%Y')
                except:
                    obj = triple[2]
                # map the relations from DBpedia to the relations of the automatically generated graphs
                for dbpedia_rel, onto_rel in mapping.items():
                    if dbpedia_rel == triple[1]:
                        # map nationalities
                        if onto_rel == 'HAS_NATIONALITY':
                            if obj in country_mapping:
                                obj = country_mapping[obj]
                        final_triples.append((triple[0], onto_rel, obj))
                    elif dbpedia_rel == 'located in' and dbpedia_rel in triple[1]:
                        final_triples.append((triple[0], onto_rel, obj))
                # handle specific cases for DBpedia triples, to match, e.g., birth/death dates, and other relations
                if 'birth' in triple[-1] and len(triple[-1].split(' ')) <= 4:
                    birthdate = triple[-1].replace('births', '').strip()
                    final_triples.append((triple[0], 'BORN_IN', birthdate))
                    final_triples.append((triple[0], 'BORN_ON', birthdate))
                elif 'death' in triple[-1] and len(triple[-1].split(' ')) <= 4:
                    deathdate = triple[-1].replace('deaths', '').strip()
                    final_triples.append((triple[0], 'DIED_IN', deathdate))
                    final_triples.append((triple[0], 'DIED_ON', deathdate))
                elif 'completed' in triple[-1] and len(triple[-1].split(' ')) <= 4:
                    builtdate = re.sub(
                        r"Houses completed in (\d{4})", r"\1", triple[-1])
                    final_triples.append((triple[0], 'BUILT_IN', builtdate))
                elif 'establishments' in triple[-1] and len(triple[-1].split(' ')) <= 4:
                    est_date = re.sub(
                        r"(\d{4}) establishments in.*", r"\1", triple[-1])
                    final_triples.append(
                        (triple[0], 'ESTABLISHED_IN', est_date))
                elif 'set in' in triple[-1] and len(triple[-1].split(' ')) <= 4:
                    set_location = re.sub(r"^.*set in (.*)", r"\1", triple[-1])
                    final_triples.append((triple[0], 'SET_IN',
                                          set_location))
                elif 'described in' in triple[-1] and len(triple[-1].split(' ')) < 4:
                    final_triples.append(
                        (triple[0], 'DESCRIBED_IN', triple[-1]))
                elif re.search(r"\b\d{4}\b", triple[-1]):
                    date = re.sub(r".*\b(\d{4})\b.*", r"\1", triple[-1])
                    final_triples.append(
                        (triple[0], 'OCCURRED_IN', date))
                elif triple[-1].startswith('Endemic') and 'of' in triple[-1]:
                    final_triples.append(
                        (triple[0], 'ENDEMIC_TO', triple[-1].split('of', 1)[1].strip()))
                elif 'World War I' == triple[-1]:
                    final_triples.append(
                        (triple[0], triple[1], 'First World War'))
                elif 'World War II' == triple[-1]:
                    final_triples.append(
                        (triple[0], triple[1], 'Second World War'))
                else:
                    final_triples.append(tuple(triple))
    with open('data/gold_triples/dbpedia/postprocessed_triples.json', 'w', encoding='utf-8') as f:
        json.dump(final_triples, f)
    return final_triples


def run_triple_extraction_dbpedia(kg_triples: List):
    '''Extracts triples for the Wikipedia articles and appends them to a DBpedia gold triples JSON file.

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
        'data/gold_triples/dbpedia/gold_triples.json')
    # extract triples for each article & sbj from the triples
    with open('data/gold_triples/dbpedia/gold_triples.json', 'a', encoding='utf-8') as f:
        for i, subject in enumerate(subjects):
            if subject not in tmp_triples:
                print(
                    f'-----Processing entity {i+1} of {len(subjects)}: {subject}-----')
                triples = extract_triples_dbpedia(subject)
                print(triples)
                print('----------------------------------------------------------------')
                json.dump({subject: triples}, f)
                time.sleep(3)
