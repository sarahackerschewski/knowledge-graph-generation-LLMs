'''This file contains functions measuringvarious statistics for the dataset, ontology and the KG
'''
from typing import Dict, List, Tuple

from src.evaluation.structural_evaluation import get_entities_onto


def get_article_lengths(articles: List[Dict]) -> Dict[str, float]:
    '''Calculates the average lengths of original and cleaned articles

    Args:
        articles: A list of dictionaries, where each dictionary represents an article with 'text' and 'cleaned_text' keys.

    Returns:
        Dict[str, float]: A dictionary containing the average lengths of original and cleaned articles.
    '''
    lengths_all_articles = []
    lengths_cleaned_articles = []
    for article in articles:
        length = len(article['text'].split(' '))
        length_clean = len(article['cleaned_text'].split(' '))
        lengths_all_articles.append(length)
        lengths_cleaned_articles.append(length_clean)
    print(
        f'Original Articles: Average {sum(lengths_all_articles)/len(articles)}')
    print(
        f'Cleaned Articles: Average {sum(lengths_cleaned_articles)/len(articles)}\n')
    return {'original': sum(lengths_all_articles)/len(articles), 'cleaned': sum(lengths_cleaned_articles)/len(articles)}


def compare_relations_between_graphs(triples1: List[Tuple[str, str, str]], triples2: List[Tuple[str, str, str]]) -> int:
    '''
    Compares two graphs and counts the number of overlapping triples .

    Args:
        triples1: The triples of the first graph
        triples2: The triples of the second graph

    Returns:
        int: The number of overlapping relations between the two sets of triples.
    '''
    overlapping_rels = 0
    for triple in triples1:
        if triple in triples2:
            overlapping_rels += 1
    return overlapping_rels


def count_entities_relations_onto(onto: Dict) -> Tuple[int, int]:
    '''Counts the number of entities and relationships in the ontology.

    Args:
        onto: The ontology containing entities and relationships.

    Returns:
        Tuple[int, int]: A tuple containing the count of entities and the count of relationships.
    '''
    _, count_e = get_entities_onto(onto['entities'])
    count_r = len(onto['relationships'])
    return count_e, count_r


def count_labels(ontology: Dict, graph: Dict, langchain: bool) -> Tuple[Dict[str, int], List[str], List[str]]:
    '''Count the labels in a knowledge graph and identify which labels are present in the ontology.

    Args:
        ontology: The ontology containing entities and their properties.
        graph: The knowledge graph represented as a dictionary.
        langchain: Flag indicating whether to use LangChain-specific processing.

    Returns:
        Tuple[Dict[str, int], List[str], List[str]]: A tuple containing a dict with label counts,labels present in the ontologynew labels not present in the ontology & new labels not present in the ontology.
    '''
    if langchain:
        entities = [key for key, value in ontology['node_props'].items()]
    else:
        entities, amount = get_entities_onto(ontology['entities'])
    label_count = {}
    labels_in_onto = []
    new_labels = []
    for node in graph['nodes']:
        # handle langchain differently due to different structure
        if langchain:
            label = node['type']
            if label in entities:
                if label not in labels_in_onto:
                    # label was used from ontology
                    labels_in_onto.append(label)
                else:
                    # label was newly added during KG generation
                    if label not in new_labels:
                        new_labels.append(label)
            if label in label_count:
                label_count[label] += 1
            else:
                label_count[label] = 1
        else:
            labels = node['labels']
            for label in labels:
                if label in entities:
                    if label not in labels_in_onto:
                        # label was used from ontology
                        labels_in_onto.append(label)
                else:
                    if label not in new_labels:
                        # label was newly added during KG generation
                        new_labels.append(label)
                if label in label_count:
                    label_count[label] += 1
                else:
                    label_count[label] = 1
    return label_count, labels_in_onto, new_labels


def count_relationships(ontology: Dict, graph: Dict, langchain: bool) -> Tuple[Dict[str, int], List[str], List[str]]:
    '''Counts the occurrences of labels in the knowledge graph.

    Args:
        ontology: The ontology containing entities and their properties.
        graph: The knowledge graph represented as a dictionary.
        langchain: Flag checking whether the ontology was created by langchain or not.

    Returns:
        Tuple[Dict[str, int], List[str], List[str]]: A tuple containing a dictionary of label counts, labels present in the ontology & new labels not present in the ontology.
    '''
    if langchain:
        # handle langchain differently due to different structure
        relationships = [key for key, value in ontology['rel_props'].items()]
    else:
        relationships = [key for key,
                         value in ontology['relationships'].items()]
    rel_count = {}
    rels_in_onto = []
    new_rels = []
    for relation in graph['relationships']:
        rel_type = relation['type']
        if rel_type in relationships:
            if rel_type not in rels_in_onto:
                # relation was used from ontology
                rels_in_onto.append(rel_type)
        else:
            if rel_type not in new_rels:
                # relation was newly added during KG generation
                new_rels.append(rel_type)
        if rel_type in rel_count:
            rel_count[rel_type] += 1
        else:
            rel_count[rel_type] = 1
    return rel_count, rels_in_onto, new_rels


def calculate_children_counts(ontology: Dict) -> Dict:
    '''Computes the number of immediate child entities for each entity in the given ontology.

    Args:
        ontology: A dictionary representing the ontology with a hierarchical structure.

    Returns:
        Dict: A dictionary where keys are entity names, and values are the count of their immediate children.
    '''
    def count_children_entities(entity: Dict, children_counts: Dict):
        '''Counts and updates the number of direct children for each entity.

        Args:
            entity: The entity whose children are being counted.
            children_counts: A dictionary storing the number of children per entity. 

        Returns:
            None: The function updates `children_counts` instead of returning a value.
        '''
        for name, sub_entity in entity.get('childrenEntities', {}).items():
            children_counts[name] = len(sub_entity.get('childrenEntities', {}))
            count_children_entities(sub_entity, children_counts)

    children_counts = {}
    for entity_name, entity in ontology['entities'].items():
        children_counts[entity_name] = len(entity.get('childrenEntities', {}))
        # for each entity count the children of their children entities
        count_children_entities(entity, children_counts)

        if entity_name == 'Other':  # Special handling for 'Other' entity
            if 'childrenEntities' not in entity:
                children_counts[entity_name] = len(entity)
            # in case the ontologies wrongly added childernEntities to entities of the `Other` class, count these too
            for other_name, other_entity in entity.items():
                if other_name not in ['properties', 'childrenEntities']:
                    children_counts[other_name] = len(
                        other_entity.get('childrenEntities', {}))
                    count_children_entities(other_entity, children_counts)
    return children_counts
