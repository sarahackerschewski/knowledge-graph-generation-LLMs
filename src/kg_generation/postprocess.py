'''This file contains functions to postprocess the LLM-generated ontologies and graphs with duplicate removal, cleaning and merging
'''
import json
from pathlib import Path
from typing import Dict, List

from openai import OpenAI
from sklearn.metrics.pairwise import cosine_similarity

from src.utils.file_utils import read_batch_json, read_json

EMBEDDING_PATH = Path('data/embeddings/description_embedding_')


def clean_ontology(tmp_ontology_file: str, output_file: str, openai_client: OpenAI):
    '''Cleans and processes ontology data from the batch ontology file.

    This function combines all batches with little duplicates as possible, processes entities and relationships,
    and saves the cleaned ontology data to an output file. It also handles description embeddings for relationships.

    Args:
        tmp_ontology_file: The path to the temporary file containing the ontology data to be cleaned.
        output_file: The path to the file where the cleaned ontology data will be saved.

    Returns:
        None
    '''
    entities = []
    relations = {}
    descr_embedding_mapping = {}
    sub_rel_counter = 0
    # if description embeddings already exist
    model = tmp_ontology_file.split('_')[-1]
    embedding_file = EMBEDDING_PATH + model
    if embedding_file.exists():
        with open('description_embeddings_gpt-4-turbo_onto.json', 'r') as embed_f:
            descr_embedding_mapping = json.load(embed_f)

    tmp_ontology = read_batch_json(tmp_ontology_file)
    for onto in tmp_ontology:
        for batch_nr, batch_ontology in onto.items():
            print(f'''Cleaning Batch {
                int(batch_nr)+1} of {len(tmp_ontology)}''')
            for entity in batch_ontology['entities']:
                entity = _replace_nonchar(entity)
                if entity not in entities:
                    entities.append(entity)
            for relation in batch_ontology['relationships']:
                name = relation['type']
                description = relation['description'].lower()
                # add to a description embedding dictionary if the embeddings do not already exist
                if description not in descr_embedding_mapping.keys():
                    descr_embedding_mapping[description] = get_embedding(
                        description, openai_client)
                source = _replace_nonchar(relation['source'])
                target = _replace_nonchar(relation['target'])
                if name not in relations.keys():
                    relations[name] = [
                        {'description': description, 'source': source, 'target': target}]
                    sub_rel_counter += 1
                else:
                    all_descriptions = [d['description']
                                        for d in relations[name]]
                    if description not in all_descriptions:
                        if not is_duplicate(descr_embedding_mapping, all_descriptions, description):
                            relations[name].append(
                                {'description': description, 'source': source, 'target': target})
                            sub_rel_counter += 1
    print(f'{len(entities)} entities in ontology')
    print(f'{len(relations)} relations in ontology (ignoring descriptions)')
    print(f'{sub_rel_counter} relations in ontology (with descriptions)')

    with open(output_file, 'w') as out_f:
        json.dump({'entities': entities, 'relationships': relations}, out_f)

    if not embedding_file.exists():
        with open(embedding_file, 'w') as embed_f:
            json.dump(descr_embedding_mapping, embed_f)


def _replace_nonchar(s: str) -> str:
    '''Replaces hyphens, underscores, and newline characters in the input string with spaces, and capitalizes the first letter of each word..

    Args:
        s: The input string to be processed.

    Returns:
        str: The processed string with non-character symbols replaced and words capitalized.

    Example:
        _replace_nonchar('hello-world_example\n')
        # Returns 'HelloWorldExample'
    '''
    # Replace hyphens and underscores with spaces
    s = s.replace('-', ' ').replace('_', ' ').replace('\n', '')

    # Split the string into words
    if ' ' in s:
        words = s.split()
        # Capitalize the first letter of each word and join them
        s = ''.join([word.capitalize() if word[0].islower()
                    else word for word in words])
    return s


def get_embedding(text: str, client: OpenAI) -> List[float]:
    '''Retrieves the OpenAI embedding for a given text using a specified model.

    Args:
        text: The input text for which the embedding is to be generated.
        client: The OpenAI client

    Returns:
        List[float]: The embedding of the input text.
    '''
    return client.embeddings.create(input=text, model='text-embedding-3-small').data[0].embedding


def is_duplicate(embedding_mapping: Dict, descriptions: List, new_description: str) -> bool:
    '''Checks if a new description is a duplicate based on the cosine similarity of the embeddings.

    Args:
        embedding_mapping: A dictionary mapping descriptions to their embeddings.
        descriptions: A list of existing descriptions.
        new_description: The new description to be checked for duplication.

    Returns:
        bool: True if the new description is a duplicate, False otherwise.
    '''
    for descr in descriptions:
        similarity = cosine_similarity([embedding_mapping[descr]], [
                                       embedding_mapping[new_description]])[0][0]
        if similarity > 0.55:
            return True
    return False


def is_entity_in_hierarchy(hierarchy: Dict, entity: str) -> bool:
    '''Check if an entity exists within a hierarchical dictionary.

    Args:
        hierarchy: The hierarchical dictionary to search within.
        entity: The entity to search for.

    Returns:
        bool: True if the entity is found in the hierarchy, False otherwise.
    '''
    if entity in hierarchy:
        return True
    for k, v in hierarchy.items():
        if isinstance(v, dict):
            if is_entity_in_hierarchy(v, entity):
                return True
    return False


def merge_property_batches(property_file: str, model: str):
    '''Merge property batches from a JSON file and save the merged properties to a new file.

    Args:
        property_file: The path to the JSON file containing property batches.
        model: The model name to be used in the output file name.

    Returns:
        None
    '''
    property_batches = read_batch_json(property_file)
    new_property_dict = {}
    for batch, content in property_batches.items():
        new_property_dict.update(content)

    with open(f'data/ontologies/merged_batches/hierarchy_property_{model}.json', 'w', encoding='utf-8') as f:
        json.dump(new_property_dict, f)


def merge_ontology_with_entities(ontology_file: str, hierarchy_file: str, outfile: str):
    '''Merge the final ontology with the new hierarchical entities

    Args:
        ontology_file: File containing the 'old' ontology, i.e. entities w/o hierarchy
        hierarchy_file: File containing the entities wihth properties and structured in a hierarchy
        outfile: The file, where the final ontology should be saved in

    Returns:
        None
    '''
    ontology = read_batch_json(ontology_file)
    hierarchy = read_batch_json(hierarchy_file)

    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'entities': hierarchy,
                  'relationships': ontology['relationships']}, f)


def clean_graph(batch_file: str, outfile: str):
    '''Clean and merge graph batches from a JSON file, updating node IDs and relationships.

    Args:
        batch_file: The path to the JSON file containing graph batches.
        outfile: The path to the output JSON file where the cleaned graph will be saved.

    Returns:
        None
    '''
    kg_batches = read_batch_json(batch_file)
    for i, batch in kg_batches.items():
        print(f'Merging batch {int(i)+1} of {len(kg_batches)}')
        if i == "0":
            new_node_list = batch['nodes']
            new_rel_list = batch['relationships']
        else:
            max_id = max([int(node['id'])
                         for node in new_node_list if node['id'].isdigit()]) + 1
            id_mapping = {}
            # replace batch id with graph id
            for node in batch['nodes']:
                old_id = node['id']
                if old_id.isdigit():
                    new_id = max_id
                    id_mapping[int(old_id)] = new_id
                    node['id'] = str(new_id)
                    max_id += 1
                new_node_list.append(node)
            # replace the ids in the relationships with the new graph ids
            for relation in batch['relationships']:
                start_node_id = relation['startNode']
                end_node_id = relation['endNode']
                relation['startNode'] = str(id_mapping.get(
                    int(start_node_id), start_node_id)) if start_node_id.isdigit() else start_node_id
                relation['endNode'] = str(id_mapping.get(
                    int(end_node_id), end_node_id)) if end_node_id.isdigit() else end_node_id
                new_rel_list.append(relation)

    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'nodes': new_node_list, 'relationships': new_rel_list}, f)


def clean_graph_langchain(batch_file: str, outfile: str):
    '''Clean and merge graph batches from a JSON file, ensuring no duplicate nodes or relationships.

    Args:
        batch_file: The path to the JSON file containing graph batches.
        outfile: The path to the output JSON file where the cleaned graph will be saved.

    Returns:
        None
    '''
    seen_batches = []
    kg_batches = read_batch_json(batch_file)
    nodes = []
    relations = []
    for i, batch in kg_batches.items():
        print(f'Merging batch {int(i)+1} of {len(kg_batches)}')
        # sometimes batches twice in there if process had to be restarted and accidental wrong start batch id was given
        if str(i) in seen_batches:
            continue
        seen_batches.append(str(i))
        for node in batch['nodes']:
            all_node_ids = [n['id'] for n in nodes]
            if node['id'] not in all_node_ids:
                nodes.append(node)
        for rel in batch['relationships']:
            all_rels = [(r['source']['id'], r['type'], r['target']['id'])
                        for r in relations]
            if (rel['source']['id'], rel['type'], rel['target']['id']) not in all_rels:
                relations.append(rel)
    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump({'nodes': nodes, 'relationships': relations}, f)


def remove_duplicates_in_kg(kg_file: str, outfile: str):
    '''Remove duplicate nodes in a knowledge graph and update relationships accordingly.

    Args:
        kg_file: The path to the JSON file containing the knowledge graph.
        outfile: The path to the output JSON file where the cleaned knowledge graph will be saved.

    Returns:
        None
    '''
    kg = read_json(kg_file)
    name_id_mapping = {}
    for node in kg['nodes']:
        name = node['properties'].get('name')
        if name:
            if name not in name_id_mapping:
                name_id_mapping[name] = []
            name_id_mapping[name].append(node['id'])

    merged_nodes = {}
    for name, ids in name_id_mapping.items():
        if len(ids) > 1:
            # replace all similar nodes with the lowest id of all similar nodes
            merged_node = {
                'id': min(ids, key=int),  # Use the lowest id
                'labels': set(),
                'properties': {}
            }

            for node_id in ids:
                node = next(n for n in kg['nodes'] if n['id'] == node_id)
                merged_node['labels'].update(node['labels'])

                for prop, value in node['properties'].items():
                    if prop not in merged_node['properties']:
                        merged_node['properties'][prop] = value
                    elif merged_node['properties'][prop] != value:
                        merged_node['properties'][prop] = ''

            merged_node['labels'] = list(merged_node['labels'])
            merged_nodes[merged_node['id']] = merged_node

    # change ids in the relationships with the new representaative node for the duplicates
    for relationship in kg['relationships']:
        for name, ids in name_id_mapping.items():
            if len(ids) > 1:
                new_id = min(ids, key=int)
                if relationship['startNode'] in ids:
                    relationship['startNode'] = new_id
                if relationship['endNode'] in ids:
                    relationship['endNode'] = new_id

     # Remove all nodes that belong to duplicate groups.
    duplicate_ids = {node_id for ids in name_id_mapping.values() if len(
        ids) > 1 for node_id in ids}
    kg['nodes'] = [node for node in kg['nodes']
                   if node['id'] not in duplicate_ids]

    # Add back in the merged nodes (each appears only once)
    kg['nodes'].extend(merged_nodes.values())

    with open(outfile, 'w', encoding='utf-8') as f:
        json.dump(kg, f)
