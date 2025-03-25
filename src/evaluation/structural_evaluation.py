'''This file contains functions to compute the structural quality metrics from Seo et al. (2023)
'''
from typing import Dict, List, Tuple


def get_entities_onto(entities: Dict) -> Tuple[List, int]:
    '''Retrieves all entities from the hierarchical entity structure and counts the total number of entities.

    Args:
        entities: The nested dictionary structure containing entities.

    Returns:
        Tuple[List, int]: A tuple containing a list of all entities and the total count of entities.
    '''
    count = 0
    entity_list = []
    for key, value in entities.items():
        if isinstance(value, dict):
            count += 1
            entity_list.append(key)
            if 'childrenEntities' in value:
                count += get_entities_onto(value['childrenEntities'])[1]
                entity_list += get_entities_onto(value['childrenEntities'])[0]
            elif 'Other' in key:
                for k, v in value.items():
                    count += 1
                    entity_list.append(k)
    return entity_list, count


def get_properties_onto(entities) -> Tuple[List, int]:
    '''Retrieves all properties from the entities and counts the total number of properties.

    Args:
        entities: The nested dictionary structure containing entities and their properties.

    Returns:
        Tuple[List, int]: A tuple containing a list of all properties and the total count of properties.
    '''
    count = 0
    property_list = []
    for key, value in entities.items():
        if key == 'properties':
            count += len(value)
            property_list += [k for k, v in value.items()]
        elif isinstance(value, dict):
            count += get_properties_onto(value)[1]
            property_list += get_properties_onto(value)[0]
    return property_list, count


def calculate_instantiated_class_ratio_onto(ontology: Dict[str, Dict], langchain_onto: bool = False):
    '''Calculates the ratio of instantiated classes(here called entities) among all classes defined in the ontology, according to Seo et al. (2022)

    Args:
        ontology: A dictionary representing the ontology, with 'entities' and 'relationships' keys.
        langchain_onto (optional): Flag checking whether the ontology was created by langchain or not. Defaults to False

    Returns:
        the ratio of instantiated classes to the total number of classes.
    '''
    def _check_if_class_in_onto(entity: str, relations: Dict[str, List[Dict]] | List[Dict], langchain_onto: bool = False) -> bool:
        '''Checks if a given class (entity type) is present in the ontology relationships.

        Args:
            entity: The entity type (class) to check for in the relationships.
            relations: A dictionary of relationships in the ontology, where keys are relationship types
                        and values are lists of relationship details. OR List containing dicts of type {start: '', end:'', type:''}
            langchain_onto (optional): Flag checking whether the ontology was created by langchain or not. Defaults to False
        Returns:
            True if the entity type is found in the relationships, False otherwise.
        '''
        if langchain_onto:
            for rel in relations:
                if rel.get('start') == entity or rel.get('target') == entity:
                    return True
        else:
            for key, rel_values in relations.items():
                for d in rel_values:
                    if d.get('source') == entity or d.get('target') == entity:
                        return True
        return False

    if langchain_onto:
        # handle langchain ontology differently due to different structure
        entities = list(ontology['node_props'].keys())
        number_of_entities = len(entities)
        instantiated_entities = 0
        for entity in entities:
            # check if class of onto is in used in relations of onto
            if _check_if_class_in_onto(entity, ontology['relationships'], langchain_onto):
                instantiated_entities += 1
    else:
        entities, number_of_entities = get_entities_onto(ontology['entities'])
        instantiated_entities = 0
        for entity in entities:
            # check if class of onto is in used in relations of onto
            if _check_if_class_in_onto(entity, ontology['relationships']):
                instantiated_entities += 1
    return (instantiated_entities/number_of_entities)


def calculate_instantiated_class_ratio_kg(ontology, graph, langchain: bool = False):
    '''Calculates the ratio of instantiated classes/entities among all classes defined in the ontology, according to Seo et al. (2022)

    Args:
        ontology: A dictionary representing the ontology, with 'entities' and 'relationships' keys.
        graph: knowledge graph based on ontology
        langchain (optional): Flag checking whether the graph was created by langchain or not. Defaults to False

    Returns:
        the ratio of instantiated classes to the total number of classes.

    Example:
        ratio = calculate_instantiated_class_ratio_onto(ontology)
    '''
    def _check_if_class_in_kg(entity: str, graph, langchain: bool = False) -> bool:
        '''Checks if a given class (entity) exists in the knowledge graph.

        Args:
            entity: The class (entity) to check for in the knowledge graph.
            graph: The knowledge graph represented as a dictionary.
            langchain (optional): Flag checking whether the ontology was created by langchain or not. Defaults to False
        Returns:
            bool: True if the class exists in the knowledge graph, False otherwise.
        '''
        if langchain:
            for node in graph['nodes']:
                if entity in node['type']:
                    return True
        else:
            for tpl in graph['nodes']:
                if entity in tpl['labels']:
                    return True
        return False
    if langchain:
        entities = list(ontology['node_props'].keys())
        number_of_entities = len(entities)
        instantiated_entities = 0
        for entity in entities:
            # check if class of onto is initialized in KG
            if _check_if_class_in_kg(entity, graph, langchain):
                instantiated_entities += 1
    else:
        entities, number_of_entities = get_entities_onto(ontology['entities'])
        instantiated_entities = 0
        for entity in entities:
            # check if class of onto is initialized in KG
            if _check_if_class_in_kg(entity, graph):
                instantiated_entities += 1
    return (instantiated_entities/number_of_entities)


def calculate_instantiated_property_ratio(ontology: Dict[str, Dict], graph: Dict, langchain: bool = False) -> float:
    '''Calculates the ratio of properties actually used in the knowledge graph among the properties defined in the ontology.

    Args:
        ontology: The ontology containing the properties defined for entities.
        graph: The knowledge graph represented as a dictionary.
        langchain (optional): Flag checking whether the ontology was created by langchain or not. Defaults to False

    Returns:
        float: The ratio of instantiated properties to the total number of properties defined in the ontology.
    '''
    def _check_if_property_in_kg(property: str, graph: Dict) -> bool:
        '''Checks if a given property exists and has a value in the nodes of a knowledge graph.

        Args:
            property: The property to check for in the knowledge graph.
            graph: The knowledge graph represented as a dictionary.

        Returns:
            bool: True if the property exists and has a value in any node, False otherwise.

        '''
        for node in graph.get('nodes', []):
            properties = node.get('properties', {})
            if property in properties and properties[property]:
                return True
        return False
    if langchain:
        properties = [prop['property'] for node,
                      props in ontology['node_props'].items() for prop in props]
        number_of_properties = len(properties)
        instantiated_properties = 0
        for property in properties:
            # check if property of onto is initialized in KG
            if _check_if_property_in_kg(property, graph, langchain=True):
                instantiated_properties += 1
    else:
        properties, number_of_properties = get_properties_onto(
            ontology['entities'])
        instantiated_properties = 0
        for property in properties:
            # check if property of onto is initialized in KG
            if _check_if_property_in_kg(property, graph):
                instantiated_properties += 1
    return (instantiated_properties/number_of_properties)


def calculate_subclass_property_acquisition(ontology: Dict) -> float:
    '''Calculates the average number of properties acquired by subclasses that are not inherited.

    Args:
        ontology: The ontology containing the hierarchical structure of entities.

    Returns:
        float: The average number of properties acquired by subclasses.
    '''
    entities, number_of_entities = get_entities_onto(ontology['entities'])
    parent_child_pairs = _get_parent_child_pairs(ontology['entities'])
    subclass_properties = 0
    for parent, child in parent_child_pairs:
        parent_properties = parent[1]
        child_properties = child[1]
        # count how many new properties are in the child entity
        subset = set(child_properties) - set(parent_properties)
        subclass_properties += len(subset)
    return (subclass_properties/number_of_entities)


def calculate_subclass_property_acquisitionV2(ontology: Dict) -> float:
    '''Calculates the average number of properties acquired by subclasses that are not inherited.
    Version that excludes entities under the 'Other' category.

    Args:
        ontology: The ontology containing the hierarchical structure of entities.

    Returns:
        float: The average number of properties acquired by subclasses.
    '''
    parent_child_pairs = _get_parent_child_pairs(ontology['entities'])
    subclass_properties = 0
    for parent, child in parent_child_pairs:
        parent_properties = parent[1]
        child_properties = child[1]
        # count how many new properties are in the child entity
        subset = set(child_properties) - set(parent_properties)
        subclass_properties += len(subset)
    # exclude `Other` categories from ratio as they should not contain parent/child relations
    return (subclass_properties/len(parent_child_pairs))


def _get_parent_child_pairs(entities: Dict, parent: Dict = None) -> List[Tuple]:
    '''Generates a list of parent-child pairs from a nested dictionary structure of entities.

    Args:
        entities: The nested dictionary structure containing entities.
        parent (optional): The parent entity being processed. Defaults to None.

    Returns:
        List[Tuple]: A list of tuples, where each tuple contains a parent entity and its properties, and a child entity and its properties.
    '''
    parent_child_tuples = []
    for key, value in entities.items():
        if parent:
            parent_key = list(parent.keys())[0]
            parent_props = list(parent[parent_key].keys())
            child_props = list(value['properties'].keys()
                               ) if 'properties' in value else []
            parent_child_tuples.append(
                ((parent, parent_props), (key, child_props))
            )
        if isinstance(value, dict) and 'childrenEntities' in value:
            parent_child_tuples += _get_parent_child_pairs(
                value['childrenEntities'], {key: value['properties'] if 'properties' in value else {}})
    return parent_child_tuples


def calculate_inverse_multiple_inheritance(ontology: Dict) -> float:
    '''Measures how little multiple inheritance appears in the ontology.

    Args:
        ontology: The ontology containing the hierarchical structure of entities.

    Returns:
        float: The inverse measure of multiple inheritance in the ontology.
    '''
    def _get_nsups(entity: str, entities: Dict, current_path: List = []) -> List[str]:
        '''Finds the path to a given entity within the hierarchy of the ontology, representing the number of superclasses in the class.

        Args:
            entity: The entity to find within the nested dictionary.
            entities: The nested dictionary structure containing entities.
            current_path (optional): The current path being traversed. Defaults to an empty list.

        Returns:
            List[str]: The path to the entity, representing the superclasses in the class. Returns None if the entity is not found.'''
        for key, value in entities.items():
            if key == entity:
                return current_path
            elif isinstance(value, dict):
                if key != 'childrenEntities':
                    path = _get_nsups(entity, value, current_path + [key])
                    if path is not None:
                        return path
                else:
                    path = _get_nsups(entity, value, current_path)
                    if path is not None:
                        return path
        return None
    entities, number_of_entities = get_entities_onto(ontology['entities'])
    nsubs = 0
    for entity in entities:
        # get path until this entity to count the numbers of classes occuring before this entity
        e_nsups = _get_nsups(entity, ontology['entities'])
        nsubs += len(e_nsups)
    return (1/(nsubs/number_of_entities))
