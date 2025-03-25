'''This file contains functions to calculate the exact and partial accuracies between gold_triples and predicted triples
'''
from typing import Dict, List, Tuple


def calculate_exact_accuracy(gold_triples: List[Tuple[str, str, str]], predicted_triples: List[Tuple[str, str, str]]) -> Tuple[float, List[bool], Dict]:
    '''Calculates the exact accuracy of predicted triples against gold standard triples.

    Args:
        gold_triples: The list of gold standard triples.
        predicted_triples: The list of predicted triples.

    Returns:
        Tuple[float, List[bool], Dict]: A tuple containing the exact accuracy score, a list of boolean values indicating correctness for each predicted triple, and a dictionary mapping each predicted triple to its matching gold standard triples.
    '''
    correct_triples = 0
    results = []
    res_dict = {}

    for triple in predicted_triples:
        tmp_res = False
        found_match = False

        for trpl in gold_triples:
            if compare_entities(triple[0], trpl[0]) and compare_entities(triple[1], trpl[1]) and compare_entities(triple[2], trpl[2]):
                tmp_res = True
                if triple not in res_dict:
                    res_dict[triple] = []
                res_dict[triple].append(trpl)
                if not found_match:
                    correct_triples += 1
                    found_match = True

        results.append(tmp_res)

    return correct_triples/len(predicted_triples), results, res_dict


def calculate_partial_accuracy(gold_triples: List[Tuple[str, str, str]], predicted_triples: List[Tuple[str, str, str]]) -> Tuple[float, List[bool], Dict]:
    '''Calculates the partial accuracy of predicted triples against gold standard triples, 
    considering a triple correct if either the subject and predicate or the subject and object match.

    Args:
        gold_triples: The list of gold standard triples.
        predicted_triples: The list of predicted triples.

    Returns:
        Tuple[float, List[bool], Dict]: A tuple containing the partial accuracy score, a list of boolean values indicating correctness for each predicted triple, and a dictionary mapping each predicted triple to its matching gold standard triples.'
    '''
    correct = 0
    results = []
    correct_result = {}

    for triple in predicted_triples:
        tmp_correct = False
        found_match = False

        for trpl in gold_triples:
            # partially correct if subject and object correct
            if compare_entities(triple[0], trpl[0]) and compare_entities(trpl[1], triple[1]):
                tmp_correct = True
                if triple not in correct_result:
                    correct_result[triple] = []
                correct_result[triple].append(trpl)
                if not found_match:
                    correct += 1
                    found_match = True
            # partially correct if subject and predicate correct
            elif compare_entities(triple[0], trpl[0]) and compare_entities(triple[2], trpl[2]):
                tmp_correct = True
                if triple not in correct_result:
                    correct_result[triple] = []
                correct_result[triple].append(trpl)
                if not found_match:
                    correct += 1
                    found_match = True

        results.append(tmp_correct)

    return correct / len(predicted_triples), results, correct_result


def compare_entities(entity1: str, entity2: str) -> bool:
    ''' Compares two entities to determine if they are a match based on direct match, substring match, or significant word overlap.

    Args:
        entity1: The first entity to compare.
        entity2: The second entity to compare.

    Returns:
        bool: True if the entities match based on the criteria, False otherwise.
    '''
    entity1 = entity1.lower()
    entity2 = entity2.lower()
    if entity1 == '' or entity2 == '':
        return False
    # # Check for direct match or substring match
    if entity1 == entity2 or entity1 in entity2 or entity2 in entity1:
        return True
    # Split the entities into words and check for significant overlap
    words1 = set(entity1.split())
    words2 = set(entity2.split())

    # Calculate the intersection of the two sets
    common_words = words1 & words2

    # Define a threshold for significant overlap
    threshold = (min(len(words1), len(words2)) / 2) + 1

    if len(common_words) >= threshold:
        return True
    return False
