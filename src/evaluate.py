'''This file contains functions to run various evaluation metrics
'''
import csv
from itertools import combinations
from pathlib import Path
from typing import List, Tuple

from src.data_extraction.dbpedia_extraction import postprocessing_triples_dbpedia
from src.data_extraction.wikidata_extraction import postprocessing_triples_wikidata
from src.evaluation.content_evaluation import (
    calculate_exact_accuracy,
    calculate_partial_accuracy,
)
from src.evaluation.statistics import (
    calculate_children_counts,
    compare_relations_between_graphs,
    count_entities_relations_onto,
    count_labels,
    count_relationships,
    get_article_lengths,
)
from src.evaluation.structural_evaluation import (
    calculate_instantiated_class_ratio_kg,
    calculate_instantiated_class_ratio_onto,
    calculate_instantiated_property_ratio,
    calculate_inverse_multiple_inheritance,
    calculate_subclass_property_acquisition,
    calculate_subclass_property_acquisitionV2,
)
from src.kg_generation.generate_kg import generate_triples
from src.utils.file_utils import read_json

WIKIDATA_POSTPROCESS_PATH = Path(
    'data/gold_triples/wikidata/postprocessed_triples.json')
DBPEDIA_POSTPROCESS_PATH = Path(
    'data/gold_triples/dbpedia/postprocessed_triples.json')


def run_structural_evaluation(ontology_file: str, kg_file: str):
    '''Runs various evaluation metrics for ontology and KG quality

    Args:
        ontology_file : The path to where the ontology is saved
        kg_file: The path to where the graph is saved
    Returns:
        None
    '''
    # read ontology_json
    ontology = read_json(ontology_file)

    graph = read_json(kg_file)

    # Instantiated Class Ratio
    icr_onto = calculate_instantiated_class_ratio_onto(ontology)
    print('*ONTOLOGY ONLY INSTANTIATED CLASS RATIO*: ', icr_onto)
    icr = calculate_instantiated_class_ratio_kg(ontology, graph)
    print('*KG INSTANTIATED CLASS RATIO*: ', icr)

    # Instantiated Property Ratio
    ipr = calculate_instantiated_property_ratio(ontology, graph)
    print('*INSTANTIATED PROPERTY RATIO*: ', ipr)

    # Subclass Property Acquisition
    spa = calculate_subclass_property_acquisition(ontology)
    print('*SUBCLASS PROPERTY ACQUISITION*: ', spa)

    spa2 = calculate_subclass_property_acquisitionV2(ontology)
    print('*SUBCLASS PROPERTY ACQUISITION V2*: ', spa2)

    # Inverse Multiple Inheritance
    imi = calculate_inverse_multiple_inheritance(ontology)
    print('*INVERSE MULIPLE INHERITANCE*', imi)

    outfile = f'data/evaluations/structural_quality/structural_metrics_{kg_file.split('/')[-1].split('.')[0]}.csv'
    with open(outfile, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        results = {'ICR Onto': icr_onto, 'ICR KG': icr,
                   'IPR': ipr, 'SPA': spa, 'SPA V2': spa2, 'IMI': imi}
        for metric, value in results.items():
            csvwriter.writerow([metric, value])


def run_content_evaluation(gold_triple_file: str, predicted_triple_file: str):
    '''Evaluate the exact and partial accuracy of predicted triples against gold standard triples.

    Args:
        gold_triple_file: The path to the file containing gold standard triples.
        predicted_triple_file: The path to the file containing predicted triples.

    Returns:
        None
    '''
    if 'wikidata' in gold_triple_file:
        if WIKIDATA_POSTPROCESS_PATH.exists():
            gold_triples = read_json(WIKIDATA_POSTPROCESS_PATH)
        else:
            gold_triples = postprocessing_triples_wikidata(gold_triple_file)
    else:
        if DBPEDIA_POSTPROCESS_PATH.exists():
            gold_triples = read_json(DBPEDIA_POSTPROCESS_PATH)
        else:
            gold_triples = postprocessing_triples_dbpedia(gold_triple_file)
    if 'langchain' in predicted_triple_file:
        predicted_triples = generate_triples(
            predicted_triple_file, langchain=True)
    else:
        predicted_triples = generate_triples(predicted_triple_file)

    # Exact accuracy
    exact_acc, results_exact, res_dict_exact = calculate_exact_accuracy(
        gold_triples, predicted_triples)
    print('*EXACT ACCURACY*', exact_acc)

    # Partial accuracy
    partial_acc, results_partial, res_dict_partial = calculate_partial_accuracy(
        gold_triples, predicted_triples)
    print('*PARTIAL ACCURACY*', partial_acc)

    # write to csv file
    kg = predicted_triple_file.split('/')[-1].split('.')[0]
    outfile = f'''data/evaluations/knowledge_quality/wikidata_{
        kg}.csv'''if 'wikidata' in gold_triple_file else f'''data/evaluations/knowledge_quality/dbpedia_{kg}.csv'''
    with open(outfile, 'w', encoding='utf-8', newline='') as f:
        csvwriter = csv.writer(f, delimiter=';')
        csvwriter.writerow(['triple', 'exact_accuracy', 'gold_triple_exact',
                           'partial_accuracyV2', 'gold_triple_partialV2'])
        for i, triple in enumerate(predicted_triples):
            gold_trpl_pa = ''
            if triple in list(res_dict_partial.keys()):
                gold_trpl_pa = res_dict_partial[triple]
            gold_trpl_ex = ''
            if triple in list(res_dict_exact.keys()):
                gold_trpl_ex = res_dict_exact[triple]
            csvwriter.writerow(
                [triple, results_exact[i], gold_trpl_ex, results_partial[i], gold_trpl_pa])


def run_structural_evaluation_langchain_graph(onto_file: str, kg_file: str):
    '''Evaluate the structural quality of a LangChain knowledge graph using ontology metrics.

    Args:
        onto_file: The path to the JSON file containing the ontology.
        kg_file: The path to the JSON file containing the knowledge graph.

    Returns:
        None
    '''
    ontology = read_json(onto_file)
    kg = read_json(kg_file)

    icr_onto = calculate_instantiated_class_ratio_onto(
        ontology, langchain_onto=True)
    print('*ONTOLOGY ONLY INSTANTIATED CLASS RATIO*: ', icr_onto)
    icr = calculate_instantiated_class_ratio_kg(ontology, kg, langchain=True)
    print('*KG INSTANTIATED CLASS RATIO*: ', icr)
    ipr = calculate_instantiated_property_ratio(ontology, kg, langchain=True)
    print('*INSTANTIATED PROPERTY RATIO*: ', ipr)

    outfile = f'data/evaluations/structural_quality/structural_metrics_{kg_file.split('/')[-1].split('.')[0]}.csv'
    with open(outfile, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        results = {'ICR Onto': icr_onto, 'ICR KG': icr,
                   'IPR': ipr}
        for metric, value in results.items():
            csvwriter.writerow([metric, value])


def run_graph_comparison(graphs: List[Tuple[str, str]]):
    '''Checks the triple overlap between the LLM-generated graphs

    Args: 
        graphs: The file names where each graph is stored

    Returns: 
        None
    '''
    triple_list = []
    for model, kg_file in graphs:
        triples = generate_triples(kg_file)
        triple_list.append((model, triples))
    for combo in combinations(triple_list, 2):
        overlap = compare_relations_between_graphs(combo[0][1], combo[1][1])
        print(
            f'Triples of {combo[0][0]} and {combo[1][0]} have *{overlap}* overlapping relations')


def run_statistics(articles: List[str], graph_file: str, ontology_file: str, langchain: bool):
    '''Compute various statistics for articles, ontology, and knowledge graph.

    Args:
        articles: A list of articles to analyze.
        graph_file: The path to the JSON file containing the knowledge graph.
        ontology_file: The path to the JSON file containing the ontology.
        langchain: Flag indicating whether to use LangChain-specific processing.

    Returns:
        None
    '''
    ontology = read_json(ontology_file)
    graph = read_json(graph_file)
    # dataset statistics
    lengths = get_article_lengths(articles)

    # ontology statistics
    if langchain:
        entities, rels = len(ontology['node_props']), len(
            ontology['rel_props'])
    else:
        entities, rels = count_entities_relations_onto(ontology)
    print(f'''Count Entities in Onto: {
        entities} \nCount Relations in Onto: {rels}''')
    if not langchain:
        print(
            f'''Number of main classes in the ontology hierarchy: {len(ontology['entities'])}\n''')

    # graph statistics
    print(f'''Count Entities in KG: {
          len(graph['nodes'])} \nCount Relations in KG: {len(graph['relationships'])}''')

    outfile_general = f'''data/evaluations/statistics/size_statistics_{
        graph_file.split('/')[-1].split('.')[0]}.csv'''
    with open(outfile_general, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        # article lengths
        csvwriter.writerow(
            ['Original Article Lengths', 'Cleaned Article Lengths'])
        csvwriter.writerow([lengths['original'], lengths['cleaned']])
        csvwriter.writerow([])
        # ontology
        csvwriter.writerow(['No. Entities Onto', 'No. Relationships Onto'])
        csvwriter.writerow([entities, rels])
        csvwriter.writerow([])
        # hierarchy in ontology
        if not langchain:
            csvwriter.writerow(['Number of main classes in hierarchy'])
            csvwriter.writerow([len(ontology['entities'])])
        # kg
        csvwriter.writerow(['No. Entities KG', 'No. Relationships KG'])
        csvwriter.writerow([len(graph['nodes']), len(graph['relationships'])])
    if not langchain:
        hierarchy_distribution = calculate_children_counts(ontology)
        outfile_hierarchy = f'''data/evaluations/statistics/hierarchy_statistics_{
            ontology_file.split('/')[-1].split('.')[0]}.csv'''
        with open(outfile_hierarchy, 'w', newline='', encoding='utf-8') as f:
            csvwriter = csv.writer(f, delimiter=';')
            csvwriter.writerow(['Class', 'No. ChildrenEntities'])
            for key, value in hierarchy_distribution.items():
                csvwriter.writerow([key, value])

    graph_labels, used_labels, new_labels = count_labels(
        ontology, graph, langchain)
    print('Labels in Onto used:', len(used_labels))
    print('Labels not in Onto:', len(new_labels))

    outfile_labels = f'''data/evaluations/statistics/label_statistics_{
        graph_file.split('/')[-1].split('.')[0]}.csv'''
    with open(outfile_labels, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        csvwriter.writerow(['Label', 'Count'])
        for key, value in graph_labels.items():
            csvwriter.writerow([key, value])

    outfile_labels2 = f'''data/evaluations/statistics/onto_label_usage_statistics_{
        graph_file.split('/')[-1].split('.')[0]}.csv'''
    with open(outfile_labels2, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        csvwriter.writerow(['Labels in Onto Used', used_labels])
        csvwriter.writerow(['Count of Labels in Onto Used', len(used_labels)])
        csvwriter.writerow([])
        csvwriter.writerow(['New Labels (not in Onto)', new_labels])
        csvwriter.writerow(['Count of New Labels', len(new_labels)])

    graph_relations, used_rels, new_rels = count_relationships(
        ontology, graph, langchain)
    print('Rels in Onto used:', len(used_rels))
    print('Rels not in Onto:', len(new_rels))
    outfile_rels = f'''data/evaluations/statistics/relationship_statistics_{
        graph_file.split('/')[-1].split('.')[0]}.csv'''
    with open(outfile_rels, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        csvwriter.writerow(['Relationship', 'Count'])
        for key, value in graph_relations.items():
            csvwriter.writerow([key, value])

    outfile_rels2 = f'''data/evaluations/statistics/onto_relation_usage_statistics_{
        graph_file.split('/')[-1].split('.')[0]}.csv'''
    with open(outfile_rels2, 'w', newline='', encoding='utf-8') as f:
        csvwriter = csv.writer(f, delimiter=';')
        csvwriter.writerow(['Relations in Onto Used', used_rels])
        csvwriter.writerow(['Count of Relations in Onto Used', len(used_rels)])
        csvwriter.writerow([])
        csvwriter.writerow(['New Relations (not in Onto)', new_rels])
        csvwriter.writerow(['Count of New Relations', len(new_rels)])


def run_evaluation(articles: List[str], ontology_file: str, kg_file: str, structural: bool = True, content: bool = True, statistics: bool = True, langchain: bool = False):
    '''Run a comprehensive evaluation of an ontology and a knowledge graph, including structural, content, and statistical analyses.

    Args:
        articles: A list of articles to analyze.
        ontology_file: The path to the JSON file containing the ontology.
        kg_file: The path to the JSON file containing the knowledge graph.
        structural (optional): Whether to run structural evaluation. Defaults to True.
        content (optional): Whether to run content evaluation. Defaults to True.
        statistics (optional): Whether to run statistical evaluation. Defaults to True.
        langchain (optional): A flag indicating whether to use LangChain-specific processing. Defaults to False.

    Returns:
        None
    '''
    print(f'''-----Evaluation of ontology {ontology_file.split('/')
          [-1]} and knowledge graph {kg_file.split('/')[-1]}-----\n''')
    if statistics:
        print('STATISTICS:')
        run_statistics(articles, kg_file, ontology_file, langchain=langchain)
    if structural:
        print('STRUCTURAL QUALITY EVALUATION')
        if langchain:
            run_structural_evaluation_langchain_graph(ontology_file, kg_file)
        else:
            run_structural_evaluation(ontology_file, kg_file)
    if content:
        print('WIKIDATA KNOWLEDGE QUALITY EVALUATION')
        # wikidata
        run_content_evaluation(
            'data/gold_triples/wikidata/gold_triples.json', kg_file)

        print('DBPEDIA KNOWLEDGE QUALITY EVALUATION')
        # dbpedia
        run_content_evaluation(
            'data/gold_triples/dbpedia/gold_triples.json', kg_file)
