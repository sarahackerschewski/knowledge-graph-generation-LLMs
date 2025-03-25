# Automatic Generation of Knowlegde Graphs with LLMs
This repository contains the code to my master thesis 'Automatic Generation of Useful Knowledge Graphs with Large Language Models'. The purpose of this thesis is the generation of knowledge graphs (KGs) using Large Language Models (LLMs)

## Installation
To get started with the project, and install the necessary requirements

    # Clone the repo
    gh repo clone sarahackerschewski/knowledge-graph-generation-LLMs
    
    # Navigate into the project directory
    cd ../knowledge-graph-generation-LLMs
    
    # Install dependencies
    pip install -r requirements.txt
  Make sure to create a `.env` file in `data/config/` following the structure of the [example template](https://github.com/sarahackerschewski/knowledge-graph-generation-LLMs/blob/b2596d1de6b87da52e6c41d3dec34664c236cd99/data/config/.env_example))
## Data
  The data used for this project was Wikipedia articles taken from the Wikipedia scrape of the 20.05.2024 from the [Wikimedia Dump page](https://dumps.wikimedia.org/). (The dump is not available anymore since more scraping was done since then, the whole dump can be found [here](https://drive.google.com/drive/folders/1iQSfY9YcW1MsOXBcy6we56wAuQJ3B0Q2))
  The cleaned and sampled dataset, consisting of 1000 articles, can be found in `data/20240520_wikipedia_cleaned_sample.json`

  All data, which was too large was uploaded to the Google Drive. The links for that can be found in [data_links.txt](https://github.com/sarahackerschewski/knowledge-graph-generation-LLMs/blob/b2596d1de6b87da52e6c41d3dec34664c236cd99/data/data_links.txt)

## Experiments
### Models
The experiments were conducted using OpenAI's GPT-4 series. For comparison, GPT-4o, GPT-4o Mini and GPT-4 Turbo were used to generate three separate KGs. Here, an OpenAI Key is necessary to run the code creating the graphs. 

### Approaches

 1. The newly introduced approach: first creating an ontology from the articles and then using ontology and articles to create the KG.
 2. LangChain's [LLMGraphTransformer](https://api.python.langchain.com/en/latest/graph_transformers/langchain_experimental.graph_transformers.llm.LLMGraphTransformer.html) using GPT-4o (for comparison). The resulting ontology and graph contained the prefix langchain to differentiate the JSONs from the JSONs of approach 1.

The implementations can be found [here](https://github.com/sarahackerschewski/knowledge-graph-generation-LLMs/blob/b2596d1de6b87da52e6c41d3dec34664c236cd99/src/kg_generation/generate_kg.py). The prompts for all subtasks of the two-step KG generation are stored [here](https://github.com/sarahackerschewski/knowledge-graph-generation-LLMs/tree/b2596d1de6b87da52e6c41d3dec34664c236cd99/data/prompts).
#### Ontologies
The batch JSONs for the ontologies created by each model can be found in `data/ontologies/batch_responses/`. The merged ontologies after each extraction step ('basic', 'hierarchy', 'property') can be found in `data/ontologies/merged_batches/`.  The final ontology, i.e. with merged batches, duplicate removal, hierarchy and properties in the classes are saved as `data/ontologies/merged_batches/final_ontology_[model].json`
_Attention: the GPT-4 Turbo ontology includes properties extracted with GT-4o Mini. The ontology only using Turbo has_ `only_gpt-4-turbo_bad` _in the file name_ 


#### Knowledge Graphs
Like for the ontologies, the KG JSON batches for the models are saved in `data/knowledge_graphs/batch_responses/` and the merged, cleaned graphs were stored in `data/knowledge_graphs/merged_batches/kg_[model].json` 


### Evaluation
Part of the thesis was an extensive KG evaluation framework. The framework consisted of structural quality evaluations, based on Seo et al.'s paper *'Structural Quality Metrics to Evaluate Knowledge Graph   Quality'* (2023), knowledge quality evaluation with exact and partial accuracy, application quality with a Question-Answering task and some additional statistics undermining structural and content quality evaluation.
The implementations of the metrics can be found at `src/evaluation/`. 
The results of the evaluations are also stored in the repo at [evaluations](https://github.com/sarahackerschewski/knowledge-graph-generation-LLMs/tree/b2596d1de6b87da52e6c41d3dec34664c236cd99/data/evaluations). 

## Usage
To run the experiments you can use the following command:

`python main.py -[method] -[model] -[eval_only] -[eval metrics to use]`

The arguments can look like this:

`[method]` -> *auto* for two-step approach, *langchain* for langchain

`[model to use]` -> *gpt-4o, gpt-4-turbo, gpt-4o-mini*

`[eval_only]` -> (optional) if only evaluation should be performed

`[evaluation methods to use]` -> (optional: if not added, defaults to all are used) otherwise any of the methods to use: *-content*, *-structural*, *-statistics*

Example:

`python main.py -auto -gpt-4o` creates a KG with the approach 1 using GPT-4o and evaluates all quality metrics for this graph.

## Results
The following table shows the final structural and knowledge quality evaluations.
|                       | GPT-4o | GPT-4Turbo | GPT-4omini | LangChain  |
|------------------------------|--------|------------|------------|------------|
| Entities Onto                | 496    | 499        | 203        | 326        |
| Relationships Onto           | 423    | 346        | 326        | 2305       |
| Entities KG                  | 4586   | 2938       | 5329       | 5781       |
| Relationships KG/No. Triples | 3965   | 2722       | 4794       | 5250       |
| ICR Onto                     | 0.6996 | 0.6052     | 0.6897     | 0.5215     |
| ICR KG                       | 0.7823 | 0.5251     | **0.936**      | ***0.9785***     |
| IPR                          | 0.903  | 0.8501     | **0.955**      | 0.7990     |
| SPA                          | 1.0081 | 1.8717     | **2.1035**     | -          |
| SPA V2                       | 1.4971 | 1.9022     | **2.1897**     | -          |
| IMI                          | 0.8158 | 0.6439     | **0.8388**     | -          |
| DBpedia exact accuracy       | **0.1829** | 0.1477     | 0.136      | 0.0581     |
| DBpedia partial accuracy     | **0.7203** | 0.638      | 0.6333     | 0.6328     |
| Wikidata exact accuracy      | **0.232**  | 0.1929     | 0.1154     | 0.0535     |
| Wikidata partial accuracy    | **0.5319** | 0.5063     | 0.432      | 0.3272     |

## Conclusion
The results suggested that additionally making use of the ontology when generating an KG from unstructured text, produces a more concise and consistent graph, particularly with GPT-4o, but also highlighted challenges such as hallucinations and incomplete knowledge. A comparison with LangChain’s one-step approach revealed trade-offs between structure and content coverage. Furthermore, the findings emphasized the need for a unified evaluation framework to assess KG quality comprehensively. Future research should address these limitations through advanced prompt engineering, refined evaluation methods, and improved techniques to enhance the factual accuracy and practicality of LLM-generated KGs.




