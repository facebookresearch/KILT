# Library for Knowledge Intensive Language Tasks

<img align="middle" src="img/KILT.jpg" height="256" alt="KILT">


| dataset | task | train | dev |
| ------------- | ------------- | ------------- | ------------- | 
| [FEVER](https://fever.ai) | Fact Checking | [fever-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl) (104,966 lines, 38MB)  | [fever-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl) (10,444 lines, 6MB) | 



### setup the env

```bash
conda create -n kilt37 -y python=3.7 && conda activate kilt37
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the retrieval evaluation

Setup retrievers following [this README](kilt/retrievers/README.md)

run the evaluation
```bash
export PYTHONPATH=.
python scripts/evaluate_ranking.py -m {dpr/kilt/drqa}
```

## KILT data format

```python
{'id': # original data point id if available otherwise unique id
 'input': # question / claim / sentence
 'output': [ # list of valid answers, at least one
    {
    'answer': # answer in textual form
    'provenance': [
        # list of relevant WikipediaPages / Spans as provenance for the answer from the ks
        {
            'wikipedia_id': ,  # *mandatory* - ID Wikipedia Page
            'title': , # Title Wikipedia Page
            'section': , # Section Wikipedia Page
            'start_paragraph_id': , # start paragraph id with relevant info
            'start_character': , 
            'end_paragraph_id': ,  # end paragraph id
            'end_character': , 
            'bleu_score': # 1.0 when gold data is exactly matched, lower for fuzzy matches 
            'meta': # dataset/task specific
        }
        ] 
      }
    ],
 'meta': {} # dataset/task specific
 }
```


## Knowledge Source

```python
from kilt.knowledge_source import KnowledgeSource

# get the knowledge souce
ks = KnowledgeSource()

# count entries - 5903530
ks.get_num_pages()

# get page by id
page = ks.get_page_by_id(27097632)

# get pages by title
page = ks.get_page_by_title("Michael Jordan")
```


Structure of each record:
```python
{
 'wikipedia_title': 'Email marketing',
 'wikipedia_id': 1101759, 
 'text': ['p1', 'p2',...., 'pn'], # list of paragraph text
 'anchors': [{"text":,"href":,"paragraph_id":,"start":,"end":} ]  , 
 'categories': 'comma separated list of categories'
 'history': # some info from wikipedia, including original url
 'wikidata_info': # wikidata info
 }
```
