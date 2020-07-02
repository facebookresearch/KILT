# Library for Knowledge Intensive Language Tasks

<img align="middle" src="img/KILT.jpg" height="256" alt="KILT">

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

## Knowledge Source

```python
from kilt.knowledge_source import KnowledgeSource

# get the knowledge souce
ks = KnowledgeSource()

# count entries - 5903530
ks.get_num_pages()

# get page by id
page = ks.get_page_by_id(27097632)
```

Structure of each record:
```python
{
 'wikipedia_title': 'Email marketing',
 'wikipedia_id': 1101759, 
 'text': ['p1', 'p2',...., 'pn'], # list of paragraph text
 'anchors': [{"text":,"href":,"paragraph_id":,"start":,"end":} ]  , 
 'categories': 'comma separated list of categories'
 }
```
# get pages by title
page = ks.get_page_by_title("Michael Jordan")
```
