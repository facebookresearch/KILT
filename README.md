# Library for Knowledge Intensive Language Tasks

<img align="middle" src="img/KILT.jpg" height="256" alt="KILT">


| dataset | task | train | dev | test |
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [FEVER](https://fever.ai) | Fact Checking | [fever-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl)<br>(104,966 lines, 38MB)  | [fever-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl)<br>(10,444 lines, 6MB) | fever-test_input-kilt.jsonl<br>(10,100 lines) | 
| [Natural Questions](https://ai.google.com/research/NaturalQuestions) | Open Domain QA | [nq-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl)<br>(87,372 lines, 50MB) | [nq-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl)<br>(2,837 lines, 8MB) | nq-test_input-kilt.jsonl<br>(1,444 lines) | 
| [HotpotQA](https://hotpotqa.github.io) | Open Domain QA | [hotpotqa-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl)<br>(88,869 lines, 51MB) | [hotpotqa-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl)<br>(5,600 lines, 4MB) | hotpotqa-test_input-kilt.jsonl<br>(5,569 lines) |
| [TriviaQA](http://nlp.cs.washington.edu/triviaqa) | Open Domain QA | triviaqa-train_id-kilt.jsonl | triviaqa-dev_id-kilt.jsonl | triviaqa-test_id_input-kilt.jsonl |
| [ELI5](https://facebookresearch.github.io/ELI5/explore.html) | Open Domain QA | [eli5-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl)<br>(272,634 lines, 523MB) | [eli5-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl)<br>(1,947 lines, 18MB) | eli5-test_input-kilt.jsonl<br>(1,040 lines) | 
| [T-REx](https://hadyelsahar.github.io/t-rex) | Slot Filling | [trex-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/trex-train-kilt.jsonl)<br>(2,284,168 lines, 1.7GB) | [trex-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl)<br>(5,000 lines, 8MB) | trex-test_input-kilt.jsonl<br>(5,000 lines) | 
| [Zero-Shot RE](http://nlp.cs.washington.edu/zeroshot) | Slot Filling | [structured_zeroshot-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-train-kilt.jsonl)<br>(147,909 lines, 69MB) | [structured_zeroshot-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl)<br>(3,724 lines, 3MB) | structured_zeroshot-test_input-kilt.jsonl<br>(4,966 lines) |
| [AIDA CoNLL-YAGO](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) | Entity Linking | [aidayago2-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/aidayago2-train-kilt.jsonl)<br>(18,395 lines, 67MB) | [aidayago2-dev-kilt.jsonl]( http://dl.fbaipublicfiles.com/KILT/aidayago2-dev-kilt.jsonl)<br>(4,784 lines, 21MB) | aidayago2-test_input-kilt.jsonl<br>(4,463 lines) | 
| [WNED-WIKI](https://github.com/U-Alberta/wned) | Entity Linking | - | [wned-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl)<br>(3,396 lines, 13MB) | wned-test_input-kilt.jsonl<br>(3,376 lines) | 
| [WNED-CWEB](https://github.com/U-Alberta/wned) | Entity Linking | - | [cweb-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/cweb-dev-kilt.jsonl)<br>(5,599 lines, 87MB)  | cweb-test_input-kilt.jsonl<br>(5,543 lines) | 
 

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
