# Library for Knowledge Intensive Language Tasks

<img align="middle" src="img/KILT.jpg" height="256" alt="KILT">


## Get KILT data

### setup the env

```bash
conda create -n kilt37 -y python=3.7 && conda activate kilt37
pip install -r requirements.txt
```

### download the data

```bash
mkdir data
python scripts/donwload_all_kilt_data.py
python scripts/get_triviaqa_input.py
```

### KILT catalogue

| dataset | task | train | dev | test |
| ------------- | ------------- | ------------- | ------------- | ------------- | 
| [FEVER](https://fever.ai) | Fact Checking | [fever-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-train-kilt.jsonl)<br>(104,966 lines, 38.9MiB)  | [fever-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-dev-kilt.jsonl)<br>(10,444 lines, 6.17MiB) | [fever-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/fever-test_without_answers-kilt.jsonl)<br>(10,100 lines, 839kiB) | 
| [Natural Questions](https://ai.google.com/research/NaturalQuestions) | Open Domain QA | [nq-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/nq-train-kilt.jsonl)<br>(87,372 lines, 51.9MiB) | [nq-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/nq-dev-kilt.jsonl)<br>(2,837 lines, 7.94MiB) | [nq-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/nq-test_without_answers-kilt.jsonl)<br>(1,444 lines, 334kiB) | 
| [HotpotQA](https://hotpotqa.github.io) | Open Domain QA | [hotpotqa-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/hotpotqa-train-kilt.jsonl)<br>(88,869 lines, 52.8MiB) | [hotpotqa-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/hotpotqa-dev-kilt.jsonl)<br>(5,600 lines, 3.97MiB) | [hotpotqa-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/hotpotqa-test_without_answers-kilt.jsonl)<br>(5,569 lines, 778kiB) |
| [TriviaQA](http://nlp.cs.washington.edu/triviaqa) | Open Domain QA | [triviaqa-train_id-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/triviaqa-train_id-kilt.jsonl)<br>(61,844 lines, 102MiB) | [triviaqa-dev_id-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/triviaqa-dev_id-kilt.jsonl)<br>(5,359 lines, 9.81MiB) | [triviaqa-test_id_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/triviaqa-test_id_without_answers-kilt.jsonl)<br>(6,586 lines, 123kiB) |
| [ELI5](https://facebookresearch.github.io/ELI5/explore.html) | Open Domain QA | [eli5-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-train-kilt.jsonl)<br>(272,634 lines, 548MiB) | [eli5-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-dev-kilt.jsonl)<br>(1,947 lines, 18.7MiB) | [eli5-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/eli5-test_without_answers-kilt.jsonl)<br>(1,040 lines, 185kiB) | 
| [T-REx](https://hadyelsahar.github.io/t-rex) | Slot Filling | [trex-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/trex-train-kilt.jsonl)<br>(2,284,168 lines, 1.75GiB) | [trex-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/trex-dev-kilt.jsonl)<br>(5,000 lines, 3.80MiB) | [trex-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/trex-test_without_answers-kilt.jsonl)<br>(5,000 lines, 896kiB) | 
| [Zero-Shot RE](http://nlp.cs.washington.edu/zeroshot) | Slot Filling | [structured_zeroshot-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-train-kilt.jsonl)<br>(147,909 lines, 71.4MiB) | [structured_zeroshot-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-dev-kilt.jsonl)<br>(3,724 lines, 2.27MiB) | [structured_zeroshot-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/structured_zeroshot-test_without_answers-kilt.jsonl)<br>(4,966 lines, 1.22MiB) |
| [AIDA CoNLL-YAGO](https://www.mpi-inf.mpg.de/departments/databases-and-information-systems/research/ambiverse-nlu/aida/downloads) | Entity Linking | [aidayago2-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/aidayago2-train-kilt.jsonl)<br>(18,395 lines, 70.1MiB) | [aidayago2-dev-kilt.jsonl]( http://dl.fbaipublicfiles.com/KILT/aidayago2-dev-kilt.jsonl)<br>(4,784 lines, 21.1MiB) | [aidayago2-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/aidayago2-test_without_answers-kilt.jsonl)<br>(4,463 lines, 14.4MiB) | 
| [WNED-WIKI](https://github.com/U-Alberta/wned) | Entity Linking | - | [wned-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wned-dev-kilt.jsonl)<br>(3,396 lines, 12.9MiB) | [wned-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wned-test_without_answers-kilt.jsonl)<br>(3,376 lines, 13.3MiB) | 
| [WNED-CWEB](https://github.com/U-Alberta/wned) | Entity Linking | - | [cweb-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/cweb-dev-kilt.jsonl)<br>(5,599 lines, 90.2MiB)  | [cweb-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/cweb-test_without_answers-kilt.jsonl)<br>(5,543 lines, 100MiB) | 
| [Wizard of Wikipedia](https://parl.ai/projects/wizard_of_wikipedia) | Dialogue | [wow-train-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wow-train-kilt.jsonl)<br>(94,577 lines, 71.9MiB) | [wow-dev-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wow-dev-kilt.jsonl)<br>(3,058 lines, 2.42MiB) | [wow-test_without_answers-kilt.jsonl](http://dl.fbaipublicfiles.com/KILT/wow-test_without_answers-kilt.jsonl)<br>(2,944 lines, 1.29MiB)|
 


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
