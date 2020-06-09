# KILT: a benchmark for Knowledge Intensive Language Tasks

Main document https://fb.quip.com/AgxuAqUrKOuA

### setup the env

```bash
conda create -n kilt37 -y python=3.7 && conda activate kilt37
pip install -r requirements.txt
python -m spacy download en_core_web_sm
```

## Run the evaluation
install DrQA
```bash
pip install -e git+https://github.com/facebookresearch/DrQA#egg=DrQA
pip install pexpect==4.8
```

Fix a bug by changing line 36 of `src/drqa/drqa/retriever/utils.py` to

```python
loader = np.load(filename, allow_pickle=True)
```

run the evaluation
```bash
python scripts/evaluate.py
```


## Useful functions for the mapping

### Knowledge Source

```python
from kilt.knowledge_source import KnowledgeSource

# get the knowledge souce
ks = KnowledgeSource()

# count entries - 5903531
ks.get_num_pages()

# get page by id
page = ks.get_page_by_id(27097632)

# get pages by title
pages = ks.get_pages_by_title("Michael Jordan")

# get page by url
url = "https://en.wikipedia.org/wiki/Tara_Chand_(Indian_politician)"
page = ks.get_page_from_url(url)
```

### Matching utilities

```python
import kilt.utils as utils
import spacy
nlp = spacy.load("en_core_web_sm")

# match an *answer* inside a wikipedia *page*
paragraph_id, start_character, end_character, bleu = utils.match_answer(answer, page, nlp = nlp)

print("match: '{}'".format(page["text"][paragraph_id][start_character:end_character]))
```


## Adding a dataset
Please build as new dataset as a subclass of the `Dataset` interface (`see kilt/datasets/base_dataset.py`).
This will require implementing 2 functions:
- `get_chunks` - which will return a list of chunks of datapoints of the dataset (this function should encapsulate reading the dataset from a file splitting into chunks)
- `process_chunk` - which will return a list of mapped datapoints from a given chunk and optionally metadata.