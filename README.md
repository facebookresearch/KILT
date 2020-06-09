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
page = ks.get_page_by_title("Michael Jordan")
