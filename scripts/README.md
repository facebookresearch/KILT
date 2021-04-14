# Scripts

## execute retrieval

This script executes retrieval from a given model and store retrieved augmented KILT files

options:
* __--model_name/-m__ : retriever model name in {drqa,solr,dpr,blink,bm25}
* __--model_configuration/-c__ : model configuration 
* __--output_folder/-o__ : output folder


example
```
python scripts/execute_retrieval.py -m dpr -c kilt/configs/retriever/default_dpr.json -o predictions/dpr/
```

To setup the retrievers see [kilt/readers/README.md](this README).


## create kilt data paragraphs
The script `scripts/create_kilt_data_paragraphs.py` create chunks of all wikipedia in the format:
```
{'_id': str,
 'wikipedia_id': str,
 'wikipedia_title': str,
 'text': str,
 'anchors': [{'text': str,
   'href': str,
   'source': {'paragraph_id': int, 'start': int, 'end': int},
   'start': int,
   'end': int}, ...],
 'categories': str,
 'history': {'revid': int,
  'timestamp': str,
  'parentid': int,
  'pre_dump': bool,
  'pageid': int,
  'url': str},
 'sources': [{'paragraph_id': int, 'start': 0, 'end': int}, ...],
 'section': str}
```
It creates a `jsonl` file(s) where for each line there is a consecutive number (ID) and a `json` dictionary.

The script can launch 3 invididual steps that has to be run in order. Here an example. First preprocess uses `threads` to split the Knowledge Bases in even parts and it saves them into `folder`.
```bash
python create_kilt_data_paragraphs \
  --step preprocess \
  --folder "./kilt_data" \
  --threads 32
```

Then, the following creates chunks of size `chunk_size`. `rank`is the id of the portion of the dataset to compute. 
```bash
python create_kilt_data_paragraphs \
  --step main \
  --chunk_size 100
  --folder "./kilt_data" \
  --rank <int> 
```

Finally, we can merge all files with
```bash
python create_kilt_data_paragraphs \
  --step merge \
  --folder "./kilt_data" \
  --threads 32
```
