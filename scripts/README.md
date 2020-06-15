
## Scripts
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
