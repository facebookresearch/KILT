# DrQA tf-idf

## install
```bash
pip install -e git+https://github.com/facebookresearch/DrQA#egg=DrQA
pip install pexpect==4.8
```

change line 36 of `src/drqa/drqa/retriever/utils.py` to
```python
loader = np.load(filename, allow_pickle=True)
```

## run
```bash
python scripts/execute_retrieval.py -m drqa -o predictions/drqa
```

# DPR

## install
```bash
pip install -e git+git@github.com:facebookresearch/DPR.git#egg=DPR
```

change line 185 of `src/dpr/dense_retriever.py` to

```python
try:
    db_id, doc_vector = doc
except:
    title, db_id, doc_vector = doc
```

## run
```bash
python scripts/execute_retrieval.py -m dpr -o predictions/dpr
```

# BLINK

## install
```bash
pip install -e git+git@github.com:facebookresearch/BLINK.git#egg=BLINK
```

## run
```bash
python scripts/execute_retrieval.py -m blink -o predictions/blink
```