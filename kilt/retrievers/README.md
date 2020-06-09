# DrQA tf-idf

```bash
pip install -e git+https://github.com/facebookresearch/DrQA#egg=DrQA
pip install pexpect==4.8
```

change line 36 of `src/drqa/drqa/retriever/utils.py` to

```python
loader = np.load(filename, allow_pickle=True)
```

# DPR
```bash
pip install -e git+git@github.com:fairinternal/DPR.git#egg=DPR
```

change line 168 of `src/dpr/dense_retriever.py` to

```python
try:
    db_id, doc_vector = doc
except:
    title, db_id, doc_vector = doc
```


# BLINK
```bash
pip install -e git+git@github.com:fairinternal/BLINK-Internal.git#egg=BLINK
```