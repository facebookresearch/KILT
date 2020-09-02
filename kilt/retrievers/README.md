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

## download models

Download the following files in the `models` folder.

- [kilt_db_simple.npz](http://dl.fbaipublicfiles.com/KILT/kilt_db_simple.npz)

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

## download models

Download the following files in the `models` folder.

- [dpr_multi_set_hf_bert.0](http://dl.fbaipublicfiles.com/KILT/dpr_multi_set_hf_bert.0)
- [kilt_passages_2048_0.pkl](http://dl.fbaipublicfiles.com/KILT/kilt_passages_2048_0.pkl)
- [kilt_w100_title.tsv](http://dl.fbaipublicfiles.com/KILT/kilt_w100_title.tsv)
- [mapping_KILT_title.p](http://dl.fbaipublicfiles.com/KILT/mapping_KILT_title.p)

## run
```bash
python scripts/execute_retrieval.py -m dpr -o predictions/dpr
```

# BLINK

## install
```bash
pip install -e git+git@github.com:facebookresearch/BLINK.git#egg=BLINK
```

## download models

Download files in the `models` folder using the following script: [download_models.sh](https://github.com/facebookresearch/BLINK/blob/master/download_models.sh)

## run
```bash
python scripts/execute_retrieval.py -m blink -o predictions/blink
```
