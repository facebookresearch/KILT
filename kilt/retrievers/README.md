# DrQA tf-idf

## install
```bash
pip install -e git+https://github.com/facebookresearch/DrQA#egg=DrQA
pip install pexpect==4.8
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
pip install -e git+https://github.com/facebookresearch/DPR.git#egg=DPR
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

# DPR distributed

Please follow instructions in the [Sphere](https://github.com/facebookresearch/Sphere) repository.

# BLINK

## install
```bash
pip install -e git+https://github.com/facebookresearch/BLINK.git#egg=BLINK
pip install flair
```

## download models

Download files in the `models` folder using the following script: [download_models.sh](https://github.com/facebookresearch/BLINK/blob/master/download_blink_models.sh)

And this file:
- [Wikipedia_title2id.p](http://dl.fbaipublicfiles.com/KILT/Wikipedia_title2id.p)

## run
```bash
python scripts/execute_retrieval.py -m blink -o predictions/blink
```

# BM25
Follow instructions in [`pyserini`](https://github.com/castorini/pyserini#installation) to download JAVA.
## install
```bash
pip install jnius
pip install pyserini==0.9.4.0
```

## run
```bash
python scripts/execute_retrieval.py -m bm25 -o predictions/bm25
```
