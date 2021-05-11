# Fusion-in-Decoder

### Install Fusion-in-Decoder

`pip install -e git+https://github.com/facebookresearch/FiD#egg=FiD`

### Convert KILT data format to FiD format

```shell
python preprocess.py input_data.jsonl outputpath
```

### Train FiD

```shell
python src/fid/train_reader.py \
        --use_checkpoint \
        --train_data train_data.json \
        --eval_data eval_data.json \
        --model_size base \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_experiment \
        --checkpoint_dir checkpoint \
```

### Eval FiD

```shell
python src/fid/test_reader.py \
        --model_path checkpoint/my_experiment/checkpoint/best_dev \
        --eval_data eval_data.json \
        --per_gpu_batch_size 1 \
        --n_context 100 \
        --name my_test \
        --checkpoint_dir checkpoint \
```

### Convert to KILT format for eval

```shell
python postprocess.py checkpoint/my_test/final_output.json my_kilt_output.jsonl initial_input_data.jsonl
```

