# T5

### aidayago2
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=1000 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=8 \
  --do_train
```

### eli5
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=500 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=8 \
  --do_train
```

### FEVER
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=1000 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=4 \
  --do_train
```

### HotpotQA
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=1000 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=4 \
  --do_train
```

### Natural Questions
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=500 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=4 \
  --do_train
```

### T-REx
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=500 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=4 \
  --do_train
```

### TriviaQA
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=2100 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=4 \
  --do_train
```

### Wizard of Wikipedia
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=1000 \
  --output_dir=$OUTPUT_DIR \
  --n_gpu=8 \
  --do_train
```

### Zeroshot RE
```
python ../finetune.py \
  --data_dir=${DATA_DIR} \
  --dataset=${DATASET} \
  --model_name_or_path=t5-base \
  --learning_rate=1e-3 \
  --num_train_epoch=1000 \
  --n_gpu=4 \
  --output_dir=$OUTPUT_DIR \
  --do_train
```