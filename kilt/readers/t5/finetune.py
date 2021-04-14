# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import argparse
import glob
import logging
import os
import time

import torch
from torch.utils.data import DataLoader
from torch.utils.data.dataset import ConcatDataset
from transformers import get_linear_schedule_with_warmup
from transformers.tokenization_utils import trim_batch

from base_transformer import BaseTransformer, add_generic_args, generic_train
from data import KiltDataset, seq2seq_to_kilt, dataset_config
from eval_downstream import normalize_answer

logger = logging.getLogger(__name__)


class Seq2seqTransformer(BaseTransformer):

    def __init__(self, hparams):
        super().__init__(hparams, num_labels=None)
        self.lr_scheduler = None
        self.devsets = {}
        self.em = -1
        self.dataset_list = self.hparams.dataset.split(',')
        self.eval_batch_size = 100000
        self.train_batch_size = 100000
        self.source_length = -1
        self.target_length = -1

        special_tokens = []

        for i in range(0, 101):
            special_tokens.append('<extra_id_' + str(i) + '>')

        special_tokens.extend(['[START_ENT]', '[END_ENT]', 'Question Answering:', 'Entity Linking:',
                               'Fact Checking:', 'Dialogue:', 'Relation Extraction:', '[SEP]'])  #
        self.tokenizer.add_special_tokens(
            {'additional_special_tokens': special_tokens})

        fevers_classes = ["<SUPPORTS>", "<REFUTES>"]

        self.tokenizer.add_tokens(fevers_classes)

        self.model.resize_token_embeddings(len(self.tokenizer))

        self.bad_words = [[self.tokenizer.convert_tokens_to_ids(bad_word)] for bad_word in
                          self.tokenizer.additional_special_tokens]

        for d in self.dataset_list:
            train_batch = int(dataset_config[d]['train_batch'])
            eval_batch = int(dataset_config[d]['eval_batch'])
            source_length = int(dataset_config[d]['source_length'])
            target_length = int(dataset_config[d]['target_length'])
            if train_batch < self.train_batch_size:
                self.train_batch_size = train_batch
            if eval_batch < self.eval_batch_size:
                self.eval_batch_size = eval_batch
            if source_length > self.source_length:
                self.source_length = source_length
            if target_length > self.target_length:
                self.target_length = target_length

        self.data_dir = self.hparams.data_dir
        self.output_dir = self.hparams.output_dir

    def forward(self, input_ids, attention_mask=None, decoder_input_ids=None, lm_labels=None):
        return self.model(
            input_ids, attention_mask=attention_mask, lm_labels=lm_labels,
        )

    def _step(self, batch):
        pad_token_id = self.tokenizer.pad_token_id
        source_ids, source_mask, y, ids = batch["source_ids"], batch["source_mask"], batch["target_ids"], batch["ids"]

        lm_labels = y.clone()
        lm_labels[y == pad_token_id] = -100

        outputs = self(source_ids, attention_mask=source_mask, lm_labels=lm_labels, )

        loss = outputs[0]

        return loss

    def training_step(self, batch, batch_idx):
        loss = self._step(batch)

        tensorboard_logs = {"train_loss": loss}
        return {"loss": loss, "log": tensorboard_logs}

    def validation_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        source_ids, source_mask, y = KiltDataset.trim_seq2seq_batch(batch, pad_token_id)
        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=self.target_length,
            repetition_penalty=1,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            do_sample=False,
            top_p=0.95,
            top_k=50,
            bad_words_ids=self.bad_words
        )

        preds = [self.tokenizer.decode(g) for g in generated_ids]
        target = [self.tokenizer.decode(t) for t in y]
        loss = self._step(batch)
        sources = [self.tokenizer.decode(s) for s in source_ids]

        return {"val_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["ids"]}

    def validation_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}

        preds = []
        ids = []
        sources = []
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])
        em = 0
        for q_id, pred in set(zip(ids, preds)):
            targets = [normalize_answer(x) for x in self.devsets[q_id]]

            if normalize_answer(pred) in targets:
                em = em + 1
        if em > self.em:
            self.em = em
            self.trainer.save_checkpoint(self.output_dir + '/' + "best_em.ckpt")
            seq2seq_to_kilt(set(ids), set(sources), set(preds), self.hparams.output_dir,
                            self.hparams.dataset, 'dev')
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs, "EM": em}

    def test_step(self, batch, batch_idx):
        pad_token_id = self.tokenizer.pad_token_id

        source_ids, source_mask, y = KiltDataset.trim_seq2seq_batch(batch, pad_token_id)
        # NOTE: the following kwargs get more speed and lower quality summaries than those in evaluate_kilt_task.py

        generated_ids = self.model.generate(
            input_ids=source_ids,
            attention_mask=source_mask,
            num_beams=1,
            max_length=self.target_length,
            repetition_penalty=1,
            length_penalty=1.0,
            early_stopping=True,
            use_cache=True,
            do_sample=False,
            top_p=0.95,
            top_k=50,
            bad_words_ids=self.bad_words
        )
        preds = [self.tokenizer.decode(g) for g in generated_ids]
        target = [self.tokenizer.decode(t) for t in y]
        loss = self._step(batch)
        sources = [self.tokenizer.decode(s) for s in source_ids]
        return {"val_loss": loss, 'sources': sources, "preds": preds, "target": target, "ids": batch["ids"]}

    def test_end(self, outputs):
        sources = []
        preds = []
        ids = []
        for batch in outputs:
            sources.extend(batch['sources'])
            preds.extend(batch['preds'])
            ids.extend(batch["ids"])

        seq2seq_to_kilt(ids, sources, preds, self.hparams.output_dir, self.hparams.dataset, 'test')

        return self.test_epoch_end(outputs)

    def test_epoch_end(self, outputs):
        avg_loss = torch.stack([x["val_loss"] for x in outputs]).mean()
        tensorboard_logs = {"val_loss": avg_loss}
        return {"avg_val_loss": avg_loss, "log": tensorboard_logs}

    def collate_fn(self, batch):
        input_ids = torch.stack([x["source_ids"] for x in batch])
        masks = torch.stack([x["source_mask"] for x in batch])
        target_ids = torch.stack([x["target_ids"] for x in batch])
        ids = [x["id"] for x in batch]
        pad_token_id = self.tokenizer.pad_token_id
        y = trim_batch(target_ids, pad_token_id)
        source_ids, source_mask = trim_batch(input_ids, pad_token_id, attention_mask=masks)
        return {"source_ids": source_ids, "source_mask": source_mask, "target_ids": y, "ids": ids}

    def get_dataloader(self, type_path: str, batch_size: int, shuffle: bool = False) -> DataLoader:
        datasets = []
        for d in self.dataset_list:
            datasets.append(
                KiltDataset(self.tokenizer, self.data_dir, d, type_path, self.source_length, self.target_length,
                            self.output_dir))
        if type_path == 'dev':
            for x in datasets:
                self.devsets.update(x.id_targets)
        concat_dataset = ConcatDataset(datasets)
        dataloader = DataLoader(concat_dataset, batch_size=batch_size, shuffle=shuffle, collate_fn=self.collate_fn)

        print(type_path, dataloader.batch_size, concat_dataset.__len__())
        return dataloader

    def train_dataloader(self) -> DataLoader:
        dataloader = self.get_dataloader("train", batch_size=self.train_batch_size, shuffle=True)
        t_total = (
                (len(dataloader.dataset) // (self.train_batch_size * max(1, self.hparams.n_gpu)))
                // self.hparams.gradient_accumulation_steps
                * float(self.hparams.num_train_epochs)
        )
        scheduler = get_linear_schedule_with_warmup(
            self.opt, num_warmup_steps=self.hparams.warmup_steps, num_training_steps=t_total
        )
        self.lr_scheduler = scheduler

        return dataloader

    def val_dataloader(self) -> DataLoader:
        return self.get_dataloader("dev", batch_size=self.eval_batch_size)

    def test_dataloader(self) -> DataLoader:
        return self.get_dataloader("test", batch_size=self.eval_batch_size)

    @staticmethod
    def add_model_specific_args(arg_parser, root_dir):
        BaseTransformer.add_model_specific_args(arg_parser, root_dir)

        arg_parser.add_argument(
            "--data_dir",
            default=None,
            type=str,
            required=True,
            help="The input data dir. Should contain the dataset files for the task.",
        )
        arg_parser.add_argument("--dataset", required=True, type=str)

        return arg_parser


def main(arguments):
    # If output_dir not provided, a folder will be generated in pwd
    if not arguments.output_dir:
        arguments.output_dir = os.path.join("./results", f"{arguments.task}_{time.strftime('%Y%m%d_%H%M%S')}", )
        os.makedirs(arguments.output_dir)
    model = Seq2seqTransformer(arguments)
    trainer = generic_train(model, arguments)

    if arguments.do_predict:
        checkpoints = list(
            sorted(glob.glob(os.path.join(arguments.output_dir, "*.ckpt"), recursive=True)))
        model = model.load_from_checkpoint(checkpoints[-1])
        model.hparams.dataset = arguments.dataset
        model.dataset_list = arguments.dataset.split(',')

        trainer.test(model)


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    add_generic_args(parser)
    parser = Seq2seqTransformer.add_model_specific_args(parser, os.getcwd())
    args = parser.parse_args()

    main(args)
