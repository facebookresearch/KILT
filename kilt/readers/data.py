# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.


import configparser
import fcntl
import gzip
import json
import os
import pathlib

import torch.utils.data
from transformers.tokenization_utils import trim_batch

dataset_task_map = {'nq': "Question Answering", "aidayago2": "Entity Linking", "cweb": "Entity Linking",
                    "fever": "Fact Checking", "hotpotqa": "Question Answering",
                    "triviaqa": "Question Answering", "wned": "Entity Linking", "wow": "Dialogue",
                    "zeroshot": "Relation Extraction", "trex":"Slot Filling", "eli5":"Question Answering"}

dataset_config = configparser.ConfigParser()
location = os.path.join(pathlib.Path(__file__).parent, 'config_file')
dataset_config.read(location)


def encode_seq(tokenizer, seqs, max_length, out_dir, dataset, side='source', type_path='train', pad_to_max_length=True,
               return_tensors="pt"):
    examples = []
    lengths = []

    output_file = os.path.join(out_dir, dataset + "-" + type_path + "-" + side + ".encoded")
    with open(output_file, "w") as f_out:
        texts = []
        for text in seqs:

            if dataset_task_map[dataset] == 'Entity Linking' and side == 'source':
                length = int(int(dataset_config[dataset]['source_length']) / 2)
                mention_start = text.find('[START_ENT]')
                mention_end = text.find('[END_ENT]')
                left = text[0:mention_start]
                right = text[mention_end + len('[END_ENT]'):]

                left_ids = tokenizer.encode(left)
                right_ids = tokenizer.encode(right)
                left = tokenizer.decode(left_ids[max(0, len(left_ids) - length):len(left_ids)])
                right = tokenizer.decode(right_ids[0:min(len(right_ids), length)])
                text = left + ' ' + text[mention_start:mention_end] + '[END_ENT] ' + right

            if dataset == 'wow' and side == 'source':
                text = text.replace('\n', '[SEP]')

            if dataset == 'fever' and side == 'target':
                if text == "REFUTES":
                    text = "<REFUTES>"
                if text == "SUPPORTS":
                    text = "<SUPPORTS>"

            txt = text if side == 'target' else \
                dataset_task_map[dataset] + ": " + text
            txt = txt + tokenizer.eos_token
            texts.append(txt)

        if dataset == 'wow' and side == 'source':
            tokenized = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=max_length, pad_to_max_length='left',
                return_tensors=return_tensors,
            )
        else:
            tokenized = tokenizer.batch_encode_plus(
                texts, add_special_tokens=True, max_length=max_length, pad_to_max_length=pad_to_max_length,
                return_tensors=return_tensors,
            )

        #lengths.append(tokenized["input_ids"].size()[1])

        for input in tokenized["input_ids"]:
            tokens = tokenizer.convert_ids_to_tokens(input)
            f_out.write(' | '.join(tokens) + "\n")



    return tokenized


class KiltDataset(torch.utils.data.Dataset):
    def __init__(
            self,
            tokenizer,
            data_dir,
            dataset,
            type_path,
            max_source_length,
            max_target_length,
            output_dir
    ):
        super().__init__()
        self.tokenizer = tokenizer

        self.source = []
        self.target = []

        # self.ids, raw_sources, raw_targets, self.id_targets = nq_jsonl_to_tsv(data_dir, type_path)

        self.ids, raw_sources, raw_targets, self.id_targets = kilt_to_seq2seq(data_dir, dataset, type_path)

        self.source = encode_seq(tokenizer, raw_sources, max_source_length, output_dir, dataset, 'source', type_path)
        self.target = encode_seq(tokenizer, raw_targets, max_target_length, output_dir, dataset, 'target', type_path)

    def __len__(self):
        return len(self.source["input_ids"])

    def __getitem__(self, index):

        source_ids = self.source["input_ids"][index].squeeze()
        target_ids = self.target["input_ids"][index].squeeze()
        src_mask = self.source["attention_mask"][index].squeeze()
        q_id = self.ids[index]
        return {"source_ids": source_ids, "source_mask": src_mask, "target_ids": target_ids, "id": q_id}

    @staticmethod
    def trim_seq2seq_batch(batch, pad_token_id):
        target_ids = trim_batch(batch["target_ids"], pad_token_id)
        source_ids, source_mask = trim_batch(batch["source_ids"], pad_token_id, attention_mask=batch["source_mask"])
        return source_ids, source_mask, target_ids


def kilt_to_seq2seq(data_dir, dataset, type_path):
    data_file = pathlib.Path(os.path.join(data_dir, dataset + '-' + type_path + "-kilt.jsonl"))
    sources = []
    targets = []
    ids = []
    id_targets = {}
    if not data_file.exists():
        return ids, sources, targets

    with open(data_file, "r") as f:

        for line in f.readlines():
            qa = json.loads(line)
            q_id = qa['id']
            question = qa['input']
            output = qa['output']
            if len(output) == 0:
                continue
            answers = set()
            id_targets[q_id] = []

            for out in output:
                if 'answer' not in out.keys():
                    continue

                answer = out['answer']
                answers.add(answer)
                id_targets[q_id].append(answer)
            if type_path == 'test':
                sources.append(question)
                targets.append(answers.pop())
                ids.append(q_id)
            else:
                for answer in answers:
                    sources.append(question)
                    targets.append(answer)
                    ids.append(q_id)
    return ids, sources, targets, id_targets


def seq2seq_to_kilt(ids, sources, targets, output_dir, dataset, type_path):
    data_file = os.path.join(output_dir, dataset + '-' + type_path + "-kilt.jsonl")

    with open(data_file, "a+") as output_file:
        data = []
        for q_id, s, t in zip(ids, sources, targets):
            qa = {"id": q_id, 'input': s, 'output': []}
            a = {'answer': t, 'provenance': []}
            qa['output'].append(a)
            data.append(json.dumps(qa))
        fcntl.flock(output_file, fcntl.LOCK_EX)
        if os.stat(data_file).st_size > 0:
            output_file.write('\n')
        output_file.write('\n'.join(data))
        fcntl.flock(output_file, fcntl.LOCK_UN)


def nq_jsonl_to_tsv(data_dir, type_path):
    def extract_answer(answer_tokens, span):
        """Reconstruct answer from token span and remove extra spaces."""
        start, end = span["start_token"], span["end_token"]
        ans = " ".join(answer_tokens[start:end])
        # Remove incorrect spacing around punctuation.
        ans = ans.replace(" ,", ",").replace(" .", ".").replace(" %", "%")
        ans = ans.replace(" - ", "-").replace(" : ", ":").replace(" / ", "/")
        ans = ans.replace("( ", "(").replace(" )", ")")
        ans = ans.replace("`` ", "\"").replace(" ''", "\"")
        ans = ans.replace(" 's", "'s").replace("s ' ", "s' ")
        return ans

    count = 0
    ids = []
    sources = []
    targets = []
    id_targets = {}
    in_fname = data_dir + '/' + type_path + '.jsonl.gz'

    for line in gzip.open(in_fname, "rb"):
        ex = json.loads(line)

        # Remove any examples with more than one answer.

        # Questions in NQ do not include a question mark.
        q_id = ex['annotations'][0]['annotation_id']
        question = ex["question_text"] + "?"
        answers = []
        for answer_span in ex['annotations'][0]['short_answers']:
            tokens = []
            # Handle the two document formats in NQ (tokens or text).
            if "document_tokens" in ex:
                tokens = [t["token"] for t in ex["document_tokens"]]
            elif "document_text" in ex:
                tokens = ex["document_text"].split(" ")
            answer = extract_answer(tokens, answer_span)
            # Write this line as <question>\t<answer>
            sources.append(question)
            targets.append(answer)
            answers.append(answer)
            ids.append(q_id)
        id_targets[q_id] = answers
        count += 1

    return ids, sources, targets, id_targets


if __name__ == "__main__":
    pass
