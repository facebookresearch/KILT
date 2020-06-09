import nltk
import spacy
import json
import os
import logging
import sys
import time
import string
import random


def normalize_answer(s):
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return remove_punc(lower(s))


def validate_datapoint(datapoint, logger):

    # input is a string
    if not isinstance(datapoint["input"], str):
        if logger:
            logger.warning(
                "[{}] input is not a string {}".format(
                    datapoint["id"], datapoint["input"]
                )
            )
        return False

    # output is not empty
    if len(datapoint["output"]) == 0:
        if logger:
            logger.warning("[{}] empty output".format(datapoint["id"]))
        return False

    for output in datapoint["output"]:
        # answer is a string
        if not isinstance(output["answer"], str):
            if logger:
                logger.warning(
                    "[{}] answer is not a string {}".format(
                        datapoint["id"], output["answer"]
                    )
                )
            return False

        # provenance is not empty
        # if len(output["provenance"]) == 0:
        #    if logger:
        #        logger.warning("[{}] empty provenance".format(datapoint["id"]))
        #    return False

        for provenance in output["provenance"]:
            # wikipedia_id is provided
            if not isinstance(provenance["wikipedia_id"], str):
                if logger:
                    logger.warning(
                        "[{}] wikipedia_id is not a string {}".format(
                            datapoint["id"], provenance["wikipedia_id"]
                        )
                    )
                return False

            # title is provided
            if not isinstance(provenance["title"], str):
                if logger:
                    logger.warning(
                        "[{}] title is not a string {}".format(
                            datapoint["id"], provenance["title"]
                        )
                    )
                return False

    return True


def load_data(filename):
    data = []
    with open(filename, "r") as fin:
        lines = fin.readlines()
        for line in lines:
            data.append(json.loads(line))
    return data


def get_bleu(candidate_tokens, gold_tokens):

    candidate_tokens = [x for x in candidate_tokens if len(x.strip()) > 0]
    gold_tokens = [x for x in gold_tokens if len(x.strip()) > 0]

    # The default BLEU calculates a score for up to
    # 4-grams using uniform weights (this is called BLEU-4)
    weights = (0.25, 0.25, 0.25, 0.25)

    if len(gold_tokens) < 4:
        # lower order ngrams
        weights = [1.0 / len(gold_tokens) for _ in range(len(gold_tokens))]

    BLEUscore = nltk.translate.bleu_score.sentence_bleu(
        [candidate_tokens], gold_tokens, weights=weights
    )
    return BLEUscore


# split a list in num parts evenly
def chunk_it(seq, num):
    assert num > 0
    chunk_len = len(seq) // num
    chunks = [seq[i * chunk_len : i * chunk_len + chunk_len] for i in range(num)]

    diff = len(seq) - chunk_len * num  # 0 <= diff < num
    for i in range(diff):
        chunks[i].append(seq[chunk_len * num + i])

    return chunks


def match_answer(
    answer,
    page,
    nlp=None,
    MAX_PARAGRAPH_CANDIDATE=3,
    debug=False,
    index_mapping=None,
    normalize_text=True,
    fast=False,
):
    if nlp == None:
        nlp = spacy.load("en_core_web_sm")

    original_answer = answer
    if normalize_text:
        answer = normalize_answer(answer)

    try:
        answer_tokens = [token.text for token in nlp(answer)]
    except Exception as e:
        print("Exception {}".format(e))
        return -1, -1, -1, -1

    if normalize_text:
        # Remove “characters with encodings larger than 3 bytes” using Python 3
        answer_tokens = [
            "".join(char for char in x if len(char.encode("utf-8")) < 3).lower()
            for x in answer_tokens
        ]

    found = False
    max_bleu = None
    start_token = None
    end_token = None
    paragraph_id = None

    # instead of scanning all lines, get the k with the higest intersection
    candidate_dict = {}
    tokenized_paragraphs = []
    tokenized_paragraphs_offset = []

    for idx, paragraph in enumerate(page["text"]):

        index = paragraph.find(answer)
        if index >= 0:
            assert paragraph[index : index + len(answer)] == answer
            return idx, index, index + len(original_answer), 1.0

        index = paragraph.find(original_answer)
        if index >= 0:
            assert paragraph[index : index + len(original_answer)] == original_answer
            return idx, index, index + len(original_answer), 1.0

        paragraph_tokens = []
        paragraph_offsets = []
        for token in nlp(paragraph):
            paragraph_tokens.append(token.text)
            # idx	int	The character offset of the token within the parent document.
            paragraph_offsets.append(token.idx)

        if normalize_text:
            # Remove “characters with encodings larger than 3 bytes” using Python 3
            paragraph_tokens = [
                normalize_answer(
                    "".join(char for char in x if len(char.encode("utf-8")) < 3)
                )
                for x in paragraph_tokens
            ]

        tokenized_paragraphs.append(paragraph_tokens)
        tokenized_paragraphs_offset.append(paragraph_offsets)

        # token intersection
        intersection = len(set(paragraph_tokens).intersection(set(answer_tokens)))

        if intersection == len(answer_tokens):
            # I found all the tokens, let me see if there is a perfect match
            ax = " ".join([x.strip() for x in answer_tokens if len(x.strip()) > 0])
            for w_start in range(len(paragraph_tokens)):
                token = paragraph_tokens[w_start]
                if token == answer_tokens[0]:
                    bx = " ".join(
                        [
                            x.strip()
                            for x in paragraph_tokens[w_start:]
                            if len(x.strip()) > 0
                        ]
                    )
                    if bx.startswith(ax):
                        for w_end in range(w_start, len(paragraph_tokens)):
                            token = paragraph_tokens[w_end]
                            if token == answer_tokens[-1]:
                                cx = " ".join(
                                    [
                                        x.strip()
                                        for x in paragraph_tokens[w_start : w_end + 1]
                                        if len(x.strip()) > 0
                                    ]
                                )
                                if ax == cx:
                                    start_character = paragraph_offsets[w_start]
                                    end_character = paragraph_offsets[w_end] + len(
                                        paragraph_tokens[w_end]
                                    )
                                    return idx, start_character, end_character, 1.0

        if intersection not in candidate_dict:
            candidate_dict[intersection] = []
        candidate_dict[intersection].append(idx)

    if fast:
        return -1, -1, -1, -1

    candidate_idx = []
    for key in sorted(candidate_dict.keys(), reverse=True):
        # if key > 0:  # if the intersection is not empty
        for idx in candidate_dict[key]:
            candidate_idx.append(idx)
        if len(candidate_idx) >= MAX_PARAGRAPH_CANDIDATE:
            break

    assert len(candidate_idx) > 0

    # hack to map to new knowledge source
    if index_mapping:
        new_candidate_idx = []
        for idx in candidate_idx:
            if idx not in index_mapping:
                new_candidate_idx.append(idx)
        candidate_idx = new_candidate_idx
        if len(candidate_idx) == 0:
            return -1, -1, -1, -1

    # then scan only the k candidates
    for idx in candidate_idx:

        paragraph_tokens = tokenized_paragraphs[idx]

        # perfect match
        for i in range(len(paragraph_tokens) - len(answer_tokens) + 1):
            if paragraph_tokens[i : i + len(answer_tokens)] == answer_tokens:
                found = True
                max_bleu = 1.0
                paragraph_id = idx
                start_token = i
                end_token = i + len(answer_tokens)
                break

        # fuzzy match
        if not found:

            # TODO: add span tollerance to speed up! Not sure about this
            # SPAN_TOLLERANCE = int(len(answer_tokens) / 2)

            for init in range(len(paragraph_tokens)):
                for end in range(init, len(paragraph_tokens)):
                    candidate = paragraph_tokens[init : end + 1]
                    BLEU = get_bleu(candidate, answer_tokens)

                    # if there is the same BLEU, the shortest answer should win
                    if (
                        not max_bleu
                        or BLEU > max_bleu
                        or (
                            BLEU == max_bleu
                            and end_token
                            and start_token
                            and (end + 1 - init) < (end_token - start_token)
                        )
                    ):
                        max_bleu = BLEU
                        paragraph_id = idx
                        start_token = init
                        end_token = end

                    if max_bleu == 1:
                        break
                if max_bleu == 1:
                    break
            if max_bleu == 1:
                break

    if debug:
        print("wikipedia_tile:", page["wikipedia_title"])
        print("bleu: {0:.2f}".format(max_bleu))
        print("paragraph_id:", paragraph_id)
        print("start_token_id:", start_token)
        print("end_token_id:", end_token)
        print("start_token:", tokenized_paragraphs[paragraph_id][start_token])
        print("end_token:", tokenized_paragraphs[paragraph_id][end_token])
        print(
            "TOKENIZED MATCH", tokenized_paragraphs[paragraph_id][start_token:end_token]
        )
        print("len(tokenized_paragraphs):", len(tokenized_paragraphs))
        print("len(tokenized_paragraphs_offset):", len(tokenized_paragraphs_offset))
        print("paragraph_tokens:", tokenized_paragraphs[paragraph_id])
        print("paragraph_offsets:", tokenized_paragraphs_offset[paragraph_id])
        print(
            "start_character:", tokenized_paragraphs_offset[paragraph_id][start_token]
        )
        print("end_character:", tokenized_paragraphs_offset[paragraph_id][end_token])

    paragraph_tokens = tokenized_paragraphs[paragraph_id]
    paragraph_offsets = tokenized_paragraphs_offset[paragraph_id]

    start_character = paragraph_offsets[start_token]
    end_character = paragraph_offsets[end_token] + len(paragraph_tokens[end_token])

    return paragraph_id, start_character, end_character, max_bleu


def init_logging(base_logdir, modelname, logger=None):

    # logging format
    # "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )

    log_directory = "{}/{}/".format(base_logdir, modelname)

    if logger == None:
        logger = logging.getLogger("KILT")

        logger.setLevel(logging.DEBUG)

        # console handler
        ch = logging.StreamHandler(sys.stdout)
        ch.setLevel(logging.DEBUG)
        ch.setFormatter(formatter)

        logger.addHandler(ch)

    else:
        # remove previous file handler
        logger.handlers.pop()

    os.makedirs(log_directory, exist_ok=True)

    # file handler
    fh = logging.FileHandler(str(log_directory) + "/info.log")
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)

    logger.addHandler(fh)

    logger.propagate = False
    logger.info("logging in {}".format(log_directory))
    return logger


def create_logdir_with_timestamp(base_logdir):
    timestr = time.strftime("%Y%m%d_%H%M%S")
    # create new directory
    log_directory = "{}/{}_{}/".format(base_logdir, timestr, random.randint(0, 1000))
    os.makedirs(log_directory)
    return log_directory


if __name__ == "__main__":
    from kilt.knowledge_source import KnowledgeSource

    ks = KnowledgeSource()
    # page = ks.get_page_by_id(10287141)
    # evidence = "While Chen Yi-hsiung's death was officially ruled a suicide, with both suicide notes (supposedly burned by his family) and a video of his wife apologizing for her husband's crime backing up this theory, opponents of president Chen disagree with this conclusion."

    page = ks.get_page_by_id(1101759)
    evidence = "a newsletter sent to an advertising firm 's customers"
    paragraph_id, start_character, end_character, max_bleu = match_answer(
        evidence, page
    )

    print(paragraph_id, start_character, end_character, max_bleu)

    print(max_bleu)

    span = page["text"][paragraph_id][start_character:end_character]
    print(span)
