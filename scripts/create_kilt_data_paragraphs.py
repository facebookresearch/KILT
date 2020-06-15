import multiprocessing
from multiprocessing.pool import ThreadPool
import sys
import argparse
import pickle
import json

import spacy
from tqdm import tqdm, trange

import kilt.kilt_utils as utils
from kilt.knowledge_source import KnowledgeSource

CHUNK_SIZE = 100


def create_chunk(document, buffer, paragraph_id, paragraph, section):
    start = buffer[0].idx
    end = buffer[-1].idx + len(buffer[-1])

    anchors = [
        {
            "text": anchor["text"],
            "href": anchor["href"],
            "source": {
                "paragraph_id": anchor["paragraph_id"],
                "start": anchor["start"],
                "end": anchor["end"],
            },
            "start": anchor["start"] - start,
            "end": anchor["end"] - start,
        }
        for anchor in document["anchors"]
        if anchor["paragraph_id"] == paragraph_id
        and anchor["start"] >= start
        and anchor["end"] <= end
    ]

    return {
        "_id": document["_id"],
        "wikipedia_id": document["wikipedia_id"],
        "wikipedia_title": document["wikipedia_title"],
        "text": paragraph.text[start : end + 1].strip(),
        "tmp_len": len(buffer),
        "anchors": anchors,
        "categories": document["categories"],
        "history": document["history"],
        "sources": [{"paragraph_id": paragraph_id, "start": start, "end": end,}],
        "section": section,
    }


def run_thread(args):
    documents = args["documents"]
    nlp = args["nlp"]
    id = args["id"]
    rank = args["rank"]

    if id == 0 and rank == 0:
        iter_ = tqdm(documents)
    else:
        iter_ = documents

    # initialization
    output = []

    for document in iter_:

        # initialization
        buffer = []
        section = "Section::::Abstract"

        # loop paragrpahs removing first (title)
        for paragraph_id, paragraph in enumerate(nlp.pipe(document["text"][1:]), 1):

            # if section then save name and move on
            if "Section::::" in paragraph.text:
                section = paragraph.text.strip()
                continue

            for sentence in paragraph.sents:
                if buffer and len(buffer) + len(sentence) >= CHUNK_SIZE:
                    # create new chunk
                    new_chunk = create_chunk(
                        document, buffer, paragraph_id, paragraph, section
                    )
                    output.append(new_chunk)
                    buffer = []

                for token in sentence:
                    word = token.text.strip()
                    if word and len(word) > 0:
                        buffer.append(token)

            if buffer:
                # create new chunk
                new_chunk = create_chunk(
                    document, buffer, paragraph_id, paragraph, section
                )

                # conditions on merging with previous chunk
                if (
                    output
                    and document["wikipedia_id"] == output[-1]["wikipedia_id"]
                    and section == output[-1]["section"]
                    and len(buffer) + output[-1]["tmp_len"] < CHUNK_SIZE
                ):

                    # adjusting anchors offsets
                    for anchor in new_chunk["anchors"]:
                        anchor["start"] += len(output[-1]["text"]) + 1
                        anchor["end"] += len(output[-1]["text"]) + 1

                    # appending new data
                    output[-1]["text"] += " " + new_chunk["text"]
                    output[-1]["anchors"] += new_chunk["anchors"]
                    output[-1]["sources"] += new_chunk["sources"]
                    output[-1]["tmp_len"] += new_chunk["tmp_len"] + 1
                else:
                    output.append(new_chunk)
                buffer = []

    for out in output:
        del out["tmp_len"]
            
    return output


def store_chunks(documents, num_threads):
    for id, chunk in enumerate(utils.chunk_it(documents, num_threads)):
        out_filename = "/checkpoint/fabiopetroni/DPR_KILT_nicola/KILT_psgs_DPR/documents_{}.p".format(
            id
        )
        pickle.dump(chunk, open(out_filename, "wb"))


def load_chunk(id):
    in_filename = "/checkpoint/fabiopetroni/DPR_KILT_nicola/KILT_psgs_DPR/documents_{}.p".format(
        id
    )
    return pickle.load(open(in_filename, "rb"))


def load_all_documents_from_ks(cursor, steps, n):
    documents = []
    j = 0
    for document in cursor:
        if j % steps == 0:
            sys.stdout.write("{}/{} \r".format(j, n))
            sys.stdout.flush()
        documents.append(document)
        j += 1
    return documents


def preprocess_data(num_threads):
    
    ks = KnowledgeSource()
    n = ks.get_num_pages()
    steps = int(n / 100)

    cursor = ks.get_all_pages_cursor()

    print("LOADING ALL DOCUMENTS", flush=True)
    ducuments = load_all_documents_from_ks(cursor, steps, n)
    store_chunks(ducuments, num_threads)


def main(rank, num_threads):

    print("loading chunk {}".format(rank), flush=True)
    documents = load_chunk(rank)

    arguments = [
        {
            "rank": rank,
            "id": id,
            "documents": chunk,
            "nlp": spacy.load("en_core_web_sm"),
        }
        for id, chunk in enumerate(utils.chunk_it(documents, num_threads))
    ]

    print("starting {} threads in {}".format(num_threads, rank))
    pool = ThreadPool(num_threads)
    results = pool.map(run_thread, arguments)

    f = open(
        "/checkpoint/fabiopetroni/DPR_KILT_nicola/KILT_psgs/kilt_psgs_w100_{}.jsonl".format(
            rank
        ),
        "w+",
    )

    i = 1
    for output in results:
        for msg in output:
            f.write("{}\t{}\n".format(i, json.dumps(msg)))
            i += 1
    f.close()
    pool.terminate()
    pool.join()
    print("done {}".format(rank))

    
def merge_files(num_threads):

    f = open(
        "/checkpoint/fabiopetroni/DPR_KILT_nicola/KILT_psgs/kilt_w100.jsonl", "w+"
    )
    i = 1
    for rank in trange(num_threads):
        filename = "/checkpoint/fabiopetroni/DPR_KILT_nicola/KILT_psgs/kilt_psgs_w100_{}.jsonl".format(
            rank
        )
        print("reading {}".format(filename), flush=True)
        with open(filename, "r") as fin:
            lines = fin.readlines()
            for line in tqdm(lines):
                elements = line.split("\t")
                if len(elements) != 2:
                    print(
                        "ERROR: len(elements)!=2 -> {}".format(len(elements)),
                        flush=True,
                    )
                else:
                    f.write(
                        "{}\t{}\n".format(
                            i, elements[1].strip()
                        )
                    )
                    i += 1
    f.close()
    print("done")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--rank", default=None, type=int, help="rank in a distributed execution",
    )

    parser.add_argument(
        "--threads", default=None, type=int, help="number of threads",
    )

    args = parser.parse_args()
    
    if args.threads == None:
        args.threads = int(multiprocessing.cpu_count())

    # step 1
    # preprocess_data(num_threads=args.threads)

    # step 2
    # main(rank=args.rank, num_threads=args.threads)

    # step 3
    # merge_files(num_threads=args.threads)
