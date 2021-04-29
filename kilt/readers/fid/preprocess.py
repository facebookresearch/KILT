import json
import sys


def convert_kilt(inputpath, outputpath):
    data = []
    inputdata=open(inputpath, 'r')
    for k, example in enumerate(inputdata):
        d = {}
        ex = json.loads(example)
        d['question'] = ex['input']
        answers = set()
        for a in ex['output']:
            if 'answer' in a:
                answers.add(a['answer'])
        d['answers'] = list(answers)
        d['id'] = ex['id']
        passages = []
        for c in ex['output'][0]['provenance']:
            p = {'title':c['wikipedia_title'],'text':c['text']}
            passages.append(p)
        d['ctxs']=passages
        data.append(d)
    with open(outputpath, 'w') as fout:
        json.dump(data, fout)

if __name__ == '__main__':
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    convert_kilt(inputpath, outputpath)
