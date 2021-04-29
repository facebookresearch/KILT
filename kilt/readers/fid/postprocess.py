import sys
import json

def convert_to_kilt(inputpath, outputpath, datapath):
    data = []
    with open(datapath, 'r') as fin:
        for k, example in enumerate(fin):
            example = json.loads(example)
            data.append(example)
    datadict = {ex['id']:ex for ex in data}
    outfile = open(outputpath, 'w')    
    with open(inputpath, 'r') as f:
        lines = f.readlines()
        for line in lines:
            d = {}
            try:
                id, answer = line.split('\t')
            except ValueError:
                print('error')
            id = int(id)
            answer = answer.split('\n')[0]
            if id in d:
                print('key already in dict', d[id], answer)
            d['id'] = id

            wikipedia_ids = [
                {'wikipedia_id': p['wikipedia_id']} for p in datadict[str(id)]['output'][0]['provenance']
            ]
            wikipedia_ids = [{
                'answer': answer,
            }]
            d['output'] = wikipedia_ids

            json.dump(d, outfile)
            outfile.write('\n')
    

if __name__ == '__main__':
    inputpath = sys.argv[1]
    outputpath = sys.argv[2]
    datapath = sys.argv[3]
    convert_to_kilt(inputpath, outputpath, datapath)
