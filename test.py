#%%
import os
import pandas as pd
from flair.models import SequenceTagger
from flair.data import Sentence
from tqdm import tqdm
import blink.main_dense as main_dense
import blink.candidate_ranking.utils as utils
import argparse

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/python/SciTweetLMs')

data = pd.read_csv('annotations.tsv', sep='\t')

#%%
data['entities'] = [x.split(';')[:-1] if x != "null;" else [] for x in data['entities']]

data = data.explode(column="entities")
data = data.fillna('null:null:-3')

data['entity_mention'] = [x.split(':')[0] for x in data['entities']]
data['entity'] = [x.split(':')[1] for x in data['entities']]
data['entity_score'] = [x.split(':')[2] for x in data['entities']]

data['entity_filtered'] = [e if float(s) > -1 else None for e, s in zip(data['entities'], data['entity_score'])]

#%%
models_path = "../blink/models/" # the path where you stored the BLINK models
os.listdir(models_path)

config = {
    "test_entities": None,
    "test_mentions": None,
    "interactive": False,
    "top_k": 5,
    "biencoder_model": models_path+"biencoder_wiki_large.bin",
    "biencoder_config": models_path+"biencoder_wiki_large.json",
    #"entity_catalogue": models_path+"entity.jsonl",
    "entity_catalogue": models_path+"entity_small.jsonl",
    #"entity_encoding": models_path+"all_entities_large.t7",
    "entity_encoding": models_path+"all_entities_small.t7",
    "crossencoder_model": models_path+"crossencoder_wiki_large.bin",
    "crossencoder_config": models_path+"crossencoder_wiki_large.json",
    #"faiss_index": "flat",
    #"index_path": "../blink/faiss_flat_index.pkl",
    "fast": False, # set this to be true if speed is a concern
    "output_path": "logs/" # logging directory
}
logger = utils.get_logger()
args = argparse.Namespace(**config)
#%%
models = main_dense.load_models(args, logger=logger)
#%%
sentences = data['text'].str.lower().to_list()

model = SequenceTagger.load("upos-fast")

mentions = []
for sent_idx, sent in tqdm(enumerate(sentences)):
    sent = Sentence(sent, use_tokenizer=True)
    sent.to_tagged_string()
    model.predict(sent)
    sent_mentions = sent.get_labels()
    for m in sent_mentions:
        if m.value == "NOUN" and m.score>0.99:
            mention = {'sent_idx': sent_idx, 'text': m.data_point.text, 'confidence': m.score,
                       'start_pos': m.data_point.start_position, "end_pos": m.data_point.end_position}
            mentions.append(mention)

ner_output_data = {"sentences": sentences, "mentions": mentions}

sentences = ner_output_data["sentences"]
mentions = ner_output_data["mentions"]
samples = []

for mention in tqdm(mentions):
    record = {}
    record["label"] = "unknown"
    record["label_id"] = -1
    # LOWERCASE EVERYTHING !
    record["context_left"] = sentences[mention["sent_idx"]][
        : mention["start_pos"]
    ].lower()
    record["context_right"] = sentences[mention["sent_idx"]][
        mention["end_pos"]:
    ].lower()
    record["mention"] = mention["text"].lower()
    record["start_pos"] = int(mention["start_pos"])
    record["end_pos"] = int(mention["end_pos"])
    record["sent_idx"] = mention["sent_idx"]
    samples.append(record)

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=samples)

#print(predictions)
#print(scores)

data_mentions = [list() for _ in range(len(data))]
data_predictions = [list() for _ in range(len(data))]
data_top_predictions = [list() for _ in range(len(data))]
data_scores = [list() for _ in range(len(data))]
data_top_scores = [list() for _ in range(len(data))]

for m, p, s in tqdm(zip(mentions, predictions, scores)):
    row_idx = m['sent_idx']
    data_mentions[row_idx].append(f"{m['text']};;{p[0]};;{s[0]}")
    #data_predictions[row_idx].append(p)
    #data_scores[row_idx].append(s)
    #data_top_predictions[row_idx].append(p[0])
    #data_top_scores[row_idx].append(s[0])

data['mentions'] = data_mentions
#data['entities_blink'] = data_predictions
#data['scores_blink'] = data_scores
#data['top_entity_blink'] = data_top_predictions
#data['top_score_blink'] = data_top_scores

data.to_csv('annotations_blink3.tsv', sep='\t', index=False)

data = data.sort_values(by=['cat1_final_answer', 'entity_score'], ascending=[False, False])
#%%
data = data.explode(column="mentions")
data = data.fillna('null;;null;;-100')

data['entity_mention'] = [x.split(';;')[0] for x in data['mentions']]
data['entity'] = [x.split(';;')[1] for x in data['mentions']]
data['entity_score'] = [float(x.split(';;')[2]) for x in data['mentions']]

data['entity_filtered'] = [e if float(s) > -1 else None for e, s in zip(data['entities'], data['entity_score'])]

#%%
data_to_link = [ {
                    "id": 0,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "I have ".lower(),
                    "mention": "anosmia".lower(),
                    "context_right": ", which means I lost my ability to smell".lower(),
                },
                {
                    "id": 1,
                    "label": "unknown",
                    "label_id": -1,
                    "context_left": "Why".lower(),
                    "mention": "Depression".lower(),
                    "context_right": " Is Underreported In Men http://t.co/wjMzvnOegD".lower(),
                }
                ]*20

_, _, _, _, _, predictions, scores, = main_dense.run(args, None, *models, test_data=data_to_link)


#%%
import json
from tqdm import tqdm

idx2remove = {'empty':[],
              'disambiguation':[],
              'city in':[],
              'country': [],
              'born': [],
              'mrt': []}

entities = []
idx2keep = []
lines2keep = []
with open(args.entity_catalogue, "r") as fin:
    lines = fin.readlines()
    for idx, line in tqdm(enumerate(lines), total=len(lines)):
        entity = json.loads(line)

        if str.isspace(entity['text']):
            idx2remove['empty'].append(idx)

        elif "disambiguation" in entity['title']:
            idx2remove['disambiguation'].append(idx)

        elif "city in" in entity['text']:
            idx2remove['city in'].append(idx)

        elif "country" in entity['text']:
            idx2remove['country'].append(idx)

        elif " born" in entity['text']:
            idx2remove['born'].append(idx)

        elif "may refer to" in entity['text']:
            idx2remove['mrt'].append(idx)

        elif idx%3==0:
            entities.append(entity)
            idx2keep.append(idx)
            lines2keep.append(line)

#%%
import torch
enc = torch.load(args.entity_encoding)
enc2 = enc[idx2keep,:]
torch.save(enc2, models_path+"all_entities_small.t7")

with open(models_path+"entity_small.jsonl", "w") as fout:
    fout.writelines(lines2keep)


#%%
import os
import pandas as pd

if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/python/SciTweetLMs')

data = pd.read_csv("2122.tsv", sep='\t', header=None)

data2 = pd.read_csv('/home/schellsn/python/sci_tweets2122.tsv', sep='\t', header=None)
data2.columns = ['tweetid', 'user', 'date', 'x1', 'x2', 'x3', 'x4', 'text', 'urls', 'response_tweetid']
data2['fake'] = data2['text'].str.contains('fake')
data2['nottrue'] = data2['text'].str.contains('not true')
data2['misleading'] = data2['text'].str.contains('misleading')
data2['false'] = data2['text'].str.contains('false')
data2['wrong'] = data2['text'].str.contains('wrong')
data2['fake'].value_counts()
data2['misleading'].value_counts()
data2['nottrue'].value_counts(normalize=True)
data2['false'].value_counts(normalize=False)
data2['wrong'].value_counts(normalize=False)

data2['candidates'] = data2[['fake', 'nottrue', 'misleading', 'false', 'wrong']].any(axis=1)
data2['candidates'].value_counts(normalize=True)


#%%
import os
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()
if 'SciTweetLMs' not in os.getcwd():
    os.chdir('/home/schellsn/python/SciTweetLMs')

data = pd.read_csv('tweets_newsoutlets_w_html.tsv', sep='\t')
del data['htmls']
#%%
from newsplease.crawler.simple_crawler import SimpleCrawler

article_html = SimpleCrawler.fetch_url('https://www.nytimes.com/2017/02/23/us/politics/cpac-stephen-bannon-reince-priebus.html?hp')

#%%
def download_websites(tweets):
    tweets = tweets.sample(n=len(tweets), replace=False)
    assert 'processed_urls' in tweets.columns, "DataFrame needs column 'processed urls' that contains list of resolved urls"

    if type(tweets['processed_urls'].iloc[0]) == str:
        tweets['processed_urls'] = tweets['processed_urls'].apply(lambda urls: ast.literal_eval(urls))

    tweets['htmls'] = tweets[['processed_urls']].progress_apply(lambda urls: SimpleCrawler.fetch_urls(*urls), axis=1)
    return tweets

data2 = data[:100]

data3 = download_websites(data2)

data3['htmls'][41]


#%%
import json
import os
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()

if 'python' not in os.getcwd():
    os.chdir('/home/schellsn/python/')

with open("data/UpdatedClaimReview/2022_01_14/claim_reviews.json") as f:
    cr = json.loads(f.read())
import json
import os
import pandas as pd
import ast
from tqdm import tqdm
tqdm.pandas()

if 'python' not in os.getcwd():
    os.chdir('/home/schellsn/python/')

with open("data/UpdatedClaimReview/2022_01_14/claim_reviews.json") as f:
    cr = json.loads(f.read())

with open("data/UpdatedClaimReview/2022_01_14/claim_reviews_raw.json") as f:
    crr = json.loads(f.read())

