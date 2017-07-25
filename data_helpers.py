from __future__ import print_function
import numpy as np
import os
import tensorflow as tf
from tqdm import tqdm
import _pickle as cPickle
import re
import itertools
from collections import Counter
import numpy as np
import json


PAD = "_PAD"
UNK = "_UNK"
VALIDATION_SPLIT = 0.1


def Q2B(uchar):
  """全角转半角"""
  inside_code = ord(uchar)
  if inside_code == 0x3000:
    inside_code = 0x0020
  else:
    inside_code -= 0xfee0
  #转完之后不是半角字符返回原来的字符
  if inside_code < 0x0020 or inside_code > 0x7e:
    return uchar
  return chr(inside_code)


def replace_all(repls, text):
  # return re.sub('|'.join(repls.keys()), lambda k: repls[k.group(0)], text)
  return re.sub('|'.join(re.escape(key) for key in repls.keys()), lambda k: repls[k.group(0)], text)


def split_sentence(txt):
  sents = re.split(r'\n|\s|;|；|。|，|\.|,|\?|\!|｜|[=]{2,}|[.]{3,}|[─]{2,}|[\-]{2,}|~|、|╱|∥', txt)
  sents = [c for s in sents for c in re.split(r'([^%]+[\d,.]+%)', s)]
  sents = list(filter(None, sents))
  return sents


def normalize_punctuation(text):
  cpun = [['	'],
          ['﹗', '！'],
          ['“', '゛', '〃', '′', '＂'],
          ['”'],
          ['´', '‘', '’'],
          ['；', '﹔'],
          ['《', '〈', '＜'],
          ['》', '〉', '＞'],
          ['﹑'],
          ['【', '『', '〔', '﹝', '｢', '﹁'],
          ['】', '』', '〕', '﹞', '｣', '﹂'],
          ['（', '「'],
          ['）', '」'],
          ['﹖', '？'],
          ['︰', '﹕', '：'],
          ['・', '．', '·', '‧', '°'],
          ['●', '○', '▲', '◎', '◇', '■', '□', '※', '◆'],
          ['〜', '～', '∼'],
          ['︱', '│', '┼'],
          ['╱'],
          ['╲'],
          ['—', 'ー', '―', '‐', '−', '─', '﹣', '–', 'ㄧ', '－']]
  epun = [' ', '!', '"', '"', '\'', ';', '<', '>', '、', '[', ']', '(', ')', '?', ':', '･', '•', '~', '|', '/', '\\', '-']
  repls = {}

  for i in range(len(cpun)):
    for j in range(len(cpun[i])):
      repls[cpun[i][j]] = epun[i]

  return replace_all(repls, text)


def clean_str(txt):
  # txt = txt.replace('説', '說')
  # txt = txt.replace('閲', '閱')
  # txt = txt.replace('脱', '脫')
  # txt = txt.replace('蜕', '蛻')
  # txt = txt.replace('户', '戶')
  # 臺
  txt = txt.replace('臺', '台')
  txt = txt.replace('　', '') # \u3000
  txt = normalize_punctuation(txt)
  txt = ''.join([Q2B(c) for c in list(txt)])
  return txt


def build_vocab(sentences):
  """
  Builds a vocabulary mapping from word to index based on the sentences.
  Returns vocabulary mapping and inverse vocabulary mapping.
  """
  # Build vocabulary
  word_counts = Counter(itertools.chain(*sentences))
  # Mapping from index to word
  vocabulary_inv = [x[0] for x in word_counts.most_common()]
  # Mapping from word to index
  vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}
  return [vocabulary, vocabulary_inv]


def get_vocab(path='./data/vocab.pkl'):
  """Loads the vocab file, if present"""
  if not os.path.exists(path) or os.path.isdir(path):
    raise ValueError('No file at {}'.format(path))

  char_list = cPickle.load(open(path, 'rb'))
  vocab = dict(zip(char_list, range(len(char_list))))

  return vocab, char_list


def build_dataset(json_path='question_labels.json', 
                  data_dir='./data', max_doc_len=30, max_sent_len=50, ):
  q_dict = {'NUMBER': 0, 'PERSON': 1, 'LOCATION': 2, 'ORGANIZATION': 3, 'ARTIFACT': 4, 'TIME': 5, \
  'PROCEDURE': 6, 'AFFIRMATION': 7, 'CAUSALITY': 8}

  with open(os.path.join(data_dir, json_path), 'r') as f:
    question_labels = json.load(f)

  q_zh = []
  q_type = []
  for line in question_labels:
    q_zh.append(line['q_zh'])
    q_type.append(q_dict[line['q_type']])


  vocab, _ = get_vocab('./data/vocab.pkl')
  perm = np.random.permutation(len(q_zh))
  idx_train = perm[:int(len(q_zh)*(1-VALIDATION_SPLIT))]
  idx_val = perm[int(len(q_zh)*(1-VALIDATION_SPLIT)):]

  train_path = os.path.join(data_dir, 'train.tfrecords')
  valid_path = os.path.join(data_dir, 'valid.tfrecords')

  def upsampling(x, size):
    if len(x) > size:
      return x
    diff_size = size - len(x)
    return x + list(np.random.choice(x, diff_size, replace=False))


  def write_data(doc, label, out_f):
    doc = split_sentence(clean_str(doc))
    document_length = len(doc)
    sentence_lengths = np.zeros((max_doc_len,), dtype=np.int64)
    data = np.ones((max_doc_len * max_sent_len,), dtype=np.int64)
    doc_len = min(document_length, max_doc_len)

    for j in range(doc_len):
      sent = doc[j]
      actual_len = len(sent)
      pos = j * max_sent_len
      sent_len = min(actual_len, max_sent_len)
      # sentence_lengths
      sentence_lengths[j] = sent_len
      # dataset
      data[pos:pos+sent_len] = [vocab.get(sent[k], 0) for k in range(sent_len)]

    features = {'sentence_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=sentence_lengths)),
                'document_lengths': tf.train.Feature(int64_list=tf.train.Int64List(value=[doc_len])),
                'label': tf.train.Feature(int64_list=tf.train.Int64List(value=[label])),
                'text': tf.train.Feature(int64_list=tf.train.Int64List(value=data))}
    example = tf.train.Example(features=tf.train.Features(feature=features))
    out_f.write(example.SerializeToString())

  # oversampling
  with tf.python_io.TFRecordWriter(train_path) as out_f:
    train_size = len(idx_train)

    train_docs = [q_zh[i] for i in idx_train]
    train_type = [q_type[i] for i in idx_train]
    
    for i in tqdm(range(train_size)):
      row = train_docs[i]
      write_data(row, train_type[i], out_f)

  with tf.python_io.TFRecordWriter(valid_path) as out_f:
    valid_size = len(idx_val)
    valid_docs = [q_zh[i] for i in idx_val]
    valid_type = [q_type[i] for i in idx_val]
    for i in tqdm(range(valid_size)):
      row = valid_docs[i]
      write_data(row, valid_type[i], out_f)

  print('Done train {}, valid {}'.format(train_size, valid_size))


if __name__ == '__main__':
  build_dataset()
