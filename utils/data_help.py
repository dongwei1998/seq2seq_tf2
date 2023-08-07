# coding=utf-8
# =============================================
# @Time      : 2022-05-19 15:09
# @Author    : DongWei1998
# @FileName  : data_help.py
# @Software  : PyCharm
# =============================================
import os

import jieba
import numpy as np
import tensorflow as tf
import tensorflow_text as tf_text
import json


# 数据加载
def load_data(path_to_file):
    with open(path_to_file, 'r', encoding='utf-8') as r:
        lines = r.readlines()
        pairs = [line.replace('\n','').split('\t') for line in lines]
        inp = [inp for targ, inp in pairs]
        targ = [targ for targ, inp in pairs]
    return inp, targ


# 编码规范化 数据预处理   西班牙语 英文
def tf_lower_and_split_punct(text):
    '''
    ¿Todavía está en casa?
    [START] ¿ todavia esta en casa ? [END]
    '''
    # # 分离重音字符
    # text = tf_text.normalize_utf8(text, 'NFKD')
    # text = tf.strings.lower(text)
    # 保留空格 a-z 标点符号.
    text = tf.strings.regex_replace(text, '[^ a-z.?!,¿]', '')
    # 标点符号周围添加空格.
    text = tf.strings.regex_replace(text, '[.?!,¿]', r' \0 ')
    # 分割添加开始结束标志.
    text = tf.strings.strip(text)
    text = tf.strings.join(['[START]', text, '[END]'], separator=' ')
    return text

# 序列化工具
def text_processor(max_seq_length,vocabulary=None):
    if vocabulary == None:
        return tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=tf_lower_and_split_punct, # 数据预处理函数，根据业务场景修改
            output_sequence_length=max_seq_length
        )
    return tf.keras.layers.experimental.preprocessing.TextVectorization(
            standardize=tf_lower_and_split_punct,
            vocabulary=vocabulary,
            output_sequence_length=max_seq_length
    )




def load_text_processor(vocab_file_path,max_seq_length):
    with open(vocab_file_path, 'r', encoding='utf-8') as ir:
        vocabulary = ir.read().split('\n')
    vocab_text_processor = text_processor(max_seq_length, vocabulary=vocabulary)
    return vocab_text_processor


def create_text_processor(vocab_file_path,vocab_list,max_seq_length):
    vocab_text_processor = text_processor(max_seq_length)
    vocab_text_processor.adapt(vocab_list)
    vocabulary = vocab_text_processor.get_vocabulary()
    with open(vocab_file_path, 'w', encoding='utf-8') as iw:
        iw.write('\n'.join(vocabulary))
    return vocab_text_processor


def jieba_conve(a,b):
    a_ = []
    b_ = []
    for word in jieba.cut(a):
        a_.append(word)
    for word in jieba.cut(b):
        b_.append(word)
    a_ = ' '.join(a_)
    b_ = ' '.join(b_)
    return '\t'.join([a_,b_])


def data_conve():
    path_file = '../preliminary_a_data_0713/preliminary_a_data/preliminary_train.json'
    # path_file = '../preliminary_a_data_0713/preliminary_a_data/preliminary_extend_train.json'
    # path_file = '../preliminary_a_data_0713/preliminary_a_data/preliminary_val.json'
    with open(path_file,'r',encoding='utf-8') as r:
        with open('../datasets/train.txt','w',encoding='utf-8') as w:
            for data_js in json.loads(r.read()):
                a = data_js['source']
                b = data_js['target']
                infos = jieba_conve(a, b)
                w.write(infos+'\n')

if __name__ == '__main__':
    data_conve()