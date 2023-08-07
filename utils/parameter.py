# coding=utf-8
# =============================================
# @Time      : 2022-05-19 15:11
# @Author    : DongWei1998
# @FileName  : parameter.py
# @Software  : PyCharm
# =============================================
import os
from easydict import EasyDict
from dotenv import load_dotenv,find_dotenv
import logging.config
import shutil



# 创建路径
def check_directory(path, create=True):
    flag = os.path.exists(path)
    if not flag:
        if create:
            os.makedirs(path)
            flag = True
    return flag


def parser_opt(model):
    load_dotenv(find_dotenv())  # 将.env文件中的变量加载到环境变量中
    args = EasyDict()
    args.mode = os.environ.get("mode")
    logging.config.fileConfig(os.environ.get("logging_ini"))
    args.logger = logging.getLogger('model_log')
    # 清除模型以及可视化文件
    if model == 'train':
        args.network_name = os.environ.get('network_name')
        args.train_data_file = os.environ.get('train_data_file')
        args.text_data_file = os.environ.get('text_data_file')
        args.input_vocab = os.environ.get('input_vocab')
        args.output_vocab = os.environ.get('output_vocab')
        args.max_seq_length = int(os.environ.get('max_seq_length'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.embedding_dim = int(os.environ.get('embedding_dim'))
        args.units = int(os.environ.get('units'))
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.output_dir = os.environ.get('output_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.step_env_model = int(os.environ.get('step_env_model'))
    elif model =='env':
        args.network_name = os.environ.get('network_name')
        args.input_vocab = os.environ.get('input_vocab')
        args.output_vocab = os.environ.get('output_vocab')
        args.max_seq_length = int(os.environ.get('max_seq_length'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.embedding_dim = int(os.environ.get('embedding_dim'))
        args.units = int(os.environ.get('units'))
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.output_dir = os.environ.get('output_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.step_env_model = int(os.environ.get('step_env_model'))
    elif model == 'server':
        args.network_name = os.environ.get('network_name')
        args.input_vocab = os.environ.get('input_vocab')
        args.output_vocab = os.environ.get('output_vocab')
        args.max_seq_length = int(os.environ.get('max_seq_length'))
        args.batch_size = int(os.environ.get('batch_size'))
        args.embedding_dim = int(os.environ.get('embedding_dim'))
        args.units = int(os.environ.get('units'))
        args.num_epochs = int(os.environ.get('num_epochs'))
        args.ckpt_model_num = int(os.environ.get('ckpt_model_num'))
        args.output_dir = os.environ.get('output_dir')
        args.model_ckpt_name = os.environ.get('model_ckpt_name')
        args.step_env_model = int(os.environ.get('step_env_model'))
    else:
        raise print('请给定model参数，可选【traian env test】')
    return args


if __name__ == '__main__':
    args = parser_opt('train')