# coding=utf-8
# =============================================
# @Time      : 2022-05-19 11:02
# @Author    : DongWei1998
# @FileName  : train.py.py
# @Software  : PyCharm
# =============================================
import os
import time

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

from utils import parameter
from utils import data_help,seq2seq

import tensorflow as tf

class MaskedLoss(tf.keras.losses.Loss):
  def __init__(self):
    self.name = 'masked_loss'
    self.loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction='none')

  def __call__(self, y_true, y_pred):

    # Calculate the loss for each item in the batch.
    loss = self.loss(y_true, y_pred)

    # Mask off the losses on padding.
    mask = tf.cast(y_true != 0, tf.float32)
    loss *= mask

    # Return the total.
    return tf.reduce_sum(loss)


# 自适应学习率
class CustomSchedule(tf.keras.optimizers.schedules.LearningRateSchedule):
    def __init__(self, embedding_size, warmup_steps=4000):
        super(CustomSchedule, self).__init__()

        self.embedding_size = tf.cast(embedding_size, tf.float32)
        self.warmup_steps = warmup_steps

    def __call__(self, step):
        arg1 = tf.math.rsqrt(step)
        arg2 = step * (self.warmup_steps ** -1.5)
        return tf.math.rsqrt(self.embedding_size) * tf.math.minimum(arg1, arg2)

# 每个批次的平均loss
def _average(epoch_loss):
    return sum(epoch_loss) / len(epoch_loss)



def train_model(args):

    # 原始数据加载
    s_time = time.time()
    args.logger.info('Start loading train data ...')
    inp, targ = data_help.load_data(args.train_data_file)
    args.logger.info(f'Data loading is complete, inp len {len(inp)}, targ len {len(targ)}, which takes {(time.time()-s_time):5f} s')



    # 数据预处理 序列化工具加载
    if os.path.exists(args.input_vocab) and os.path.exists(args.output_vocab):
        args.logger.info('Loading text processor...')
        input_text_processor = data_help.load_text_processor(args.input_vocab,args.max_seq_length)
        output_text_processor = data_help.load_text_processor(args.output_vocab,args.max_seq_length)
    else:
        args.logger.info('Create text processor...')
        input_text_processor = data_help.create_text_processor(args.input_vocab,inp,args.max_seq_length)
        output_text_processor = data_help.create_text_processor(args.output_vocab,targ,args.max_seq_length)

    # 原数据据打乱 批次化
    dataset = tf.data.Dataset.from_tensor_slices((inp, targ))
    dataset = dataset.batch(args.batch_size)

    # 模型恢复


    # 模型构建
    translator = seq2seq.MyModel(
        args,
        args.embedding_dim, args.units,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
        use_tf_function=False
    )
    

    optimizer = tf.keras.optimizers.Adam(
            CustomSchedule(args.embedding_dim),
            beta_1=0.9, beta_2=0.98, epsilon=1e-9)
    # 模型优化器
    translator.compile(
        optimizer=optimizer,
        loss=MaskedLoss(),
    )
    # 模型保存器
    checkpoint = tf.train.Checkpoint(network=translator, optimizer=optimizer)
    translator.save_ckpt_model(args.output_dir, checkpoint, args.model_ckpt_name)
    # 模型恢复
    if tf.train.latest_checkpoint(args.output_dir)==None:
        args.logger.info(f'The new training model from {args.output_dir}')
    else:
        # 模型恢复
        checkpoint.restore(tf.train.latest_checkpoint(args.output_dir))
        args.logger.info(f'Loading model from {args.output_dir}')
    # 模型训练
    step = 0
    for epoch in range(args.num_epochs):
        loss_epoch = []
        for example_input_batch, example_target_batch in dataset:
            step += 1
            loss = translator.train_step([example_input_batch,example_target_batch])
            # 每个时间步的loss
            loss_epoch.append(loss['batch_loss'].numpy())
            args.logger.info('epoch {}, batch {}, loss:{:.4f}'.format(
                epoch + 1, step, loss['batch_loss'].numpy()
            ))
            # 模型保存
            if step % args.ckpt_model_num == 0:
                translator.ckpt_manager.save()
                args.logger.info('epoch {}, save model at {}'.format(
                    epoch + 1, args.output_dir
                ))
            if step % args.step_env_model == 0:
                env_model(args, translator)

        args.logger.info('epoch {}, batch {}, loss:{:.4f}'.format(
            epoch + 1, step, _average(loss_epoch)
        ))

def env_model(args,translator):
    # 数据加载
    args.logger.info('Start loading env data ...')
    inp, targ = data_help.load_data(args.text_data_file)
    # 原数据据打乱 批次化
    env_dataset = tf.data.Dataset.from_tensor_slices((inp, targ))
    env_dataset = env_dataset.batch(args.batch_size)
    loss_epoch = []
    step = 0
    for example_input_batch, example_target_batch in env_dataset:
        step += 1
        loss = translator.train_step([example_input_batch, example_target_batch])
        # 每个时间步的loss
        args.logger.info('env model batch {}, loss:{:.4f}'.format(
            step, loss['batch_loss'].numpy()))
        loss_epoch.append(loss['batch_loss'].numpy())
    args.logger.info('Env load loss:{:.4f}'.format(_average(loss_epoch)
    ))




if __name__ == '__main__':
    args = parameter.parser_opt('train')
    # 训练
    train_model(args)
