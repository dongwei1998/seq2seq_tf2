# coding=utf-8
# =============================================
# @Time      : 2022-05-23 15:06
# @Author    : DongWei1998
# @FileName  : server.py
# @Software  : PyCharm
# =============================================
from flask import jsonify, request, Flask
import tensorflow as tf
from utils import parameter,seq2seq,data_help

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

if __name__ == '__main__':

    args = parameter.parser_opt('server')
    # 词表工具
    input_text_processor = data_help.load_text_processor(args.input_vocab, args.max_seq_length)
    output_text_processor = data_help.load_text_processor(args.output_vocab, args.max_seq_length)
    # 模型加载
    translator = seq2seq.MyModel(
        args,
        args.embedding_dim, args.units,
        input_text_processor=input_text_processor,
        output_text_processor=output_text_processor,
        use_tf_function=False)
    # 模型加载
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
    checkpoint.restore(tf.train.latest_checkpoint(args.output_dir))
    args.logger.info(f'Loading model from {args.output_dir}')

    app = Flask(__name__)
    app.config['JSON_AS_ASCII'] = False
    @app.route('/api/v1/translator', methods=['POST'])
    def predict():
        try:
            # 参数获取
            infos = request.get_json()
            data_dict = {
                'text': ''
            }
            for k, v in infos.items():
                data_dict[k] = v

            queries = data_dict['text'].replace('\n', '').replace('\r', '')
            # 参数检查
            if queries is None:
                return jsonify({
                    'code': 500,
                    'msg': '请给定参数text！！！'
                })
            # 直接调用预测的API
            three_input_text = tf.constant([queries])
            result = translator.text_predict(three_input_text)
            translation_text = result['text'][0].numpy().decode()
            return jsonify({
                'code': 200,
                'msg': '成功',
                'source_text ': queries,
                'translation_text':translation_text,
            })


        except Exception as e:
            # args.logger.error("异常信息为:{}".format(e))
            return jsonify({
                'code': 500,
                'msg': '预测数据失败!!!',
                'error':e
            })
    # 启动
    app.run(host='0.0.0.0',port=5557)