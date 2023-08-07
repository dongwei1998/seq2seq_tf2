# coding=utf-8
# =============================================
# @Time      : 2022-05-19 15:09
# @Author    : DongWei1998
# @FileName  : seq2seq.py
# @Software  : PyCharm
# =============================================
import typing
from typing import Any, Tuple
import tensorflow as tf
import numpy as np





# 编码器
class Encoder(tf.keras.layers.Layer):
    def __init__(self, input_vocab_size, embedding_dim, enc_units):
        super(Encoder, self).__init__()
        self.enc_units = enc_units
        self.input_vocab_size = input_vocab_size
        self.embedding = tf.keras.layers.Embedding(self.input_vocab_size,
                                                   embedding_dim)
        self.gru = tf.keras.layers.GRU(self.enc_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

    def call(self, tokens, state=None):
        vectors = self.embedding(tokens)
        output, state = self.gru(vectors, initial_state=state)

        return output, state

# 注意力
class BahdanauAttention(tf.keras.layers.Layer):
    def __init__(self, units):
        super().__init__()
        # For Eqn. (4), the  Bahdanau attention
        self.W1 = tf.keras.layers.Dense(units, use_bias=False)
        self.W2 = tf.keras.layers.Dense(units, use_bias=False)

        self.attention = tf.keras.layers.AdditiveAttention()

    def call(self, query, value, mask):
        # shape_checker = ShapeChecker()
        # shape_checker(query, ('batch', 't', 'query_units'))
        # shape_checker(value, ('batch', 's', 'value_units'))
        # shape_checker(mask, ('batch', 's'))

        # From Eqn. (4), `W1@ht`.
        w1_query = self.W1(query)
        # shape_checker(w1_query, ('batch', 't', 'attn_units'))

        # From Eqn. (4), `W2@hs`.
        w2_key = self.W2(value)
        # shape_checker(w2_key, ('batch', 's', 'attn_units'))

        query_mask = tf.ones(tf.shape(query)[:-1], dtype=bool)
        value_mask = mask

        context_vector, attention_weights = self.attention(
            inputs=[w1_query, value, w2_key],
            mask=[query_mask, value_mask],
            return_attention_scores=True,
        )
        # shape_checker(context_vector, ('batch', 't', 'value_units'))
        # shape_checker(attention_weights, ('batch', 't', 's'))

        return context_vector, attention_weights

# 解码器
class DecoderInput(typing.NamedTuple):
    new_tokens: Any
    enc_output: Any
    mask: Any


class DecoderOutput(typing.NamedTuple):
    logits: Any
    attention_weights: Any

class Decoder(tf.keras.layers.Layer):
    def __init__(self, output_vocab_size, embedding_dim, dec_units):
        super(Decoder, self).__init__()
        self.dec_units = dec_units
        self.output_vocab_size = output_vocab_size
        self.embedding_dim = embedding_dim

        # For Step 1. The embedding layer convets token IDs to vectors
        self.embedding = tf.keras.layers.Embedding(self.output_vocab_size,
                                                   embedding_dim)

        # For Step 2. The RNN keeps track of what's been generated so far.
        self.gru = tf.keras.layers.GRU(self.dec_units,
                                       return_sequences=True,
                                       return_state=True,
                                       recurrent_initializer='glorot_uniform')

        # For step 3. The RNN output will be the query for the attention layer.
        self.attention = BahdanauAttention(self.dec_units)

        # For step 4. Eqn. (3): converting `ct` to `at`
        self.Wc = tf.keras.layers.Dense(dec_units, activation=tf.math.tanh,
                                        use_bias=False)

        # For step 5. This fully connected layer produces the logits for each
        # output token.
        self.fc = tf.keras.layers.Dense(self.output_vocab_size)

    def call(self,
             inputs: DecoderInput,
             state=None) -> Tuple[DecoderOutput, tf.Tensor]:

        # Step 1. Lookup the embeddings
        vectors = self.embedding(inputs.new_tokens)


        # Step 2. Process one step with the RNN
        rnn_output, state = self.gru(vectors, initial_state=state)



        # Step 3. Use the RNN output as the query for the attention over the
        # encoder output.
        context_vector, attention_weights = self.attention(
            query=rnn_output, value=inputs.enc_output, mask=inputs.mask)

        # Step 4. Eqn. (3): Join the context_vector and rnn_output
        #     [ct; ht] shape: (batch t, value_units + query_units)
        context_and_rnn_output = tf.concat([context_vector, rnn_output], axis=-1)

        # Step 4. Eqn. (3): `at = tanh(Wc@[ct; ht])`
        attention_vector = self.Wc(context_and_rnn_output)


        # Step 5. Generate logit predictions:
        logits = self.fc(attention_vector)


        return DecoderOutput(logits, attention_weights), state


class MyModel(tf.keras.Model):
    def __init__(self, args,embedding_dim, units,input_text_processor,output_text_processor, use_tf_function=True):
        super().__init__()
        # 参数
        self.args = args
        # 编码层
        encoder = Encoder(input_text_processor.vocabulary_size(),
                          embedding_dim, units)

        # 解码层
        decoder = Decoder(output_text_processor.vocabulary_size(),
                          embedding_dim, units)
        self.encoder = encoder
        self.decoder = decoder

        # 文本向量化类
        self.input_text_processor = input_text_processor
        self.output_text_processor = output_text_processor

        # 是否使用 @tf.function 方法 todo python版本问题，报错：待解决
        self.use_tf_function = use_tf_function

        self.output_token_string_from_index = (
            tf.keras.layers.experimental.preprocessing.StringLookup(
                vocabulary=output_text_processor.get_vocabulary(),
                mask_token='',
                invert=True))
        # The output should never generate padding, unknown, or start.
        index_from_string = tf.keras.layers.experimental.preprocessing.StringLookup(
            vocabulary=output_text_processor.get_vocabulary(), mask_token='')
        token_mask_ids = index_from_string(['', '[UNK]', '[START]']).numpy()

        token_mask = np.zeros([index_from_string.vocabulary_size()], dtype=np.bool)
        token_mask[np.array(token_mask_ids)] = True
        self.token_mask = token_mask

        self.start_token = index_from_string(tf.constant('[START]'))
        self.end_token = index_from_string(tf.constant('[END]'))


    def _preprocess(self, input_text, target_text):


        # Convert the text to token IDs
        input_tokens = self.input_text_processor(input_text)
        target_tokens = self.output_text_processor(target_text)

        # Convert IDs to masks.
        input_mask = input_tokens != 0
        target_mask = target_tokens != 0

        return input_tokens, input_mask, target_tokens, target_mask

    def _loop_step(self,new_tokens, input_mask,enc_output, dec_state):
        input_token, target_token = new_tokens[:, 0:1], new_tokens[:, 1:2]

        # Run the decoder one step.
        decoder_input = DecoderInput(new_tokens=input_token,
                                     enc_output=enc_output,
                                     mask=input_mask)

        # 解码
        dec_result, dec_state = self.decoder(decoder_input, state=dec_state)


        y_pred = dec_result.logits


        step_loss = self.loss(target_token, y_pred)

        return step_loss, dec_state

    # 训练
    def train_step(self, inputs):
        # 是否使用 @tf.function 方法
        if self.use_tf_function:
            return self._tf_train_step(inputs)
        else:
            return self._train_step(inputs)

    # tf加速   使用静态编译将函数内的代码转换成计算图
    @tf.function(input_signature=[[tf.TensorSpec(dtype=tf.string, shape=[None]), tf.TensorSpec(dtype=tf.string, shape=[None])]])
    def _tf_train_step(self, inputs):
        return self._train_step(inputs)


    def _train_step(self,inputs):

        input_text, target_text = inputs

        (input_tokens, input_mask,target_tokens, target_mask) = self._preprocess(input_text, target_text)

        max_target_length = tf.shape(target_tokens)[1]


        with tf.GradientTape() as tape:

            # Encode the input
            enc_output, enc_state = self.encoder(input_tokens)

            dec_state = enc_state
            loss = tf.constant(0.0)
            for t in tf.range(max_target_length - 1):
                # Pass in two tokens from the target sequence:
                # 1. The current input to the decoder. 编码器的输入
                # 2. The target for the decoder's next prediction. 第一个解码的目标
                new_tokens = target_tokens[:, t:t + 2]
                step_loss, dec_state = self._loop_step(new_tokens, input_mask,
                                                       enc_output, dec_state)
                loss += step_loss

                # Average the loss over all non padding tokens.
            average_loss = loss / tf.reduce_sum(tf.cast(target_mask, tf.float32))

        # Apply an optimization step
        variables = self.trainable_variables
        gradients = tape.gradient(average_loss, variables)
        self.optimizer.apply_gradients(zip(gradients, variables))
        # Return a dict mapping metric names to current value
        return {'batch_loss': average_loss}

    def save_ckpt_model(self,output_dir,checkpoint,model_ckpt_name):
        # 模型保存
        # ckpt管理器
        self.ckpt_manager = tf.train.CheckpointManager(
            checkpoint=checkpoint,
            directory=output_dir,
            max_to_keep=2,
            checkpoint_name=model_ckpt_name)


    def sample(self, logits, temperature):

        # Set the logits for all masked tokens to -inf, so they are never chosen.
        logits = tf.where(self.token_mask, -np.inf, logits)

        if temperature == 0.0:
            new_tokens = tf.argmax(logits, axis=-1)
        else:
            logits = tf.squeeze(logits, axis=1)
            new_tokens = tf.random.categorical(logits / temperature,
                                               num_samples=1)

        return new_tokens

    def tokens_to_text(self, result_tokens):
        result_text_tokens = self.output_token_string_from_index(result_tokens)
        result_text = tf.strings.reduce_join(result_text_tokens,
                                             axis=1, separator=' ')
        result_text = tf.strings.strip(result_text)
        return result_text

    def text_predict(self,input_text, *,
                           return_attention=True,
                           temperature=1.0):
        batch_size = input_text.shape[0]
        input_tokens = self.input_text_processor(input_text)
        enc_output, enc_state = self.encoder(input_tokens)

        dec_state = enc_state
        new_tokens = tf.fill([batch_size, 1], self.start_token)

        result_tokens = []
        attention = []
        done = tf.zeros([batch_size, 1], dtype=tf.bool)

        for _ in range(self.args.max_seq_length):

            dec_input = DecoderInput(new_tokens=new_tokens,
                                     enc_output=enc_output,
                                     mask=(input_tokens != 0))

            dec_result, dec_state = self.decoder(dec_input, state=dec_state)

            attention.append(dec_result.attention_weights)

            new_tokens = self.sample(dec_result.logits, temperature)

            # If a sequence produces an `end_token`, set it `done`
            done = done | (new_tokens == self.end_token)
            # Once a sequence is done it only produces 0-padding.
            new_tokens = tf.where(done, tf.constant(0, dtype=tf.int64), new_tokens)

            # Collect the generated tokens
            result_tokens.append(new_tokens)

            if tf.executing_eagerly() and tf.reduce_all(done):
                break

        # Convert the list of generates token ids to a list of strings.
        result_tokens = tf.concat(result_tokens, axis=-1)
        result_text = self.tokens_to_text(result_tokens)

        if return_attention:
            attention_stack = tf.concat(attention, axis=1)
            return {'text': result_text, 'attention': attention_stack}
        else:
            return {'text': result_text}






