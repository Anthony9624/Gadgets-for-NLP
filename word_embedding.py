import modeling
import tokenization
import tensorflow as tf
from utils import  word_ids
import numpy as np
import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

class Config():
    max_seq_length = 30
    batch_size = 1
    bert_config = modeling.BertConfig.from_json_file('uncased_L-12_H-768_A-12/bert_config.json')
    init_checkpoint = 'uncased_L-12_H-768_A-12/bert_model.ckpt'
input_ids = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="input_ids")
input_mask = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="input_mask")
segment_ids = tf.placeholder(shape=[1, Config.max_seq_length], dtype=tf.int32, name="segment_ids")
# 创建bert 模型
model = modeling.BertModel(
    config=Config.bert_config,
    is_training=False,
    input_ids=input_ids,
    input_mask=input_mask,  # input_mask是样本中有效词句的标识
    token_type_ids=segment_ids,  # token_type是句子标记 ##
    use_one_hot_embeddings=False
)
embedding = model.get_sequence_output()  # 获取字向量

tvars = tf.trainable_variables()  #加载bert 参数
# 加载bert 模型参数
(assignment_map, initialized_variable_names) = \
    modeling.get_assignment_map_from_checkpoint(tvars,
                                                Config.init_checkpoint)
tf.train.init_from_checkpoint(Config.init_checkpoint, assignment_map)

session = tf.InteractiveSession()
session.run(())

# with open("PMID-1370299.txt","r") as f:
#     str = f.read()
#     print(str)
# path = os.getcwd()
# list_files = os.walk(path)
# for dirpath,dirnames,filenames in list_files:
#     for dir in dirnames:
#         print(dir)
#         for file in filenames:
#            print(file)
#         print("再见")



# os.path.dirname(__file__)
# path1 = os.path.dirname(__file__)
texts = ['无论走到哪里，都应该记住过去都是假的，回忆是一条没有尽头的路，一切以往的春天都不复存在，就连那最坚韧而又狂乱的爱情归根结底也不过是一种转瞬即逝的现实 ']#读取文件的形式，在此以一段文字来实现
tokenizer = tokenization.FullTokenizer(vocab_file='../uncased_L-12_H-768_A-12/vocab.txt', do_lower_case=True)
input_ids_list, input_mask_list, segment_ids_list = word_ids(texts, tokenizer, Config.max_seq_length)
input_ids_list = np.reshape(input_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
input_mask_list = np.reshape(input_mask_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
segment_ids_list = np.reshape(segment_ids_list, newshape=[-1, Config.batch_size, Config.max_seq_length])
embedding_r = session.run(embedding, feed_dict={input_ids: input_ids_list[0],
                                                   input_mask: input_mask_list[0],
                                                   segment_ids: segment_ids_list[0]})
print(type(embedding_r))
# print(embedding_r.shape)
#print(embedding_r[-1][-1])



# 也可以直接加载模型
# session.run(tf.global_variables_initializer())
# saver = tf.train.Saver()
# sess.run(tf.global_variables_initializer())
# saver.restore(sess, init_checkpoint)




