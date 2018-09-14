# -*-encoding:utf-8 -*-
"""
Created on Tue 25 Apr 2017

@author: Aiting LIU
NLU interface for both yizhifu and tianyilinghang domain.
"""
import nlu_yzf_master.nlu_src.nlu_yzf as yzf_nlu
import nlu_tylh_master.nlu_src.nlu_tylh as tylh_nlu

import json
import tensorflow as tf

# Example.
sess = tf.Session()

# yizhifu domain
model1 = yzf_nlu.load_nlu_model(sess)
test1 = json.dumps({"request_data": {"input": ['是', '消', '费', '限', '额']},
                   "dr_result": {"domain": '翼支付'}})

result1 = yzf_nlu.nlu_interface(test1, sess, model1)
print('result', json.loads(result1))
test1 = json.dumps({"request_data": {"input": ['我', '想', '要', '交', '交', '通', '罚', '款']},
                   "dr_result": {"domain": '翼支付'}})

result1 = yzf_nlu.nlu_interface(test1, sess, model1)
print('result', json.loads(result1))

# #  tianyilinghang domain
model2 = tylh_nlu.load_nlu_model(sess)
test2 = json.dumps({"request_data": {"input": ['对', '于', '柳', '州', '用', '户', '来', '说', '，', '天','翼','领','航','A','8','共','享','版','光','纤','套','餐','如','何','办','理','？']},
                   "dr_result": {"domain": '天翼领航'}})

result2 = tylh_nlu.nlu_interface(test2, sess, model2)
print(json.loads(result2))

test2 = json.dumps({"request_data": {"input": ['不', '是']},
                   "dr_result": {"domain": '天翼领航'}})

result2 = tylh_nlu.nlu_interface(test2, sess, model2)
print(json.loads(result2))
