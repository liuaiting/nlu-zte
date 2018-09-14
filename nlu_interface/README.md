Joint model for spoken language understanding (slot filling & intent detection)
=====================
nlu interface for zte (including yizhifu & tianyilinghang)

## Setup

* Tensorflow 1.0
* Python 3.6

## Model

Hierarchical LSTM model

![model](model.jpg)

**slot_filling**: sequence classification (fixed slot value)

**intent_filling**: sequence classification

> Note: slots and intent don't have the "_UNK" value
> three intents: inform, confirm, reject

## File

**/nlu_master/nlu_all.py**

```
提供NLU模块与外部的接口
```

**/nlu_master/nlu_tylh_master/nlu_data/**

```
天翼领航领域：NLU做inference需要的基础数据，包括词典、槽值集合以及意图集合
intent.txt 意图集合
s_attr.txt 属性槽槽值集合
s_loc.txt 地点槽槽值集合
s_name.txt 业务名称槽槽值集合
s_ope.txt 操作槽槽值集合
sent_vocab_500.txt 词典
```

**/nlu_master/nlu_tylh_master/nlu_model/**

```
天翼领航领域：NLU做inference时需要加载的已经训练好的模型
```

**/nlu_master/nlu_tylh_master/nlu_src/**   

```
data_utils.py 天翼领航领域：NLU模块对输入进行预处理源码
multi_task_model.py 天翼领航领域：构建模型部分的源码
nlu_tylh.py 天翼领航领域：与其他模块间接口
```

**/nlu_master/nlu_yzf_master/nlu_data/**

```
翼支付领域：NLU做inference需要的基础数据，包括词典、槽值集合以及意图集合
intent.txt 意图集合
s_attr.txt 属性槽槽值集合
s_loc.txt 地点槽槽值集合
s_name.txt 业务名称槽槽值集合
s_ope.txt 操作槽槽值集合
s_way.txt 渠道槽槽值集合
sent_vocab_500.txt 词典
```

**/nlu_master/nlu_yzf_master/nlu_model/**

```
翼支付领域：NLU做inference时需要加载的已经训练好的模型
```

**/nlu_master/nlu_yzf_master/nlu_src/**

```
data_utils.py 翼支付领域：NLU模块对输入进行预处理源码
multi_task_model.py 翼支付领域：构建模型部分的源码
nlu_tylh.py 翼支付领域：与其他模块间接口
```



## References

- Wen L, Wang X, Dong Z, et al. [Jointly Modeling Intent Identification and Slot Filling with Contextual and Hierarchical Information](http://tcci.ccf.org.cn/conference/2017/papers/1093.pdf)[J]. 2017.