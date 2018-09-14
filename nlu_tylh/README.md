# Joint model for spoken language understanding (slot filling & intent detection)

joint-nlu for zte tianyilinghang

## Setup

- Tensorflow 1.0
- Python 3.6

## Model

Hierarchical LSTM model

![model](model.jpg)

**slot_filling**: sequence classification (fixed slot value)

**intent_filling**: sequence classification

> Note: slots and intent don't have the "_UNK" value
>
> three intents: inform, confirm, reject
>
> This version contain raw yizhifu FAQ data.

## Code and Dataset

| file                | notes                                                        |
| ------------------- | ------------------------------------------------------------ |
| generate_data.py    | 数据预处理，将原始数据（/raw文件夹下的文件）处理成句子与标签分开的文件（/data） |
| data_utils.py       | 包括一些处理数据的函数，用于数据处理成模型所需的数据格式     |
| multi_task_model.py | 定义了一个类MultiTaskModel，包括step(), get_batch(), get_one()等用于模型训练的方法 |
| run_multi_task.py   | 完成LSTM模型的建模，train()函数用于训练模型                  |
| nlu.py              | 定义了NLU与其他模块的接口，包括用于加载已经训练好的模型的函数: load_nlu_model()以及接口函数nlu_interface() |
| /raw                | 原始数据集                                                   |
| /data               | 预处理后的数据集，以及模型训练过程中生成的词典文件           |
| /model              | 训练好的NLU模型                                              |

## Vocabulary

intent and slot value set

| intent                 | attr                                              | loc                                                          | name                                                         | ope               |
| ---------------------- | ------------------------------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ | ----------------- |
| confirm,inform, reject | NULL,VPN,上网电脑数量,优惠,在网时限,定义,用户群体 | NULL,北海,南宁,崇左,来宾,柳州,桂林,梧州,河池,玉林,百色,贵港,贺州,钦州防城港 | NULL,A6套餐,A8信息版套餐,A8光纤宽带套餐,A8光速版套餐,A8共享版光纤套餐,A8共享版套餐,A8自主团购版套餐,A8自主版套餐,A8自主版新融合套餐,A9行业版村级卫生室69元套餐,A9行业版村级卫生室89元套餐,A9行业版村级卫生室套餐,Xa/Xta+C套餐,企业邮箱 | NULL,办理,收费 |

## References

- Wen L, Wang X, Dong Z, et al. [Jointly Modeling Intent Identification and Slot Filling with Contextual and Hierarchical Information](http://tcci.ccf.org.cn/conference/2017/papers/1093.pdf)[J]. 2017.