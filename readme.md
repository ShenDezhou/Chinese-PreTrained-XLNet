[**中文说明**](./readme.md) | [**English**](./readme_en.md)

<p align="center">
    <img src="./pics/banner.svg" width="500"/>
</p>
<p align="center">
    <a href="https://github.com/shendezhou/Chinese-PreTrained-XLNet/blob/master/LICENSE">
        <img alt="LICENCE" src="" />
    </a>
</p>

本项目提供了面向中文的XLNet预训练模型，旨在丰富中文自然语言处理资源，提供多元化的中文预训练模型选择。
我们欢迎各位专家学者下载使用，并共同促进和发展中文资源建设。

本项目基于CMU/谷歌官方的XLNet：https://github.com/zihangdai/xlnet

其他相关资源：
- 中文XLNet预训练模型：https://github.com/ymcui/Chinese-XLNet

查看更多发布的资源：https://github.com/ 

## 新闻
**2021/2/4 所有模型已支持Pytorch和Tensorflow1以及Tensorflow2，请通过transformers库进行调用或下载。https://huggingface.co/**


<details>
<summary>历史新闻</summary>
2021/2/4 本目录发布的模型未来可接入[Huggingface-Transformers](https://github.com/huggingface/transformers)，查看[快速加载](#快速加载)

2021/2/4 `XLNet-tiny`已可下载，查看[模型下载](#模型下载)

2021/2/4 提供了在大规模通用语料（1.76GB）上训练的中文`XLNet-tiny`模型，查看[模型下载](#模型下载)
</details>

## 内容导引
| 章节 | 描述 |
|-|-|
| [模型下载](#模型下载) | 提供了中文预训练XLNet下载地址 |
| [基线系统效果](#基线系统效果) | 列举了部分基线系统效果 |
| [预训练细节](#预训练细节) | 预训练细节的相关描述 |
| [下游任务微调细节](#下游任务微调细节) | 下游任务微调细节的相关描述 |
| [FAQ](#faq) | 常见问题答疑 |
| [引用](#引用) | 本目录的技术报告 |

## 模型下载
* **`XLNet-tiny`**：6-layer, 768-hidden, 12-heads, 72M parameters(71766926)


| 模型简称 | 语料 | Google下载 | 百度云下载 |
| :------- | :--------- | :---------: | :---------: |
| **`XLNet-tiny, Chinese`** | **中文问答/<br/>通用数据<sup>[1]</sup>** | **[TensorFlow1](https://drive.google.com/drive/folders/1-4ZFSuVvgAEazcqnCwELQhBEKOszUTvn?usp=sharing)** <br/>**[TensorFlow2](https://drive.google.com/drive/folders/1-hzDQ9fKkhwqCFEH1TVMXEj_VN4mG_2b?usp=sharing)** <br/>**[PyTorch](https://drive.google.com/drive/folders/1-3RteqvOeyE3qvmRADq2P7ifYNHsO7Kt?usp=sharing)** | **[TensorFlow1,密码:tfxl](https://pan.baidu.com/s/1sUKsad2ZS6xQkUdxrj0qfw)** <br/>**[TensorFlow2,密码:tfxl](https://pan.baidu.com/s/1jzCUpx4VLYA8tbL_JIRllw)** <br/>**[PyTorch,密码:toxl](https://pan.baidu.com/s/1bdNtnz1Lts-24zhBtoxIRQ)** |

> [1] 通用数据包括：问答等数据，总大小1.74GB，记录数72万，字数983万。

### PyTorch/Tensorflow版本

提供PyTorch版本，TF1和TF2版本。

### 使用说明

中国大陆境内建议使用百度云下载点，境外用户建议使用谷歌下载点，`XLNet-tiny`模型文件大小约**343M**。 以TensorFlow版`XLNet-tiny, Chinese`为例，下载完毕后对zip文件进行解压得到：

```
tf_chinese_xlnet_tiny_L-6_H-768_A-12.zip
    |- checkpoint                                           # 存盘点信息
    |- xlnet_tiny_chinese.ckpt.data-00000-of-00001          # 模型权重
    |- xlnet_tiny_chinese.ckpt.index                        # 模型index信息
    |- xlnet_tiny_chinese.ckpt.data                         # 模型meta信息
    |- spiece.vocab          # 分词词表
    |- spiece.model          # 分词模型
```

TensorFlow2版本为：

```
tf2_chinese_xlnet_tiny_L-6_H-768_A-12.zip
    |- tf_model.h5           # 模型权重
    |- config.json           # 模型参数
    |- spiece.vocab          # 分词词表
    |- spiece.model          # 分词模型
```

Pytorch版本为：

```
chinese_xlnet_tiny_L-6_H-768_A-12.zip
    |- pytorch_model.bin     # 模型权重
    |- config.json           # 模型参数
    |- training_args.bin     # 模型训练信息
    |- spiece.vocab          # 分词词表
    |- spiece.model          # 分词模型
```


### 快速加载
依托于[Huggingface-Transformers 3.1.0](https://github.com/huggingface/transformers) ，可轻松调用以上模型。
```
tokenizer = AutoTokenizer.from_pretrained("MODEL_NAME")
model = AutoModel.from_pretrained("MODEL_NAME")
```
其中`MODEL_NAME`对应列表如下：  

| 模型名 | MODEL_NAME |
| - | - |
| XLNet-tiny-Chinese | /chinese-xlnet-tiny<sup>[1]</sup>|

> [1] 待上传,暂时需要手动下载。



## 基线系统效果
为了对比基线效果，我们在以下几个中文数据集上进行了测试。对比了中文BERT-wwm-ext、XLNet-base以及本项目的XLNet-tiny。
时间及精力有限，并未能覆盖更多类别的任务，请大家自行尝试。


### 简体中文分词：MSR 2005
**[MSR 2005数据集](http://aclweb.org/anthology/I05-3017)** 是MSR在2005年发布的中文分词数据集。
详细说明见Thomas Emerson. 2005. The second international chinese word segmentation bakeoff. In Proceedings of the fourth SIGHAN workshop on Chinese language Processing.
根据给定句子，模型需要给出适当的划分，使得有联合含义的字组合在一起。
评测指标为：Acc / F1

| 模型 | 开发集ACC/F1 | 验证ACC/F1 | 测试集 |
| :------- | :---------: | :---------: | :---------: |
| LSTM | 0.9526 /	0.94500|	0.940177 /	0.92627 |
| BERT-wwm-ext<sup>[1]</sup> | 0.96106 /	0.95476|	0.95565 /	0.9465  |
| **XLNet-tiny** | 0.9880 / 0.9863 |  **0.9679**  /    **0.96184** |
| **XLNet-base**<sup>[2]</sup> | 0.9988 /	0.99853 |	**0.9825** /	**0.97877**|

> [1] BERT-wwm-ext：是崔一鸣等人提出的[BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm) 。  
> [2] XLNet-base：是崔一鸣（哈工大讯飞联合实验室）等人提出的[XLNet](https://github.com/ymcui/Chinese-XLNet) 。


## 预训练细节
以下以`XLNet-tiny`模型为例，对预训练细节进行说明。

### 生成词表
按照XLNet官方教程步骤，首先需要使用[Sentence Piece](https://github.com/google/sentencepiece) 生成词表。
在本项目中，我们使用的词表大小为21128，其余参数采用官方示例中的默认配置。

```
SentencePieceTrainer.train(
    input=paths, 
    model_prefix='model/spbpe/spiece',  
    vocab_size=21_128, 
    user_defined_symbols=[]
)
```

### 预训练
获得以上数据后，正式开始预训练XLNet。
之所以叫`XLNet-tiny`是因为仅相比`XLNet-base`层数（12层减少到6层），词表数量由32000变为21128，其余参数没有变动，主要因为计算设备受限。
使用的命令如下：
```
    from transformers import XLNetConfig,XLNetTokenizer,XLNetLMHeadModel,LineByLineTextDataset,DataCollatorForPermutationLanguageModeling,Trainer, TrainingArguments
    
    config = XLNetConfig(
        vocab_size=21_128,
        d_model=768,
        n_head=12,
        n_layer=6,
    )

    tokenizer = XLNetTokenizer.from_pretrained("./model/spbpe", max_len=512)

    model = XLNetLMHeadModel(config=config)
    model.resize_token_embeddings(len(tokenizer))
    print(model.num_parameters())

    dataset = LineByLineTextDataset(
        tokenizer=tokenizer,
        file_path="./data/data_train.csv",
        block_size=128,
    )

    data_collator = DataCollatorForPermutationLanguageModeling(
        tokenizer=tokenizer, plm_probability=1.0/6, max_span_length=5
    )

    training_args = TrainingArguments(
        output_dir="./model/xlnet_v1",
        overwrite_output_dir=True,
        num_train_epochs=5,
        per_gpu_train_batch_size=32,
        save_steps=10_000,
        save_total_limit=2,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        data_collator=data_collator,
        train_dataset=dataset,
        prediction_loss_only=True,
    )

    trainer.train()

    if trainer.is_world_master():
        trainer.save_model("./model/spbpe")
```

## 下游任务微调细节
下游任务微调使用的设备是谷歌Cloud GPU（16G HBM），以下简要说明各任务精调时的配置。
**相关代码请查看[EXLNet](https://github.com/ShenDezhou/EXLNet) 项目。**


## FAQ
**Q: 会发布更大的模型吗？**  
A: 不一定，不保证。如果我们获得了显著性能提升，会考虑发布出来。

**Q: 在某些数据集上效果不好？**  
A: 选用其他模型或者在这个checkpoint上继续用你的数据做预训练。

**Q: 预训练数据会发布吗？**  
A: 抱歉，因为版权问题无法发布。

**Q: 训练XLNet花了多长时间？**  
A: `XLNet-tiny`使用了Cloud TPU v3 (128G HBM)训练了30K steps（batch=32），大约需要8小时时间。

**Q: 为什么XLNet官方没有发布Multilingual或者Chinese XLNet？**  
A: 
（以下是个人看法）不得而知，很多人留言表示希望有，戳[XLNet-issue-#3](https://github.com/zihangdai/xlnet/issues/3)。
以XLNet官方的技术和算力来说，训练一个这样的模型并非难事（multilingual版可能比较复杂，需要考虑各语种之间的平衡，也可以参考[multilingual-bert](https://github.com/google-research/bert/blob/master/multilingual.md) 中的描述。
**不过反过来想一下，作者们也并没有义务一定要这么做。** 
作为学者来说，他们的technical contribution已经足够，不发布出来也不应受到指责，呼吁大家理性对待别人的工作。

**Q: XLNet多数情况下比BERT要好吗？**  
A: 目前看来至少上述几个任务效果都还不错，虽然使用的数据和发布的[BERT-wwm-ext](https://github.com/ymcui/Chinese-BERT-wwm) 是不一样的。

**Q: ？**  
A: 。


## 引用
如果本目录中的内容对你的研究工作有所帮助，欢迎在论文中引用下述技术报告：
https://arxiv.org/abs/
```
TBD
```


## 致谢
项目作者： tsinghuaboy

建设该项目过程中参考了如下仓库，在这里表示感谢：
- XLNet: https://github.com/zihangdai/xlnet


## 免责声明
本项目并非[XLNet官方](https://github.com/zihangdai/xlnet) 发布的Chinese XLNet模型。
该项目中的内容仅供技术研究参考，不作为任何结论性依据。
使用者可以在许可证范围内任意使用该模型，但我们不对因使用该项目内容造成的直接或间接损失负责。


## 关注我们
欢迎关注知乎专栏号。

[学习兴趣小组](https://www.zhihu.com/column/thuil)


## 问题反馈 & 贡献
如有问题，请在GitHub Issue中提交。  
我们没有运营，鼓励网友互相帮助解决问题。  
如果发现实现上的问题或愿意共同建设该项目，请提交Pull Request。  
