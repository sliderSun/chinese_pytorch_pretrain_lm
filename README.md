# chinese_pytorch_pretrain_lm
# 基于pytorch的中文预训练语言模型的DAPT和TAPT
[![License](https://img.shields.io/badge/license-Apache%202-green.svg)](https://www.apache.org/licenses/LICENSE-2.0)
[![Build Status](https://travis-ci.org/xialonghua/kotmvp.svg?branch=master)](https://travis-ci.org/xialonghua/kotmvp) 

ACL2020 Best Paper有一篇论文提名奖，《Don’t Stop Pretraining: Adapt Language Models to Domains and Tasks》。这篇论文做了很多语言模型预训练的实验，系统的分析了语言模型预训练对子任务的效果提升情况。有几个主要结论：
* 在目标领域的数据集上继续预训练（DAPT）可以提升效果；目标领域的语料与RoBERTa的原始预训练语料越不相关，DAPT效果则提升更明显。

* 在具体任务的数据集上继续预训练（TAPT）可以十分“廉价”地提升效果。

* 结合二者（先进行DAPT，再进行TAPT）可以进一步提升效果。

* 如果能获取更多的、任务相关的无标注数据继续预训练（Curated-TAPT），效果则最佳。

* 如果无法获取更多的、任务相关的无标注数据，采取一种十分轻量化的简单数据选择策略，效果也会提升。

**———————————————————————————————————————————————🤗前方高能🤗————————————————————————————————————————————**

对GPT、GPT-2、GPT文本数据集上的语言建模库模型进行微调(或从头开始训练)。ALBERT, BERT, distillbert, RoBERTa, XLNet… GPT和GPT-2使用因果语言建模进行训练或微调  
(CLM)丢失，而ALBERT、BERT、DistilBERT和RoBERTa使用掩码语言建模(MLM)进行训练或微调的损失。 XLNet使用排列语言建模(PLM)，关于它们之间差异的更多信息[模型总结](https://huggingface.co/transformers/model_summary.html)

这里提供了两组脚本。 第一个利用了Trainer API。 第二个带' no_trainer '后缀的集合使用自定义训练循环并利用🤗Accelerate库。 这两个集合都使用🤗Datasets库。 如果需要对数据集进行额外处理，可以根据需要轻松定制它们。
**PS:** 旧版本的脚本 `run_language_modeling.py` 可以从 [这里](https://github.com/huggingface/transformers/blob/main/examples/legacy/run_language_modeling.py). 获取。

### GPT-2/GPT and causal language modeling
pre-training: causal language modeling

```bash
python run_clm.py \
    --model_name_or_path gpt2 \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-clm
```

这使用内置的HuggingFace“训练器”进行训练。 如果您想使用自定义的训练回路，您可以利用或修改' run_clm_no_trainer.py '脚本。 查看脚本以获得受支持的参数列表。 示例如下: 

```bash
python run_clm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path gpt2 \
    --output_dir /tmp/test-clm
```

### RoBERTa/BERT/DistilBERT and masked language modeling

pre-training: masked language modeling.

根据RoBERTa的论文，使用动态屏蔽而不是静态屏蔽。 因此，模型可能收敛稍微慢一些(过度拟合需要更多的epochs)


```bash
python run_mlm.py \
    --model_name_or_path roberta-base \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-mlm
```

如果您的数据集是每行一个样本组织的, 需要使用 `--line_by_line` 

这使用内置的HuggingFace“训练器”进行训练。 如果您想使用自定义的训练回路，您可以利用或修改' run_mlm_no_trainer.py '脚本。 查看脚本以获得受支持的参数列表。 示例如下: 

```bash
python run_mlm_no_trainer.py \
    --dataset_name wikitext \
    --dataset_config_name wikitext-2-raw-v1 \
    --model_name_or_path roberta-base \
    --output_dir /tmp/test-mlm
```

**PS:** 在TPU上，你应该使用标志 `--pad_to_max_length` 和 `--line_by_line` 确保所有的批次都有相同的长度.
 

### Whole word masking

首先使用LTP分词

```bash
export TRAIN_FILE=/path/to/train/file
export LTP_RESOURCE=/path/to/ltp/tokenizer
export BERT_RESOURCE=/path/to/bert/tokenizer
export SAVE_PATH=/path/to/data/ref.txt

python run_chinese_ref.py \
    --file_name=$TRAIN_FILE \
    --ltp=$LTP_RESOURCE \
    --bert=$BERT_RESOURCE \
    --save_path=$SAVE_PATH
```
然后执行训练脚本

```bash
export TRAIN_FILE=/path/to/train/file
export VALIDATION_FILE=/path/to/validation/file
export TRAIN_REF_FILE=/path/to/train/chinese_ref/file
export VALIDATION_REF_FILE=/path/to/validation/chinese_ref/file
export OUTPUT_DIR=/tmp/test-mlm-wwm

python run_mlm_wwm.py \
    --model_name_or_path roberta-base \
    --train_file $TRAIN_FILE \
    --validation_file $VALIDATION_FILE \
    --train_ref_file $TRAIN_REF_FILE \
    --validation_ref_file $VALIDATION_REF_FILE \
    --do_train \
    --do_eval \
    --output_dir $OUTPUT_DIR
```

### XLNet and permutation language modeling

XLNet使用不同的训练目标，即排列语言建模。 这是一种自回归方法，通过最大化输入的所有排列的期望可能性来学习双向上下文序列分解顺序。

使用 `--plm_probability` 用于排列语言建模的掩码token的长度与周围上下文长度的比值。

使用 `--max_span_length` 限制用于排列语言建模的屏蔽令牌的长度

```bash
python run_plm.py \
    --model_name_or_path=xlnet-base-cased \
    --train_file path_to_train_file \
    --validation_file path_to_validation_file \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --do_train \
    --do_eval \
    --output_dir /tmp/test-plm
```

如果您的数据集是每行一个样本组织的, 需要使用 `--line_by_line` 

**PS:** 在TPU上，你应该使用标志 `--pad_to_max_length` 和 `--line_by_line` 确保所有的批次都有相同的长度.


## Creating a model on the fly

从头开始训练模型时，配置值可以重写  `--config_overrides`:


```bash
python run_clm.py --model_type gpt2 --tokenizer_name gpt2 \ --config_overrides="n_embd=1024,n_head=16,n_layer=48,n_positions=102" \
[...]
```

此功能仅在 `run_clm.py`, `run_plm.py` 和 `run_mlm.py`.




