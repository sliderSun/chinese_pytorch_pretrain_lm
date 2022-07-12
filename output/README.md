---
tags:
- generated_from_trainer
metrics:
- accuracy
model-index:
- name: output
  results: []
---

<!-- This model card has been generated automatically according to the information the Trainer had access to. You
should probably proofread and complete it, then remove this comment. -->

# output

This model is a fine-tuned version of [uer/chinese_roberta_L-4_H-256](https://huggingface.co/uer/chinese_roberta_L-4_H-256) on an unknown dataset.
It achieves the following results on the evaluation set:
- Loss: 8.6543
- Accuracy: 0.025

## Model description

More information needed

## Intended uses & limitations

More information needed

## Training and evaluation data

More information needed

## Training procedure

### Training hyperparameters

The following hyperparameters were used during training:
- learning_rate: 5e-05
- train_batch_size: 8
- eval_batch_size: 8
- seed: 42
- optimizer: Adam with betas=(0.9,0.999) and epsilon=1e-08
- lr_scheduler_type: linear
- num_epochs: 3.0

### Training results



### Framework versions

- Transformers 4.20.1
- Pytorch 1.7.1+cpu
- Datasets 2.3.2
- Tokenizers 0.12.1
