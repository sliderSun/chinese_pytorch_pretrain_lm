#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1  python run_mlm.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--train_file train.txt \
--validation_file eval.txt \
--per_device_train_batch_size 8 \
--per_device_eval_batch_size 8 \
--do_train \
--do_eval \
--output_dir output \
--overwrite_output_dir \
>> mlm.log 2>&1 &
