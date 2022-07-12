#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1  python run_mlm.py \
--output_dir=output \
--model_type=bert  \
--model_name_or_path=hfl/chinese-roberta-wwm-ext \
--do_train \
--train_data_file=train.txt \
--do_eval \
--eval_data_file=eval.txt \
--line_by_line \
--per_device_train_batch_size=8 \
>> mlm.log 2>&1 &
