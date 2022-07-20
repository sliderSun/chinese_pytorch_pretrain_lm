#!/bin/bash

CUDA_VISIBLE_DEVICES=0,1  python run_mlm_www_v2.py \
--model_name_or_path hfl/chinese-roberta-wwm-ext \
--train_file train_www.txt \
--validation_file eval_www.txt \
--train_ref_file ref_train.txt \
--validation_ref_file ref_eval.txt \
--do_train \
--do_eval \
--output_dir output \
--line_by_line \
--overwrite_output_dir \
>> mlm_www.log 2>&1 &
