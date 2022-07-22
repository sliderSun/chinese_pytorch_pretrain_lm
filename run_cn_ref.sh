#!/bin/bash
CUDA_VISIBLE_DEVICES=0,1 python run_chinese_ref.py \
--file_name=train_www_data.txt \
--ltp=small \
--bert=hfl/chinese-roberta-wwm-ext \
--save_path=train_www_data_ref.txt \
>> ref.log 2>&1 &
