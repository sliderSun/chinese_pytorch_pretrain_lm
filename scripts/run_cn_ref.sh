#!/bin/bash

python run_chinese_ref.py \
--file_name=train.txt \
--ltp=small \
--bert=hfl/chinese-roberta-wwm-ext \
--save_path=train_ref.txt \
>> ref.log 2>&1 &
