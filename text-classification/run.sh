nohup  python3 run_glue.py \
--model_name_or_path uer/chinese_roberta_L-4_H-256 \
--train_file ./data/cf/train_cf.csv \
--validation_file ./data/cf/valid_cf.csv \
--do_train \
--do_eval \
--max_seq_length 128 \
--per_device_train_batch_size 32 \
--learning_rate 2e-5 \
--num_train_epochs 3 \
--output_dir ./tmp/ \
--overwrite_output_dir \
>> glue.log 2>&1 &