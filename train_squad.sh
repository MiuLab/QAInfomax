export SQUAD_DIR=squad
export SQUAD_ADV=squad_adv

CUDA_VISIBLE_DEVICES=2 python3 run_squad.py \
  --bert_model bert-base-uncased \
  --do_train \
  --do_predict \
  --do_lower_case \
  --train_file $SQUAD_DIR/train-v1.1.json \
  --predict_file $SQUAD_ADV/sample1k-HCVerifyAll.json \
  --train_batch_size 5 \
  --learning_rate 3e-5 \
  --num_train_epochs 2.0 \
  --max_seq_length 384 \
  --doc_stride 128 \
  --output_dir squad_info_output \
  --seed 5555
