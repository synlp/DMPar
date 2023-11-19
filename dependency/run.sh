python dep_main.py --do_train --train_data_path=./data/train.conllu --dev_data_path=./data/dev.conllu --use_bert --bert_model=/path/to/bert --n_mlp_arc=500 --n_mlp_rel=100 --max_seq_length=300 --train_batch_size=16 --eval_batch_size=16 --use_biaffine --num_train_epochs=2 --warmup_proportion=0.1 --learning_rate=1e-5 --patient=100 --model_name=demo

