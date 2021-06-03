BERT_BASE_DIR=cased_L-12_H-768_A-12

#if [ ! -d ${BERT_BASE_DIR} ]; then
	#wget https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip
	#unzip cased_L-12_H-768_A-12.zip
#fi

SEQ_LEN="64"
BS="64"
LR="2e-5"
EPOCHS="8"
cur_dir="/content/drive/MyDrive/projects/Capstone_Project/Checkpoint_5/GAN-BERT/Q1"
LABEL_RATE="0.05"

python -u /content/drive/MyDrive/projects/Capstone_Project/Checkpoint_5/GAN-BERT/Q1/ganbert.py \
        --task_name=general \
        --label_rate=${LABEL_RATE} \
        --do_train=true \
    	--do_test=true \
        --do_val=true \
        --do_predict=false \
    	--do_eval_predict=false \
    	--pred_OOS=false \
        --data_dir=${cur_dir} \
        --vocab_file=$BERT_BASE_DIR/vocab.txt \
        --bert_config_file=$BERT_BASE_DIR/bert_config.json \
        --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
        --max_seq_length=${SEQ_LEN} \
        --train_batch_size=${BS} \
        --learning_rate=${LR} \
        --num_train_epochs=${EPOCHS} \
        --warmup_proportion=0.1 \
        --do_lower_case=false \
        --output_dir=ganbert_output_model/train_Q1
