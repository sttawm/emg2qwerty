  python train_remote.py \
    --spot \
    --experiment lstm_bidirectional_1x256+cosine_lr+no_dropout_resumed_v2 \
    checkpoint=gs://emg2qwerty-team-logs/logs/lstm_bidirectional_1x256+cosine_lr+no_dropout/checkpoints/last.ckpt
  
