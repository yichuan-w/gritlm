cd /home/ubuntu/gritlm
export WANDB_NAME="gritlm-medi2-embedding"
torchrun --nproc_per_node 1 \
-m gritlm.training.run \
--output_dir ./my_embedding_model \
--model_name_or_path openaccess-ai-collective/tiny-mistral \
--train_data real_data/MEDI2 \
--learning_rate 2e-5 \
--warmup_ratio 0.03 \
--lr_scheduler_type linear \
--max_steps 10000 \
--per_device_train_batch_size 2 \
--gradient_accumulation_steps 4 \
--dataloader_drop_last True \
--normalized True \
--temperature 0.02 \
--query_max_len 256 \
--passage_max_len 512 \
--train_group_size 2 \
--mode embedding \
--attn cccc \
--logging_steps 100 \
--save_steps 5000 \
--bf16