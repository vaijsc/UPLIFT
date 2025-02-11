MODEL_NAME="./checkpoints/models--stabilityai--sd-turbo/snapshots/1681ed09e0cff58eeb41e878a49893228b78b94c"
TEACHER_MODEL_NAME="./checkpoints/stable-diffusion-2-1-base"

JOURNEYDB_DATA_DIR="./data/journeydb_trungdt21.txt"
LAION_DATA_DIR="./data/laion_prompt.txt"

OUTPUT_DIR="./training-runs/test_local_attention_v4"

BATCH_SIZE=32

NUM_GPUS=1

CUDA_VISIBLE_DEVICES=0 accelerate launch --mixed_precision="bf16" --num_processes ${NUM_GPUS} \
--main_process_port 30500 train_scripts/train_sb_v2_norandcfg.py \
	--student_model_name_or_path $MODEL_NAME \
	--pretrained_model_name_or_path $TEACHER_MODEL_NAME \
	--prompt_path $LAION_DATA_DIR \
	--train_data_dir $JOURNEYDB_DATA_DIR \
	--output_dir $OUTPUT_DIR \
	--resolution 512 \
	--validation_prompts "A racoon wearing formal clothes, wearing a tophat. Oil painting in the style of Rembrandt" "a zoomed out DSLR photo of a hippo biting through a watermelon" "a lanky tall alien on a romantic date at italian restaurant with a smiling woman, nice restaurant, photography, bokeh" \
	--train_batch_size $BATCH_SIZE \
	--gradient_accumulation_steps 1 --gradient_checkpointing \
	--set_grads_to_none \
	--learning_rate 1e-06 \
	--learning_rate_lora 1e-03 \
	--lr_scheduler "constant" --lr_warmup_steps 0 \
	--lora_rank 64 \
	--lora_alpha 128 \
	--checkpointing_steps 500 \
	--validation_steps 100 \
	--seed 0 \
	--adam_weight_decay=1e-4 \
	--allow_tf32 \
    --max_train_steps 100000 \
	--use_ema \
	# --clip_weight=0.1 \
	# --use_tinyvae \
	# --clip_shrink=2 \
	# --target_clip_score=0.37 \
	# --enable_xformers_memory_efficient_attention \