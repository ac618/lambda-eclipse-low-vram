export DATASET_PATH="/data/data/matt/learning-to-compose-unseen-concepts/GitHub_Releases/eclipse-inference/dreambench/dreambooth/dataset/backpack_dog"
export OUTPUT_DIR="./checkpoints/backpack_dog"
export CONCEPT="dog" # !!! Note: This is to check concept overfitting. This never supposed to generate your concept images.
export TRAINING_STEPS=400

python train_text_to_image_decoder.py \
        --instance_data_dir=$DATASET_PATH \
        --subject_data_dir=$DATASET_PATH \
        --output_dir=$OUTPUT_DIR \
        --validation_prompts="A $CONCEPT" \
        --resolution=768 \
        --train_batch_size=1 \
        --gradient_accumulation_steps=4 \
        --gradient_checkpointing \
        --max_train_steps=$TRAINING_STEPS \
        --learning_rate=1e-05 \
        --max_grad_norm=1 \
        --checkpoints_total_limit=4 \
        --lr_scheduler=constant \
        --lr_warmup_steps=0 \
        --report_to=wandb \
        --validation_epochs=100 \
        --checkpointing_steps=100 \
        --push_to_hub