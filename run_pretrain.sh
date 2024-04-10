export INPUT_SIZE=224
export ACCUM_ITER=16
export BATCH_SIZE=64
export MASK_RATIO=0.75
export NB_EPOCHS=100
export DATA_DIR="/home/ella_understory_ai/deep/mae/imagenet/ILSVRC/Data/CLS-LOC/"
export JOB_DIR="/home/ella_understory_ai/deep/mae"

python main_pretrain.py \
    --log_dir ${JOB_DIR} \
    --batch_size ${BATCH_SIZE} \
    --epochs ${NB_EPOCHS} \
    --accum_iter ${ACCUM_ITER} \
    --model mae_vit_base_patch16 \
    --input_size ${INPUT_SIZE} \
    --mask_ratio ${MASK_RATIO} \
    --norm_pix_loss \
    --warmup_epochs 40 \
    --blr 1.5e-4 \
    --weight_decay 0.05 \
    --data_path ${DATA_DIR} \
    --dist_on_itp
    

#--nproc_per_node=4