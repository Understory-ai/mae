export NUMBER_OF_PROCESSES=4
export MASTER_ADDR="localhost"
export MASTER_PORT=12345
export WARM_UP=0
export INPUT_SIZE=64
export ACCUM_ITER=16
export BATCH_SIZE=64
export MASK_RATIO=0.75
export NB_EPOCHS=2
export DECAY=0.05
export DATA_DIR="/home/ella_understory_ai/deep/mae/doty_dataset"
export JOB_DIR="/home/ella_understory_ai/deep/mae"


mpirun -np ${NUMBER_OF_PROCESSES} \
    python main_pretrain.py \
        --log_dir ${JOB_DIR} \
        --batch_size ${BATCH_SIZE} \
        --epochs ${NB_EPOCHS} \
        --accum_iter ${ACCUM_ITER} \
        --model mae_vit_base_patch16 \
        --input_size ${INPUT_SIZE} \
        --mask_ratio ${MASK_RATIO} \
        --norm_pix_loss \
        --warmup_epochs ${WARM_UP} \
        --blr 1.5e-4 \
        --weight_decay ${DECAY} \
        --data_path ${DATA_DIR} \
        --dist_on_itp
