#!/bin/bash
set -e

MEGATRON_PATCH_PATH="/home/workspace/dataset/zrh/pai-megatron-patch/Pai-Megatron-Patch/"
MEGATRON_PATH=${MEGATRON_PATCH_PATH}/Megatron-LM-240405
export PYTHONPATH=${MEGATRON_PATH}:${MEGATRON_PATCH_PATH}:$PYTHONPATH
export CUDA_DEVICE_MAX_CONNECTIONS=1
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

MASTER_ADDR=$KAW_MASTER_ADDR
MASTER_PORT=$KAW_MASTER_PORT
# shellcheck disable=SC2071
if [[ "$KAW_NNODES" > 1 ]]; then
        NNODES=$KAW_NNODES
        NODE_RANK=$KAW_NODE_RANK
else
        export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
        MASTER_ADDR=localhost
        MASTER_PORT=$(shuf -n 1 -i 10000-65535)
        NNODES=1
        NODE_RANK=0
fi

GPUS_PER_NODE=8

DISTRIBUTED_ARGS="--nproc_per_node $GPUS_PER_NODE --nnodes $NNODES --node_rank $NODE_RANK --master_addr $MASTER_ADDR --master_port $MASTER_PORT"

# 模型结构参数量级
MODEL_SIZE=${MODEL_SIZE}

MODEL_ID=${MODEL_ID}
# 每卡训练一次迭代样本数
BATCH_SIZE=${MICRO_BATCH_SIZE}
# 全局batch size
GLOBAL_BATCH_SIZE=${GLOBAL_BATCH_SIZE}
# 学习率
LR=${LR}
# 最小学习率
MIN_LR=${MIN_LR}
# 序列长度
SEQ_LEN=${SEQ_LEN}
# 词表扩充大小
EXTRA_VOCAB_SIZE=256
# 训练精度
PR=bf16
# 模型并行度
TP=${TP}
# 流水并行度
PP=1
# 激活检查点模式
AC=sel
# 是否使用Megatron版Zero-1降显存优化器
DO=true
# 是否使用Flash Attention
FL=false
# 是否使用序列并行
SP=false
# 是否使用Transformer Engine
TE=false
MOE=false
# 保存ckpt的间隔
SAVE_INTERVAL=${SAVE_INTERVAL}
# 训练数据集路径
DATASET_PATH=${DATASET_PATH}
# 模型路径
MODEL_PATH=${MODEL_PATH}
# 训练token数
#TRAIN_TOKENS=${TRAIN_TOKENS}
# shellcheck disable=SC2125
TRAIN_TOKENS=$((${TRAIN_TOKENS_BY_BILLION}*1000000000))
# 预热token数
# shellcheck disable=SC2125
WARMUP_TOKENS=$((${TRAIN_TOKENS} * ${WARMUP_RATIO} /100))
OUTPUT_BASEPATH="/home/workspace/output/${MODEL_ID}"
# 训练输出文件路径
CHECKPOINT_PATH="${OUTPUT_BASEPATH}/checkpoint"
TENSORBOARD_DIR="${OUTPUT_BASEPATH}/tensorboard"
LOG_DIR="${OUTPUT_BASEPATH}/log"

mkdir -p "${CHECKPOINT_PATH}"
mkdir -p "${LOG_DIR}"
mkdir -p "${TENSORBOARD_DIR}"

if [ $MODEL_SIZE = 0.5B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=1024
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=2816
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

elif [ $MODEL_SIZE = 1.8B ]; then

NUM_LAYERS=24
HIDDEN_SIZE=2048
NUM_ATTN_HEADS=16
INTERMEDIATE_SIZE=5504
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

elif [ $MODEL_SIZE = 4B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=2560
NUM_ATTN_HEADS=20
INTERMEDIATE_SIZE=6912
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

elif [ $MODEL_SIZE = 7B ]; then

NUM_LAYERS=32
HIDDEN_SIZE=4096
NUM_ATTN_HEADS=32
INTERMEDIATE_SIZE=11008
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

elif [ $MODEL_SIZE = 14B ]; then

NUM_LAYERS=40
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=13696
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

elif [ $MODEL_SIZE = 32B ]; then

NUM_LAYERS=64
HIDDEN_SIZE=5120
NUM_ATTN_HEADS=40
INTERMEDIATE_SIZE=27392
MAX_POSITION_EMBEDDINGS=32768

gqa_options=" \
		    --group-query-attention \
		    --num-query-groups 8"

elif [ $MODEL_SIZE = 72B ]; then

NUM_LAYERS=80
HIDDEN_SIZE=8192
NUM_ATTN_HEADS=64
INTERMEDIATE_SIZE=24576
MAX_POSITION_EMBEDDINGS=32768
gqa_options=""

fi

if [ $AC = full ]; then
    activation_checkpoint_options=" \
		    --recompute-method uniform \
		    --recompute-granularity full"
elif [ $AC = sel ]; then
    activation_checkpoint_options=" \
        --recompute-activations"
elif [ $AC = none ]; then
    activation_checkpoint_options=" \
    "
fi

if [ $PR = fp16 ]; then
    pr_options=" \
		    --fp16"
elif [ $PR = bf16 ]; then
    pr_options=" \
        --bf16"
elif [ $PR = fp8 ]; then
    pr_options=" \
        --bf16 \
        --fp8-hybrid \
        --fp8-amax-compute-algo max \
        --fp8-amax-history-len 1024 \
        --transformer-impl transformer_engine"
fi

if [ $DO = true ]; then
    do_options=" \
		    --use-distributed-optimizer"

elif [ $DO = false ]; then
    do_options=" \
                    "
fi

if [ $FL = true ]; then
    flash_options=" \
		    --use-flash-attn"

elif [ $FL = false ]; then
    flash_options=" \
                    "
fi

if [ $TE = true ]; then
    te_options=" \
		    --transformer-impl transformer_engine"

elif [ $TE = false ]; then
    te_options=" \
        --transformer-impl local"
fi

if [ $MOE = true ]; then
    moe_options=" \
		    --moe-router-topk 2 \
		    --num-experts 8 \
		    --moe-aux-loss-coeff 1e-2 \
		    --expert-model-parallel-size 1 \
		    --moe-router-load-balancing-type aux_loss"

elif [ $MOE = false ]; then
    moe_options=" \
                    "
fi

if [ $SP = true ] && [ $TP -gt 1 ]; then
    sp_options=" \
		    --sequence-parallel"

elif [ $SP = false ]; then
    sp_options=" \
                    "
fi


GLOBAL_BATCH_SIZE=$(((${GPUS_PER_NODE}/${TENSOR_MODEL_PARALLEL_SIZE})*${MICRO_BATCH_SIZE}*${NNODES}*${GRADIENT_ACCUMULATINR_STEPS}))
TRAIN_ITERS=$((${TRAIN_TOKENS_BY_BILLION}*1000000000/(${GLOBAL_BATCH_SIZE}*2048)))
SAVE_INTERVAL=$((${TRAIN_ITERS}*${SAVE_TOKENS_BY_BILLION}/${TRAIN_TOKENS_BY_BILLION}))
LR_WARMUP_ITERS=$(( ${WARMUP_TOKENS}  / ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))
LR_DECAY_ITERS=$(( ${TRAIN_TOKENS} /  ${GLOBAL_BATCH_SIZE} / ${SEQ_LEN} ))

if [ "${CHECKPOINT_PATH}" != none ]; then
    if [ -f  "${CHECKPOINT_PATH}"/latest_checkpointed_iteration.txt ]; then
        echo "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt exist"
        MODEL_PATH=${CHECKPOINT_PATH}
    else
        echo "${CHECKPOINT_PATH}/latest_checkpointed_iteration.txt not exist"
    fi
fi

TRAIN_DATA_RECIPES_FILE=${TRAIN_DATASET_DIR}/${TRAIN_DATA_RECIPES_FILE}
VALID_DATA_RECIPES_FILE=${VALID_DATASET_DIR}/${VALID_DATA_RECIPES_FILE}
echo "---------Using data recipes: ${TRAIN_DATA_RECIPES_FILE}----------"
for line in `cat ${TRAIN_DATA_RECIPES_FILE}`
do
   line=`echo $line | sed "s#TRAIN_DATASET_DIR#${TRAIN_DATASET_DIR}#"`
   TRAIN_DATA_PATH="${TRAIN_DATA_PATH} ${line}"
done

echo "---------Using data recipes: ${VALID_DATA_RECIPES_FILE}----------"
for line in `cat ${VALID_DATA_RECIPES_FILE}`
do
   line=`echo $line | sed "s#VALID_DATASET_DIR#${VALID_DATASET_DIR}#"`
   VALID_DATA_PATH="${VALID_DATA_PATH} ${line}"
done

megatron_options="  \
        --save ${CHECKPOINT_PATH} \
        --train-data-path $TRAIN_DATA_PATH \
        --valid-data-path $VALID_DATA_PATH \
        --lr ${LR} \
        --min-lr ${MIN_LR} \
        --lr-decay-style cosine \
        --weight-decay 0.1 \
        --adam-beta1 0.9 \
        --adam-beta2 0.95 \
        --clip-grad 1.0 \
        --init-method-std 0.008 \
        --attention-dropout 0.0 \
        --hidden-dropout 0.0 \
        --lr-decay-iters ${LR_DECAY_ITERS} \
        --lr-warmup-iters ${LR_WARMUP_ITERS} \
        --train-iters ${TRAIN_ITERS} \
        --split 99,1,0 \
        --micro-batch-size ${BATCH_SIZE} \
        --global-batch-size ${GLOBAL_BATCH_SIZE} \
        --num-layers ${NUM_LAYERS} \
        --hidden-size ${HIDDEN_SIZE} \
        --num-attention-heads ${NUM_ATTN_HEADS} \
        --ffn-hidden-size ${INTERMEDIATE_SIZE} \
        --seq-length ${SEQ_LEN} \
        --max-position-embeddings ${MAX_POSITION_EMBEDDINGS} \
        --max-padding-length ${SEQ_LEN} \
        --log-interval 1 \
        --eval-interval 10000 \
        --eval-iters 10 \
        --save-interval ${SAVE_INTERVAL} \
        --tensorboard-queue-size 1 \
        --tensorboard-dir ${TENSORBOARD_DIR} \
        --log-timers-to-tensorboard \
        --log-batch-size-to-tensorboard \
        --log-validation-ppl-to-tensorboard \
        --tensor-model-parallel-size ${TP} \
        --pipeline-model-parallel-size ${PP} \
        --no-load-optim \
        --no-load-rng \
        --num-workers 8 \
        --extra-vocab-size ${EXTRA_VOCAB_SIZE} \
        --patch-tokenizer-type Qwen2Tokenizer \
        --dataset LLama-Pretrain-Idxmap \
        --swiglu \
        --normalization RMSNorm \
        --norm-epsilon 1e-06 \
        --use-rotary-position-embeddings \
        --no-rope-fusion \
        --position-embedding-type rope \
        --untie-embeddings-and-output-weights \
        --disable-bias-linear \
        --add-qkv-bias \
        --use-mcore-models \
        --rotary-percent 1.0 \
        --rotary-base 1000000 \
        --rotary-seq-len-interpolation-factor 1
        "

run_cmd="torchrun $DISTRIBUTED_ARGS pretrain_mcore_qwen.py
 ${megatron_options} ${pr_options} ${te_options} ${activation_checkpoint_options} ${do_options} ${flash_options} ${sp_options} ${gqa_options} ${moe_options}"

echo ${run_cmd}
eval ${run_cmd}
set +x
