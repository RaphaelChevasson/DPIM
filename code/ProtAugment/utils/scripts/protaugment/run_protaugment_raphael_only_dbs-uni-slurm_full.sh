#!/usr/bin/env bash
now=$(date "+%Y-%m-%dT%H-%M-%S")


paraphrase_tokenizer_name_or_path="facebook/bart-base"
max_time="7-00:00:00"

for cv in 01 ; do # 02 03 04 05; do
    for seed in 42; do
        for C in 5; do
            for K in 5; do
                for dataset in BANKING77; do
                    if [[ "${K}" == "5" ]] || [[ "${dataset}" == "20newsgroup" ]]; then
                        slurm_gpu_req="gpu:1"
                        slurm_partition="GPU,GPU-DEPINFO"
                    else
                        slurm_gpu_req="gpu:1"
                        slurm_partition="GPU,GPU-DEPINFO"
                    fi

                    # --------------------
                    #   Set SLURM params
                    # --------------------
                    sbatch_params="
                        -n 1
                        -p ${slurm_partition} \
                        -c 1 \
                        --mem=20G \
                        --gres=${slurm_gpu_req} \
                        --exclude=calcul-gpu-lahc-2 \
                        -t ${max_time}"

                    # --------------------
                    #   Set data params
                    # --------------------
                    data_params_full="
                        --data-path data/${dataset}/full.jsonl
                        --train-labels-path data/${dataset}/few_shot/${cv}/labels.train.txt
                        --valid-labels-path data/${dataset}/few_shot/${cv}/labels.valid.txt
                        --test-labels-path data/${dataset}/few_shot/${cv}/labels.test.txt
                        --unlabeled-path data/${dataset}/raw.txt"

                    few_shot_params="
                        --n-support ${K}
                        --n-query 5
                        --n-classes ${C}"

                    training_params="
                        --evaluate-every 100
                        --n-test-episodes 600
                        --max-iter 10000
                        --early-stop 20
                        --log-every 10
                        --seed 42"

                    backtranslation_params="
                        --n-unlabeled 5
                        --augmentation-data-path data/${dataset}/back-translations.jsonl"

                    model_params="
                        --metric euclidean
                        --supervised-loss-share-power 1
                        --model-name-or-path transformer_models/${dataset}/fine-tuned"

                    # .-----------------------------------------.
                    # | Full dataset, unigram, flat, p_mask 0.7 |
                    # .-----------------------------------------'
                    OUT_PATH="runs/full_datasets/${dataset}/${cv}/${C}C_${K}K/seed${seed}/DBS_unigram_flat_auc_0.7"
                    if [[ -d "${OUT_PATH}" ]]; then
                        echo "${OUT_PATH} already exists. Skipping."
                    else
                        mkdir -p ${OUT_PATH}
                        run_name="${OUT_PATH}"
                        LOGS_PATH="${OUT_PATH}/training.log"
                        sbatch ${sbatch_params} \
                            -J ${run_name} \
                            -o ${LOGS_PATH} \
                            models/proto/protaugment.sh \
                            $(echo ${data_params_full}) \
                            $(echo ${few_shot_params}) \
                            $(echo ${training_params}) \
                            $(echo ${paraphrase_params}) \
                            $(echo ${model_params}) \
                            --paraphrase-drop-strategy unigram \
                            --paraphrase-drop-chance-speed flat \
                            --paraphrase-drop-chance-auc 0.7 \
                            --output-path "${OUT_PATH}/output"
                    fi

                done
            done
        done
    done
done
