cd $WORK/NAG/P2

getuid() (
    flock 9
    oldid=$(<~/.uid)
    newid=$((oldid+1))
    echo $newid >&9
    echo $newid
) 9<>~/.uid


K=1
dataset=HWU64  # /!\ IMPORTANT /!\
paraphrases_dir=$(pwd)/../paraphrase/result
for cv in 01 02 03; do
# for paraphrase_file in ${paraphrases_dir}/_out_dbs-uni-flat-1.0_${dataset}.jsonl ${paraphrases_dir}/_out_eda_${dataset}.jsonl ${paraphrases_dir}/_out_aeda_${dataset}.jsonl $paraphrases_dir/${dataset}-back-translations.jsonl ${paraphrases_dir}/${dataset}-back-translations-limit2000_3kw_5paraphrases-p0.5syn-p1shuf_checkpoint-65536-greedy.jsonl; do #${paraphrases_dir}/${dataset}-back-translations-limit2000_3kw_5paraphrases-p0.0syn-p1shuf_checkpoint-65536-greedy.jsonl ${paraphrases_dir}/${dataset}-back-translations-limit2000_3kw_5paraphrases-p0.5syn-p0.0shuf_checkpoint-65536-greedy.jsonl ${paraphrases_dir}/${dataset}-back-translations-limit2000_3kw_5paraphrases-p0.5syn-p1shuf_checkpoint-1-greedy.jsonl; do
for paraphrase_file in ../paraphrase/xps/{147..171}*/paraphrases.jsonl; do
    # limit sample to 2000 for now
    limit=2000
    if [ $(cat $paraphrase_file | wc -l) -gt $limit ] ; then
        new_file=$(dirname $paraphrase_file)/limit${limit}_$(basename $paraphrase_file)
        head -$limit $paraphrase_file > $new_file
        paraphrase_file=$new_file
    fi
    
#    xp_id=$(getuid)
#    xp_id=$(printf "%05d\n" $xp_id)  # 0-pad to 5 chars
#    run_name=${xp_id}_$(basename $paraphrase_file)
#    run_name=${run_name%.jsonl}  # remove extension
#    xp_dir=logs/$run_name
    xp_dir=$(dirname "$paraphrase_file")/protaugment-cv$cv-k$K-low
    run_name=$xp_dir
    
    if [[ ! -v first_xp_id ]]; then first_xp_id=$xp_id; fi
#    mkdir -p $xp_dir
#    ln -s $paraphrase_file ${xp_dir}/paraphrases.jsonl

    sbatch_params="--nodes=1 --ntasks-per-node=1 --cpus-per-task=10 -A yhg@v100 --gres=gpu:1 -t 0-20:00:00 --hint=nomultithread -J $run_name -o ${xp_dir}.log"
     
    low_regime="--train-path data/${dataset}/few_shot/${cv}/train.10samples.jsonl   --unlabeled-path data/${dataset}/raw.txt"

    echo -n [slurm scheduler] xp $run_name "-> "
    
    sbatch $sbatch_params models/proto/protaugment.sh   --data-path data/${dataset}/full.jsonl   --train-labels-path data/${dataset}/few_shot/${cv}/labels.train.txt   --valid-labels-path data/${dataset}/few_shot/${cv}/labels.valid.txt   --test-labels-path data/${dataset}/few_shot/${cv}/labels.test.txt   --unlabeled-path data/${dataset}/raw.txt   --n-support $K --n-query 5 --n-classes 5 --evaluate-every 100 --n-test-episodes 600   --max-iter 10000 --early-stop 8 --log-every 10 --seed 42 --n-unlabeled 5   --augmentation-data-path $paraphrase_file   --metric euclidean --supervised-loss-share-power 1   --model-name-or-path transformer_models/ProtAugment-LM-${dataset}   --output-path $xp_dir   $low_regime
done
done

squeue -u $USER --format="%.18i %.9P %.100j %.8u %.8T %.10M %.9l %.6D %R"

trap "
    echo
    echo [slurm scheduler] to check last line of every log:
    echo tail -n 1 \"xps/{$first_xp_id..$xp_id}*/protaugment.log\"
    echo
    echo [slurm scheduler] to tail last log:
    echo tail -F ${xp_dir}.log
" SIGINT SIGTERM

echo [slurm scheduler] tailing last log:
tail -F ${xp_dir}.log  2> /dev/null  # suppress stderr to ignore 'file does not exists yet' tail error
echo
