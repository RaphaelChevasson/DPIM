# utils
getuid() (
    flock 9
    oldid=$(<~/.uid)
    newid=$((oldid+1))
    echo $newid >&9
    echo $newid
) 9<>~/.uid

# slurm
sbatch_params="
    -n 1
    -p GPU-DEPINFO,GPU
    -c 1
    --mem 16G
    --gres gpu:titanxt:1
    --exclude calcul-gpu-lahc-[2-3],calcul-bigcpu-lahc-[4-5]
    -t 7-00:00:00"  # calcul-gpu-depinfo-[1-2]

# dataset
datasets="BANKING77"  # list, default: "BANKING77 HWU64 Liu OOS"
limit=1000

# kw augs
psyns="0.25 0.0"  # list, default: "1.0"
pshufs="1.0 0.75 0.5 0.25 0.0"  # list, default: "0.5"

# POINTER
checkpoints="65536"  # list, default: "1 2 4 8 16 32 64 128 256 512 1024 2048 4096 8192 16384 32768 65536"

# ProtAugment
cvs="1"  # list, default: "1 2 3 4 5"
ks="5"  # list, default: "1 5"
regimes="full"  # list, default: "10samp full"

for dataset in $datasets; do
for psyn in $psyns; do
for pshuf in $pshufs; do
for checkpoint in $checkpoints; do
pointer_model=../POINTER/ckpt/${dataset}/checkpoint-${checkpoint}       # finetuned on dataset
#pointer_model=../POINTER/ckpt/paraphrasing_model/checkpoint-${checkpoint}  # finetuned on paraphrasing
for cv in $cvs; do
for k in $ks; do
for regime in $regimes; do

    # dir for logs and i/o
    xp_id=$(getuid)
    xp_id=$(printf "%03d\n" $xp_id)  # 0-pad to 3 chars
    run_name=${xp_id}_aug_${dataset}-limit${limit}_${psyn}syn-${pshuf}shuf_ckpt-${checkpoint} # _r${regime}-k${k}-cv${cv}
    xp_dir=xps/$run_name
    mkdir -p $xp_dir

    # store first xpid
    if [[ ! -v first_xp_id ]]; then first_xp_id=$xp_id; fi
    
    # run pipeline
    echo -n [slurm scheduler] xp $run_name "-> "

    script_params="1 1 0
        ${dataset} --limit_dataset $limit 
        --p_synonyms $psyn --p_shuffles $pshuf 
        --pointer_model $pointer_model 
        --regimes $regime --K $k --CV $cv
        --log_path $xp_dir --out_path $xp_dir"
    
    sbatch $sbatch_params -J ${run_name} -o ${xp_dir}/slurm.log simple_bash.sh \
        python -u augmentation_pipeline.py ${script_params}

done
done
done
done
done
done
done

# debug
echo [slurm scheduler] scheduled:
squeue -u $USER --format="%10T %11M %7A %65j %8p %4D %16b %3C %5m %11l %16P %60x"
echo

trap "
    echo
    echo [slurm scheduler] to check last line of every log:
    echo tail -n 1 \"xps/{$first_xp_id..$xp_id}*/slurm.log\"
    echo
    echo [slurm scheduler] to tail last log:
    echo tail -F ${xp_dir}/slurm.log
" SIGINT SIGTERM

echo [slurm scheduler] tailing last log:
tail -F ${xp_dir}/slurm.log  2> /dev/null  # suppress stderr to ignore 'file does not exists yet' tail error
echo
