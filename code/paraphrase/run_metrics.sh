# slurm
sbatch_params="
    -n 1
    -p GPU-DEPINFO,GPU
    -c 1
    --mem 16G
    --gres gpu:titanxt:1
    --exclude calcul-gpu-lahc-[2-3],calcul-bigcpu-lahc-[4-5]
    -t 7-00:00:00"  # calcul-gpu-depinfo-[1-2]

for file_in in result/*translation*.jsonl; do
    # avoid computing *-metrics-metrics.jsonl
    if [[ $file_in == *-metrics.jsonl ]]; then continue; fi
    # evaluate diversity if the metric file does not exists
    file_out="${file_in%.jsonl}-metrics.jsonl"  # in python: file_in.rstrip('.jsonl')+'-metrics.jsonl'
    if [[ ! -f $file_out ]]; then
        # run via slurm
        run_name="metrics_$(basename $file_in)"
        echo -n [slurm scheduler] metrics $file_in "-> "
        sbatch $sbatch_params -J $run_name -o logs/$run_name simple_bash.sh \
            ./evaluate_paraphrase_diversity.py --in_file $file_in --cached
    fi
done

# debug
echo [slurm scheduler] scheduled:
squeue -u $USER --format="%10T %11M %7A %65j %8p %4D %16b %3C %5m %11l %16P %60x"
echo

echo [slurm scheduler] tailing last log:
touch logs/$run_name  # suppress tail error saying file does not exists yet
tail -F logs/$run_name
