from glob import glob
import json

from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns


results_by_metric = {}
skippped_metrics = set()
for result_file in glob("result/*-metrics.jsonl"):
    print(result_file)
    with open(result_file) as f:
        results = json.load(f)
    #print(results)
    for metric, value in results.items():
        if isinstance(value, list):
            skippped_metrics.add(metric)
            continue
        if metric not in results_by_metric:
            results_by_metric[metric] = {}
        expe_name = result_file.rstrip('-metrics.jsonl')
        results_by_metric[metric][expe_name] = value
print("skipped metric with several results:", skippped_metrics)

for metric in tqdm(results_by_metric):
    expe_names = list(results_by_metric[metric].keys())
    heights = [results_by_metric[metric][expe_name] for expe_name in expe_names]
    #print(expe_names, heights)
    plt.figure()
    plt.title(metric)
    sns.barplot(x=heights, y=expe_names)
    #plt.xticks(rotation=45)
    plt.savefig(f'result/metric/{metric}.png', bbox_inches="tight")
