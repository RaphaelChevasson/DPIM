import subprocess

import print_compiled_accs as p


print('\n\n\n===== table 3 =====')
path_for_methods = {
    'EDA': '_eda_',
    'AEDA': '_aeda_',
    'RTT': '_translation',
    'BART-dbs': '_bart',
    'Ours': '_aug',
}
format='.3f'
print(*[rf'\textbf{{{col}}}' for col in ['Dataset'] + list(path_for_methods.keys())], sep=' & ', end=r'\\\n')
print(r'\midrule')
scores_test, scores_valid = [], []
for dataset, xps in  p.grid_search_xps_for_dataset.items():
    dataset = dataset.split('[')[0]
    for xp_type, args in p.args_for_xp_type.items():
        if len(xps) > 1:
            continue
        args = args.format(dataset=dataset, xp=xps[0])
        T, V = subprocess.run(p.cmd.format(args=args), check=True, shell=True)
        breakpoint()
        if xp_type == 'baselines':
            scores_test += T
        else:
            t = max(T, key=lambda k: sum(V[k]) / len(V[k]))  # grid search = test score for max valid score among xps
            scores_test.append(t)
        scores_for_method = {}
        for method, path in path_for_methods.items():
            st = [scores for (xp_path, scores) in scores_test if path in xp_path]
            assert len(st) >= 1, f"No scores found for grouped xp with path containing '{path}'"
            assert len(st) <= 1, f"More than one grouped xp matched with path containing '{path}'"
            scores = st[0]
            scores_for_method[method] = f'{np.mean(scores):{format}} ± {np.std(scores):{format}}'
        print(*scores_test.values(), sep=' & ', end=r'\\')

