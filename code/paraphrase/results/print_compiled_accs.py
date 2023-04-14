import subprocess


cmd = 'python results/compile_accs.py {args} --verbose 0'
grid_search_xps_for_dataset = {
    'BANKING77[:1000]': ['{044..068}'],
    'HWU64[:1000]':     ['{147..171}'],
    'Liu[:1000]':       ['{172..196}'],
    'OOS[:1000]':       ['{197..221}'],
    '(BANKING77, HWU64)[:1000]':           ['{044..068}', '{147..171}'],
    '(BANKING77, HWU64, Liu)[:1000]':      ['{044..068}', '{147..196}'],
    '(BANKING77, HWU64, Liu, OOS)[:1000]': ['{044..068}', '{147..221}'],
}
args_for_xp_type = {
    'baselines': '../P2/jzxps/*{dataset}/protaugment-cv*-k1-low/logs/metrics.json',
    'grid search': 'xps/{xp}*/protaugment-cv*-k1-low/metrics.json -i',
}


def main():
    print('\n\n\n===== xps =====')
    for dataset, xps in grid_search_xps_for_dataset.items():
        if len(xps) == 1:
            print(f'{xps}: grid search 5x5 {dataset}')


    for dataset, xps in grid_search_xps_for_dataset.items():
        print(f'\n\n\n===== {dataset} =====')
        dataset = dataset.split('[')[0]
        for xp_type, args in args_for_xp_type.items():
            if len(xps) >= 1 and xp_type == 'baselines':
                continue
            path, *other_args = args.split()
            args = ' '.join([path.format(dataset=dataset, xp=xp) for xp in xps] + other_args)
            print(f'\n\n=== {xp_type} ===')
            print(cmd.format(args=args), end='\n\n')
            subprocess.run(cmd.format(args=args), check=True, shell=True)


if __name__ == '__main__':
    main()
