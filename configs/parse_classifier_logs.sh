import os
import pandas as pd

folder = 'D:/Project Results/Transfer Invariance/'

filenames = [
    'result.csv',
    # 'rda_stl10_src.csv',
    # 'rda_stl10_src_dt.csv',
    # 'rda_stl10_src_eqdt.csv',
]

for filename in filenames:
    result_path = os.path.join(folder, filename)

    col_names = ['prefix', 'std', 'rob', 'inv', 'inv_src', 'ood_1', 'ood_2', 'epoch']
    # col_names = ['prefix', 'std', 'rob', 'inv', 'inv_src', 'epoch']

    df = pd.read_csv(result_path, header=None, names=col_names)

    contains_str = "cifar10_stl10_randaugment_simclr_none_0.001_0.0_0.0"
    df_selected = df[df['prefix'].str.contains(contains_str)]
    print(df_selected)

    df_selected[['std', 'rob', 'inv', 'inv_src']] *= 100
    df_selected = df_selected.agg(['mean', 'std'])


    def format_table_latex(pair):
        # return f'{pair[0]:.1f}'
        return f'\\facc{{{pair[0]:.1f}}}{{{pair[1]:.1f}}}'

    def format_table_text(pair):
        return f'{pair[0]:.1f} ± {pair[1]:.1f}'
        # return f'\\facc{{{pair[0]:.1f}}}{{{pair[1]:.1f}}}'

    table_data_latex = df_selected[['rob', 'inv', 'std']].apply(format_table_latex, axis=0).to_list()
    table_data_text = df_selected[['rob', 'inv', 'std']].apply(format_table_text, axis=0).to_list()

    print(filename)
    print(' & '.join(table_data_latex))
    print('\t'.join(table_data_text))
    print('\n')