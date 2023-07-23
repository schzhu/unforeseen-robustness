import os
import pandas as pd


pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 1000)
pd.set_option('display.expand_frame_repr', False)
pd.set_option('max_colwidth', -1)

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

    contains_str_1 = "stl10"
    contains_str_2 = "0.1_0.1"
    df_selected = df[df['prefix'].str.contains(contains_str_1) &
                     df['prefix'].str.contains(contains_str_2)]
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