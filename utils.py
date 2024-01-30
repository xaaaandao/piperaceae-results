import itertools
import pandas as pd
import pathlib
import re

descriptors = {
    'vgg16': [512],
    'mobilenetv2': [1280],
    'resnet50v2': [2048],
    'lbp': [59],
    'surf64': [257]
}
classifiers = ['DecisionTreeClassifier', 'MLPClassifier', 'SVC', 'RandomForestClassifier', 'KNeighborsClassifier']
sizes = ['256', '400', '512']
minimums = [5, 10, 20]
colors = ['RGB', 'grayscale']


def get_indexs_rgb():
    return ['%s+%s+%d' % (k, m, v[0]) for k, v in descriptors.items() for m in ['mean', 'std'] if
            k not in ['lbp', 'surf64']]


def get_indexs():
    return ['%s+%s+%d' % (k, m, v[0]) for k, v in descriptors.items() for m in ['mean', 'std']]


def get_columns():
    return ['%s+%s' % (c, s) for c in classifiers for s in sizes]


def get_df(color):
    return pd.DataFrame(index=get_indexs_rgb(), columns=get_columns()) if 'RGB' in color else pd.DataFrame(
        index=get_indexs(), columns=get_columns())


def get_dfs():
    return {'%s+%s' % (minimum, color): get_df(color) for color in colors for minimum in minimums}


def parse_folder_name(folder_name):
    pat = r'clf\=(.*?)\+len\=(.*?)\+ex\=(.*?)\+ft\=(.*?)\+c\=(.*?)\+dt\=(.*?)\+m\=(.*?)$'
    finds = re.findall(pat, folder_name)
    finds = list(itertools.chain(*finds))
    return {'classifier': finds[0], 'len': finds[1], 'extractor': finds[2], 'features': finds[3], 'color': finds[4],
            'dataset': finds[5], 'minimum': finds[6]}


def get_df_sheet(infos, minimum):
    return '%s+%s' % (str(minimum), infos['color'])


def get_index_mean(infos):
    return '%s+mean+%s' % (infos['extractor'], infos['features'])


def get_index_std(infos):
    return '%s+std+%s' % (infos['extractor'], infos['features'])


def get_column(infos):
    return '%s+%s' % (infos['classifier'], infos['len'])


def create_sheets(dfs, path):
    for minimum in minimums:
        for c in colors:
            for p in pathlib.Path(path).rglob('mean+f1+sum.csv'):

                infos = parse_folder_name(p.parent.parent.parent.name)
                if str(infos['minimum']) == str(minimum) and str(infos['features']) == str(
                        descriptors[infos['extractor']][0]) and c == str(infos['color']):
                    df = pd.read_csv(p, index_col=0, header=None, sep=';')
                    if pd.isna(dfs[get_df_sheet(infos, minimum)].loc[get_index_mean(infos), get_column(infos)]):
                        dfs[get_df_sheet(infos, minimum)].loc[get_index_mean(infos), get_column(infos)] = \
                            df.loc['mean_f1'].values[0]

                    if pd.isna(dfs[get_df_sheet(infos, minimum)].loc[get_index_std(infos), get_column(infos)]):
                        dfs[get_df_sheet(infos, minimum)].loc[get_index_std(infos), get_column(infos)] = \
                            df.loc['std_f1'].values[0]
