
import json
import numpy as np
import pandas as pd

def check_output(file):

    print("")
    print("-"*80)
    print(' File (%s)' % file.split("/")[-1])
    print("-"*80)

    # Write a proxy method to be passed to `pipe`
    def agg_assign(gb, fdict):
        data = {
            (cl, nm): gb[cl].agg(fn)
            for cl, d in fdict.items()
            for nm, fn in d.items()
        }
        pd.options.display.float_format = '{:.0f}'.format
        return pd.DataFrame(data)

    # Read data
    data = []
    with open(file) as input_file:

        for line in input_file:
            data.append(json.loads(line))

    # Dataframe
    df = pd.DataFrame(data)

    # New columns
    df['text_length'] = df['text'].apply(lambda x: len(x))
    df['newspaper'] = df['url'].apply(lambda x: x.split(".")[2].split("/")[1] if x.split(".")[1]=="elpais" else x.split(".")[1])

    # groupby object
    gb = df.groupby('newspaper')

    print("\nArticles parsed %s" % gb['newspaper'].count().sum())
    print("\nInformation:")
    print("-"*80)

    # Identical dictionary passed to `agg`
    funcs = {
        'newspaper': {
            'count': 'count',
        },
        'text_length': {
            'mean': 'mean',
            'min': 'min',
            'max': 'max'
        }
    }

    print(gb.pipe(agg_assign, fdict=funcs))

    print("\nArticles with lowest amount of text")
    print("-"*80)
    print(df.sort_values('text_length', ascending=True)[['url', 'text_length']].head(10))
