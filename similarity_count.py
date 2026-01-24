# -*- coding: cp1251 -*-
import pandas as pd
from similarity_funcs import (make_example_tensor,
                              similarity_economic_meaning,
                              get_embedding)


def similarity_finding(df, narratives):
    examples = make_example_tensor(narratives)
    nar_list = []

    def clean(x):
        x = x.replace('[', '').replace(']', '').replace("'", '').replace(",", '').strip()
        return "" if x == "-" else x

    cols = ['actors', 'actions', 'objects']

    for col in cols:
        df[col] = df[col].map(
            lambda x: ' '.join(x) if isinstance(x, list) and len(x) > 0 else ''
        )

    for _, row in df[['actors', 'actions', 'objects']].fillna('-').replace('[]', '-').iterrows():
        actor, action, obj = clean(row['actors']), clean(row['actions']), clean(row['objects'])
        narrative = " ".join(p for p in [actor, action, obj] if p)
        nar_list.append(narrative)

    embs = get_embedding(nar_list, verbose=False)

    res = similarity_economic_meaning(embs, examples)

    sim_df = pd.DataFrame(res, columns=[f'Сходство с "{narratives[i]}"' for i in range(len(narratives))])
    # "Сходство с" используется для доступа к колонкам далее!!!!
    df_new = pd.concat([df.reset_index(drop=True), sim_df.reset_index(drop=True)], axis=1)
    return df_new


