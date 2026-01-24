# -*- coding: cp1251 -*-
import pandas as pd


def narratives_ranging(df):
    """Функция возвращает список ранжированных DataFrames
     по каждому нарративу по установленному порогу"""
    threshold = 0.8
    filtered_dfs = []
    # df = pd.read_excel(file, engine='openpyxl')

    # Находим столбцы, которые есть в текущем файле
    sim_columns = [col for col in df.columns if col.startswith('Сходство с')]
    # Фильтруем строки, где хотя бы в одном столбце sim значение > threshold
    df_filtered = df[df[sim_columns].gt(threshold).any(axis=1)]

    filtered_dfs.append(df_filtered)

    all_dfs = []
    for sim_column in sim_columns:
        # Объединяем все отфильтрованные строки в один DataFrame
        filtered_df = pd.concat(filtered_dfs, ignore_index=True).sort_values(
            by=sim_column, ascending=False  # можно сортировать по первому sim, если нужно
        )
        filtered_df = filtered_df[filtered_df[sim_column] > threshold]

        new_order = ["sentence", "connected sentences",
                     f"{sim_column}", "actors", "actions", "objects"]
        filtered_df = filtered_df[new_order]
        all_dfs.append(filtered_df)
        # Сохраняем результат
        # filtered_df.to_excel(fr'E:\Грант\Формализация и схожесть\Результаты\Сходство_{sim_column}.xlsx', index=False)

    return all_dfs
