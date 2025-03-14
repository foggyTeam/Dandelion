import os


def save_csv(df, path, df_name):
    df.to_csv(os.path.join(path, df_name), index=False)

    print(df_name + ' saved as csv.')
