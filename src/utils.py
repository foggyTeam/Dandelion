import os


def check_folders(raw_data_path, processed_data_path):
    if not os.path.exists(raw_data_path):
        os.mkdir(raw_data_path)
    if not os.path.exists(processed_data_path):
        os.mkdir(processed_data_path)


def save_csv(df, path, df_name):
    df.to_csv(os.path.join(path, df_name), index=False)

    print(df_name + ' saved as csv.')
