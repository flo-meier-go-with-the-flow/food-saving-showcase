import os
import pandas as pd




if __name__ == '__main__':

    path_to_csv_files=os.path.join('..','..', 'data','month_csv_files')

    file_names = os.listdir(path_to_csv_files)
    file_paths= [os.path.join(path_to_csv_files, file_name) for file_name in file_names]
    dataframes=[pd.read_csv(file_path) for file_path in file_paths]
    flow_counts_df=pd.concat(dataframes, axis=0)
    master_csv_file_name='flow_counts.csv'
    path_to_master_csv = os.path.join('..', '..', 'data',master_csv_file_name)
    flow_counts_df.to_csv(path_to_master_csv, index=False)



