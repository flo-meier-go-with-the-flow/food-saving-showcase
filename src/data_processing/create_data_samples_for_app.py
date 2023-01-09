import os
import pandas as pd
import pickle


if __name__ == '__main__':

    pickle_path = os.path.join('..', '..', 'data', 'features_by_date')
    features_by_date = pd.read_pickle(pickle_path)
    features=features_by_date[['log_num_people_11_30','zuehlke_day','before_after_holydays']]
    pickle_filename = os.path.join('features_2023.pkl')
    with open(pickle_filename,'wb') as file:
        pickle.dump(features, file)