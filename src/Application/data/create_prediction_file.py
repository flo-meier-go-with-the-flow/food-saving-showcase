from datetime import date,timedelta
import os
import pandas as pd
import pickle



if __name__ == '__main__':
    features=pd.read_pickle('features_2023.pkl')

    predictions = pd.DataFrame(columns=['log_num_people_11_30','num_people_11_30','num_menus_sold'], index=features.index)

    with open('predictions_2023.pkl','wb') as file:
        pickle.dump(predictions,file)





