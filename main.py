import argparse
import pandas as pd 
import numpy as np
import datetime
import keras

def loadPredictData(pathG, pathC):
    dfG = pd.read_csv(pathG)
    dfC = pd.read_csv(pathC)
    
    df = pd.merge(dfG, dfC, on='time')
    
    df['time'] = pd.to_datetime(df['time'])
    df['month'] = df['time'].dt.month
    df['day'] = df['time'].dt.dayofweek
    df['hour'] = df['time'].dt.hour
    df = df.drop(['time'], axis=1)
    
    return df

import datetime

def Predict(data, last_date):
    trained_modelG = keras.models.load_model('modeG.h5')
    trained_modelC = keras.models.load_model('modeC.h5')
    
    date = datetime.datetime(2018, last_date[0], last_date[1], last_date[2], 0, 0)
    
    resultG = []
    resultC = []
    
    SIZE = 24 # 1 day / 24 hours
    for _ in range(SIZE):
        predictG = trained_modelG.predict(data)[0]
        predictC = trained_modelC.predict(data)[0]
        
        
        resultG.append(float(predictG))
        resultC.append(float(predictC))
        
        date += datetime.timedelta(hours=1)
        
        tmp = [predictG[0], predictC[0], date.month, date.day, date.hour]
        
        data = np.array([np.vstack((data[0], tmp))[1:]])
        
        print('{} done'.format(date))
        
    return resultG, resultC


# You should not modify this part.
def config():
    parser = argparse.ArgumentParser()
    parser.add_argument("--consumption", default="./sample_data/consumption.csv", help="input the consumption data path")
    parser.add_argument("--generation", default="./sample_data/generation.csv", help="input the generation data path")
    parser.add_argument("--bidresult", default="./sample_data/bidresult.csv", help="input the bids result path")
    parser.add_argument("--output", default="output.csv", help="output the bids path")

    return parser.parse_args()


def output(path, data):
    df = pd.DataFrame(data, columns=["time", "action", "target_price", "target_volume"])
    df.to_csv(path, index=False)

    return


if __name__ == "__main__":
    args = config()

    # data = [["2018-01-01 00:00:00", "buy", 2.5, 3],
    #         ["2018-01-01 01:00:00", "sell", 3, 5]]
    # output(args.output, data)
    
    # load data for prediction
    data_to_predict = loadPredictData()
    
    # start predict...
    last_month = data_to_predict['month'].iloc[-1]
    last_day = data_to_predict['day'].iloc[-1]
    last_hour = data_to_predict['hour'].iloc[-1]
    resultG, resultC = Predict(np.array([data_to_predict.values]), (last_month, last_day, last_hour))
    
    # load bid-result history
    bid_res = pd.read_csv(args.bidresult)
    
    
    
    
