## Importing packages
from distributed import Client, LocalCluster
import dask
import numpy as np
import pandas as pd
from fbprophet import Prophet
import time
from multiprocessing import Pool, cpu_count
## To generate random time-series
def rnd_timeserie(min_date, max_date):
    time_index = pd.date_range(min_date, max_date)
    dates = (pd.DataFrame({'ds': pd.to_datetime(time_index.values)},
                          index=range(len(time_index))))
    y = np.random.random_sample(len(dates))*10
    dates['y'] = y
    return dates
## Function to multi-process
def run_prophet(x, timeserie):
    model = Prophet(yearly_seasonality=False,daily_seasonality=False,uncertainty_samples = False)
    model.fit(timeserie)
    forecast = model.make_future_dataframe(periods=90, include_history=False)
    forecast = model.predict(forecast)
    return (forecast, x)
# Define callback function to collect the output in `results` (Used for multiprocess async)
def collect_result(result):
    global results
    results.append(result)
## Generating timeseries and some random input

import datetime as dt
min_date = dt.date(2010, 1, 1)
max_date = dt.date(2019, 12, 31)

num_lod_combinations = 20

series = [rnd_timeserie(min_date,max_date) for x in range(0,num_lod_combinations)]
index = [x for x in range(0,num_lod_combinations)]
results = []
if __name__ == '__main__':
    print("Starting")
    ################ Sequential processing ##########################
    start_time = time.time()
    result = list(map(lambda x, timeserie: run_prophet(x, timeserie), index, series))
    timed_fit_seq = time.time() - start_time
    # print("--- %s sequential seconds ---" % (time.time() - start_time))
    # #################################################################
    # ################ Multiprocess synchronized##########################
    # start_time = time.time()
    # p = Pool(cpu_count())
    # predictions = list(p.starmap(run_prophet, zip(index, series)))
    # p.close()
    # p.join()
    # print("--- %s parallel sync seconds ---" % (time.time() - start_time))
    #################################################################
    ################ Multiprocess asynchronized##########################
    start_time = time.time()
    p = Pool(cpu_count())
    # Use loop to parallelize
    for i, row in enumerate(series):
        p.apply_async(run_prophet, args=(i, row), callback=collect_result)
    p.close()
    p.join()
    timed_fit_async = time.time() - start_time
    # print("--- %s parallel async seconds ---" % (time.time() - start_time))
    ##################################################################
    ###################### Using Dask ##############################
    # cluster = LocalCluster()
    # client = Client(cluster)
    # start_time = time.time()
    # delayed_output = []
    # for i, row in enumerate(series):
    #     delayed_output.append(dask.delayed(run_prophet)(i, row))
    # dask.compute(delayed_output)
    # client.shutdown()
    # timed_fit_dask = time.time() - start_time
    # # print("--- %s parallel dask seconds ---" % (time.time() - start_time))
    #################################################################

    print("--- %s sequential fit seconds ---" % timed_fit_seq)
    print("--- %s parallel Async fit seconds ---" % timed_fit_async)
    # print("--- %s dask fit seconds ---" % timed_fit_dask)
