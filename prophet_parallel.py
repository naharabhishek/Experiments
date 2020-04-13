## Importing packages
import numpy as np
import pandas as pd
from fbprophet import Prophet
import time
from multiprocessing import Pool, cpu_count
import datetime as dt
from aion.utils import suppress_stdout_stderr
import warnings
warnings.filterwarnings("ignore")

try:
    from rpy2 import robjects, rinterface
    import rpy2.robjects.packages as rpackages
    from rpy2.rinterface import RRuntimeError
except ImportError as e:
    raise ImportError(str(e)) from e
stats_pkg = rpackages.importr("stats")

import os

# Find the R files that you would like to run
this_dir = os.path.dirname(os.path.realpath(__file__))

# R only supports forward slashes (in windows only probably)
this_dir = this_dir.replace("\\", "/")
r_files = [
    n[:-2] for n in os.listdir(f"{this_dir}/R/") if n[-2:] == ".R"
]
# Load the R files
for n in r_files:
    try:
        robjects.r(f'source("{this_dir}/R/{n}.R")')
    except RRuntimeError as er:
        raise RRuntimeError(str(er)) from er


# To generate random time-series
def rnd_timeserie(min_date, max_date, future=False):
    time_index = pd.date_range(min_date, max_date)
    dates = (pd.DataFrame({'ds': pd.to_datetime(time_index.values)},
                          index=range(len(time_index))))
    if future:
        return dates
    y = np.random.random_sample(len(dates))*10
    dates['y'] = y
    return dates


# Python wrapper to use R Forecast
class RForecast:
    """
    RForecastModel instantiates the RForecast to fit and predict.

    Parameters
    ----------
     method_name
        The method from rforecast to be used one of
        "ets", "arima", "tbats", "croston", "mlp". (Case-Sensitive)

    """
    def __init__(self, method_name):

        # Supported methods come from forecast_methods.R
        supported_methods = ["ets", "arima", "tbats", "croston", "mlp"]

        # Method name should be one of the supported methods (Case sensitive)
        self._method_name = method_name
        rinterface.initr()

        # Check if method supported
        assert (method_name in supported_methods), \
            f"method {method_name} is not supported please use one of {supported_methods}"

        self._r_method_fit = robjects.r[method_name]
        self._r_method_predict = robjects.r['predict']
        self.model = None

    def _unlist(self, l):
        if type(l).__name__.endswith("Vector"):
            return [self._unlist(x) for x in l]
        else:
            return l

    def fit(self, df: pd.DataFrame):
        # convert observations to R vector and time-series
        vec = robjects.FloatVector(df.y)
        ts = stats_pkg.ts(vec)
        # Use method_name selected by user to call the R function
        self.model = self._r_method_fit(ts)

    def predict(self, df: pd.DataFrame):
        # prediction_length or forecasting horizon or back test window offset if forecasting horizon not mentioned
        prediction_length = df.shape[0]
        params = {'prediction_length': prediction_length}
        r_params = robjects.vectors.ListVector(params)
        # use fit model to call predict function in forecast_methods.R
        forecast = self._r_method_predict(self.model, r_params)
        forecast_dict = dict(
            zip(forecast.names, map(self._unlist, list(forecast)))
        )
        # results from R stored in dataframe
        df['yhat'] = forecast_dict['mean']
        return df

def fit_model(level, timeseries,model):
    model.fit(timeseries)
    return (level, model)

def predict_model(level, forecast_timeseries, model):
    level_forecast_df = model.predict(forecast_timeseries)
    return (level, level_forecast_df)

def run_r(x, timeserie):
    model = RForecast('ets')
    model.fit(timeserie)
    forecast_start_date = dt.date(2020, 1, 1)
    forecast_end_date = dt.date(2020, 3, 30)
    forecast = rnd_timeserie(forecast_start_date, forecast_end_date, future=True)
    forecast = model.predict(forecast)
    return (forecast, x)

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

min_date = dt.date(2010, 1, 1)
max_date = dt.date(2019, 12, 31)

num_lod_combinations = 250
series = [rnd_timeserie(min_date,max_date) for x in range(0,num_lod_combinations)]

forecast_start_date = dt.date(2020, 1, 1)
forecast_end_date = dt.date(2020, 3, 30)
forecast_series = [rnd_timeserie(forecast_start_date,forecast_end_date,future=True) for x in range(0,num_lod_combinations)]

index = [x for x in range(0,num_lod_combinations)]
results = []

if __name__ == '__main__':
    print("Starting")
    with suppress_stdout_stderr():

        ################ Sequential processing ##########################
        start_time = time.time()
        result = list(map(lambda x, timeserie: run_r(x, timeserie), index, series))
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
            p.apply_async(run_r, args=(i, row), callback=collect_result)
        p.close()
        p.join()
        timed_fit_async = time.time() - start_time
        results = []
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
        ################ Multiprocess asynchronized (fit only)##########################
        start_time = time.time()
        p = Pool(cpu_count())
        model = RForecast('arima')
        # Use loop to parallelize
        for i, row in enumerate(series):
            p.apply_async(fit_model, args=(i, row,model), callback=collect_result)
        p.close()
        p.join()
        print(results)
        model_dict = {mod[0]: mod[1] for mod in results}
        timed_fitonly_async = time.time() - start_time
        results = []
        #print("--- %s parallel Async fit only seconds ---" % timed_fitonly_async)

        #################################################################
        ################ Multiprocess asynchronized (predict only)##########################
        start_time = time.time()
        p = Pool(cpu_count())
        # Use loop to parallelize
        for i, row in enumerate(forecast_series):
            p.apply_async(predict_model, args=(i, row,model_dict[i]), callback=collect_result)
        p.close()
        p.join()
        pred_df_dict = {mod[0]: mod[1] for mod in results}
        timed_predictonly_async = time.time() - start_time
        #print("--- %s parallel Async predict only seconds ---" % timed_predictonly_async)
        results = []

    print("--- %s sequential fit seconds ---" % timed_fit_seq)
    print("--- %s parallel Async fit seconds ---" % timed_fit_async)
    print("--- %s parallel Async fit only seconds ---" % timed_fitonly_async)
    print("--- %s parallel Async predict only seconds ---" % timed_predictonly_async)
        # print("--- %s dask fit seconds ---" % timed_fit_dask)
