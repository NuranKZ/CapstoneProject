#!/usr/bin/env python
# coding: utf-8

# 0. Necessary imports
import pandas as pd
import numpy as np
import operator
import matplotlib.pyplot as plt
import networkx as nx
import seaborn as sns
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import adfuller
import itertools
import statsmodels.api as sm
from datetime import date as dt
from datetime import timedelta
from sklearn.metrics import r2_score, f1_score, accuracy_score
from matplotlib.pylab import rcParams
import time
from pprint import pprint
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping
import requests
import json

# 1. functions for Data Scrapping
# -- functions used in S1_DataScrapping.ipynb

def get_jsonparsed_data(url):
    """
    Receive the content of `url`, parse it as JSON and return the object.
    
    Parameters
    ----------
    url : str given by API-syntax
    
    Returns
    -------
    dict or list of dicts
    """
    r = requests.get(url)
    if r.ok:
        data = json.loads(r.text, encoding='utf-8')
        return data
    else:
        print('Connection Failed')
    

def find_shablons(key_word, array):
    """
    Receive the `key_word` as string shablon, find all words in `array`
    which contain it and return list of those words
    
    Parameters
    ----------
    key_word : str
    array: list
    
    Returns
    -------
    list
    """
    
    result_list = []
    key_word = key_word.lower()
    for comp in array:
        comp = str(comp)
        comp_lower = comp.lower()
        if comp_lower.find(key_word) != -1:
            result_list.append(comp)
    return result_list

def dt_to_string(data_in_dt):
    """
    Receive the `data_in_dt` and return string-version of date as in loaded files.
    This function used only for `url_for_stock()` function
    
    Parameters
    ----------
    data_in_dt : DateTime.Date
    
    Returns
    -------
    STR
    """
        
    year, month, day = data_in_dt.year, data_in_dt.month, data_in_dt.day
    if month<10:
        month = '0'+str(month)
    if day<10:
        day = '0'+str(day)
    return f'{year}-{month}-{day}'


def url_for_stock(base_url, ticker, start_date_in_dt=None, end_date_in_dt=None):
    """
    Receive the `base_url`, `ticker`, `start_date_in_dt`, `end_date_in_dt`  
    and return URL for API for loading quotes for selected ticker with given dates
    dates are not given full available period till today used in URL
    This function used only for `url_for_stock()` function
    
    Parameters
    ----------
    base_url : STR from API
    ticker: STR (ticker name)
    start_date_in_dt: DateTime.Date
    end_date_in_dt: DateTime.Date
    
    Returns
    -------
    STR
    """
    
    url = base_url+ticker
    if not end_date_in_dt:
        end_date_in_dt = dt.today()
    if start_date_in_dt and start_date_in_dt<=end_date_in_dt:
        url = f'{url}?from={dt_to_string(start_date_in_dt)}&to={dt_to_string(end_date_in_dt)}'
    return url


# 2. functions for Data Cleaning
# -- functions used in S3_DataCleaning.ipynb

def create_grouped_df(group_param, df, agg_func=np.mean):
    """
    Receive the `group_param`, `df`, `agg_func`  
    and return grouped by group_param DataFrame
    with values aggregated by agg_func
    
    Parameters
    ----------
    group_param : STR (name of column)
    df: pandas.DataFrame
    agg_func: function
        
    Returns
    -------
    DataFrame
    """
    gdf = df.groupby(by=group_param)
    df_grouped = gdf.agg(agg_func)
    return df_grouped

def check_and_replace_inf(df, show_inf=True):
    """
    Check and replace np.inf by np.nan in given DataFrame
    and return modified DataFrame
    If show_inf: print identified number of np.inf per column
    
    Parameters
    ----------
    df: pandas.DataFrame
    show_inf: bool
        
    Returns
    -------
    DataFrame
    """
    
    if show_inf:
        inf_dict = dict()
        for col in df.columns:
            n_inf = 0
            for row in df[col].index:
                if df[col].loc[row] == np.inf:
                    n_inf += 1
            if n_inf>0:
                inf_dict[col] = n_inf
        for key, value in inf_dict.items():
            print(f'in {key}: {value} inf values')
    df = df.replace([np.inf, -np.inf], np.nan)
    return df



# 3. functions for Clustering
# -- functions used in S4_EDA_Clustering.ipynb

def create_grouped_df_adv(group_param, df, ticker_name='symbol', 
                      agg_func=np.count_nonzero, show_value_counts=False):
    
    """
    Receive the `group_param`, `df`, `ticker_name`, `agg_func`  
    and return grouped by group_param DataFrame
    with values aggregated by agg_func and Dict with
    group names as keys and list of companies
    included to group as values 
    
    Parameters
    ----------
    group_param : STR (name of column)
    df: pandas.DataFrame
    ticker_name: STR (name of column for adding to dict values)
    agg_func: function
    show_value_counts: bool (print allocation if True)
        
    Returns
    -------
    Tuple of (DataFrame, Dict)
    """
    
    gdf = df.groupby(by=group_param)
    df_grouped = gdf.agg(agg_func)
    
    comp_dict = dict()
    if show_value_counts:
            print(f'number of items per group of {group_param}:')
            print(35*'-')
    
    for group in df_grouped.index:
        df_temp = df.loc[df[group_param]==group]
        name_list = df_temp[ticker_name].tolist()
        comp_dict[group] = name_list
        if show_value_counts:
            print(f'in {group}: {len(name_list)} items')
            
    return df_grouped, comp_dict


def change_categorical_value(df, new_name, old_values, col):
    """
    replace categorical values in `old_values` with `new_name` 
    in given `col` of `df`
    
    Parameters
    ----------
    df: pandas.DataFrame
    new_name: STR
    old_values: LIST
    col: STR
        
    Returns
    -------
    DataFrame
    """
    
    df[col] = [new_name if i in old_values else i for i  in df[col]]
    return df



def collect_corr_tickers(corr_matrix, corr_threshold, n_cluster_limit=None, show_stats=False):
    """
    generate `correlated_features_filtered` and `stats_dict` DICTS on given `corr_matrix` and
    `corr_threshold` by defining positively correlated groups of companies. The algorithm steps:
    (1) collect list of correlated tickers for each ticker with given threshold excluding company itself
    (2) create `stats_dict` dict with tickers sorted by number of correlated tickers
    (3) exclude tickers with smallest list by using `n_cluster_limit`
    (4) show stats for `correlated_features_filtered`  if `show_stats`=True
    
    Parameters
    -----------------------------------------------------------------------------------------------------
    corr_matrix: pandas.DataFrame
    corr_threshold: FLOAT
    n_cluster_limit: BOOL
    show_stats: BOOL
        
    Returns
    -----------------------------------------------------------------------------------------------------
    tuple of DICTS (`correlated_features_filtered`, `stats_dict`)
    """
    
    
    # Generation of internal dict with Ticker and corellated Ticker's list to it
    correlated_features = dict()
    for col in corr_matrix.columns:
        similar_comps = []
        for row in corr_matrix[col].index:
            if corr_threshold <= corr_matrix.loc[row, col] < 1: # only positive corr taken to account
                    similar_comps.append(row)
        if similar_comps != []:
            correlated_features[col] = similar_comps
    # Generation of sorted list of tickers and number of correlated tickers to it
    stats_dict = {key:len(value)+1 for (key, value) in correlated_features.items()}
    stats_dict = sorted(stats_dict.items(),key=operator.itemgetter(1),reverse=True)
    # Filtering dict by size
    if n_cluster_limit:
        correlated_features_filtered = dict()
        for pair in stats_dict:
            if len(correlated_features_filtered)<n_cluster_limit:
                correlated_features_filtered[pair[0]] = correlated_features[pair[0]]
    else:
        correlated_features_filtered = correlated_features.copy()
    # Updating stats_dict
    stats_dict = {key:len(value)+1 for (key, value) in correlated_features_filtered.items()}
    stats_dict = sorted(stats_dict.items(),key=operator.itemgetter(1),reverse=True)
    # Displaying stats
    if show_stats:
        stat_values = pd.Series([i[1] for i in stats_dict])
        biggest_size = stat_values.max()
        smallest_size = stat_values.min()
        average_size = int(round(stat_values.median(),0))
        print(f'total number of identified clusters: {len(correlated_features_filtered)}')
        print(f'biggest cluster size: {biggest_size}')
        print(f'smallest cluster size: {smallest_size}')
        print(f'median cluster size: {average_size}')
    return correlated_features_filtered, stats_dict



def clean_cluster_collection(correlated_features_filtered, show_stats=True):
    """
    calculate and return `cluster_collection` list of sets and `globe_set`
    based on given `correlated_features_filtered`
    where:
      - `cluster_collection` - is a list of sets with companies which are
        not appeared in previos sets
      - `globe_set` - all companies 
    This function's goal - exclude companies dublicated in different clusters
            
    Parameters
    ------------------------------------------------
    `correlated_features_filtered`: DICT (object, 
    generated as result of `collect_corr_tickers`
    
    `show_stats`: BOOL
    
    Returns
    -------
    TUPLE of LIST and SET
    """    
    
    glob_set = set()
    current_cluster_set = set()
    cluster_collection = []
    for c in correlated_features_filtered.keys():
        raw_comps = set([c]+correlated_features_filtered[c])
        current_cluster_set = raw_comps-glob_set
        glob_set.update(raw_comps)
        if len(current_cluster_set) != 0:
            cluster_collection.append(current_cluster_set)
    if show_stats:
        print('after cleaning..')
        print(30*'-')
        clusters_power = pd.Series([len(cl) for cl in cluster_collection])
        print(f'total inclusive clusters number: {len(cluster_collection)}')
        print(f'biggest cluster contains {clusters_power.max()} comps')
        print(f'smallest cluster contains {clusters_power.min()} comps')
        print(f"average cluster's power {clusters_power.median():.0f} comps")
    
    return cluster_collection, glob_set


def calculate_tickers_set(correlated_features_filtered):
    """
    calculate and return unique tickers in SET based
    on given `correlated_features_filtered` DICT
        
    Parameters
    ------------------------------------------------
    `correlated_features_filtered`: DICT (object, 
    generated as result of `collect_corr_tickers`
            
    Returns
    -------
    SET
    """
    
    total_tickers = set()
    for key, value in correlated_features_filtered.items():
        total_tickers.update([key] + value)
    return len(total_tickers)


def plot_cluster_coverage(correlated_features_filtered, stats_dict, corr_matrix,
                          start_step=1, end_step=None, step_in_range=1):

    """
    run iterative process for given `start_step` and `end_step` as numbers of
    remained clusters in `correlated_features_filtered` and create 3 plots:
    (a) portion for all clusters in total tickers set
    (b) average size of cluster
    (c) n_clusters vs real cleaned clusters number
    
    Parameters
    ------------------------------------------------
    `correlated_features_filtered`: DICT (object, 
    generated as result of `collect_corr_tickers`
    `stats_dict`: DICT (object, generated as result 
    of `collect_corr_tickers`)
    `corr_matrix`: pd.DataFrame (initial corr_matrix)
    `start_step`: INT
    `end_step`: INT (if None, end_step defined as size
    of stats_dict
    `step_in_range`: INT (step scale)
            
    Returns
    -------
    None
    """
    #import matplotlib.pyplot as plt
    if not end_step:
        end_step = len(stats_dict)+1
    else:
        end_step = min(end_step, len(stats_dict)+1)
    n_cl_range = list(range(start_step, end_step, step_in_range))
    overall_tickers = len(corr_matrix)
    cleaned_cluster_number, cluster_coverage, avg_cluster_sizes = [], [], []
    for n in n_cl_range:
        cff_reduced = dict()
        for pair in stats_dict:
            if len(cff_reduced) < n:
                cff_reduced[pair[0]] = correlated_features_filtered[pair[0]]
        
        cluster_collection, glob_set = clean_cluster_collection(cff_reduced, show_stats=False)
        cluster_coverage.append(len(glob_set) / overall_tickers)
        clusters_power = pd.Series([len(cl) for cl in cluster_collection])
        avg_cluster_sizes.append(int(round(clusters_power.median(),0)))
        cleaned_cluster_number.append(len(cluster_collection))
    if len(cluster_coverage)>0:
        fig, ax = plt.subplots(3,1, figsize=(14,10), sharex=True)
        ax[0].title.set_text('portion for all clusters in total tickers set')
        ax[1].title.set_text('average size of cluster')
        ax[2].title.set_text('n_clusters vs real cleaned clusters number')
        ax[0].plot(n_cl_range, cluster_coverage, color='g')
        ax[1].plot(n_cl_range, avg_cluster_sizes, color='b')
        ax[2].plot(n_cl_range, cleaned_cluster_number, color='r')
        plt.tight_layout()
        plt.show()


def define_nodes_set(cff):
    """
    calculate and return unique tickers in SET based
    on given `correlated_features_filtered` DICT
    this function works same as `calculate_tickers_set`
        
    Parameters
    ------------------------------------------------
    `correlated_features_filtered`: DICT (object, 
    generated as result of `collect_corr_tickers`
            
    Returns
    -------
    SET
    """
    
    total_nodes = set()
    for key, value in cff.items():
        total_nodes.update([key] + value)
    return total_nodes


def create_graph_object(total_nodes, cff):
    """
    create NetworkX Graph object based on given
    `total_nodes` for nodes and connect nodes by 
    pairs from `cff` clusters
        
    Parameters
    ------------------------------------------------
    `total_nodes`: SET (from `define_nodes_set`)
    `cff`: DICT (object, 
    generated as result of `collect_corr_tickers`
            
    Returns
    -------
    Graph Object
    """
    
    graph = nx.Graph()
    graph.add_nodes_from(total_nodes)
    connected_nodes = []
    for key in cff.keys():
        for comp in cff[key]:
            connected_pair = (key, comp)
            connected_nodes.append(connected_pair)
    graph.add_edges_from(connected_nodes)
    return graph


def calculate_graph_centers(graph, verbose=False):
    """
    calculate graph centers on given `graph`
    and return centers in LIST `centers`
        
    Parameters
    ------------------------------------------------
    `graph`: Graph object
    `verbose`: Bool
      - if True: display calc results
            
    Returns
    -------
    LIST
    """
    
    try:
        nodes_to_remove = list(nx.isolates(graph))
        graph.remove_nodes_from(nodes_to_remove)
        centers = list(nx.center(graph))
        if verbose:
            print(f'{len(nodes_to_remove)} unconnected nodes removed')
            print(f'{len(centers)} clusters identified')
    except Exception as e:
        if verbose:
            print(e)
        centers = []
    return centers

def plot_graph(graph, centers, node_size=8000, font_size=12, figsize=(20,15)):
    """
    plot graph object with colored centers
    
    Parameters
    ------------------------------------------------
    `graph`: Graph object
    `centers`: LIST (centers tickers)
    `node_size`: INT (size of node in plot)
    `font_size`: INT (size of text in nodes)
    `figsize`: TUPLE (figure size)
    
    Returns
    -------
    None
    """
    
    color_map = []
    for node in graph:
        if node in centers:
            color_map.append('red')
        else: 
            color_map.append('green')
    plt.figure(figsize=figsize)
    nx.draw(graph, pos=nx.spring_layout(graph, k=3, seed=10), with_labels=True, alpha=.8, 
            node_size=node_size, font_weight="bold", font_size=font_size, node_color=color_map) 


def show_groups_dynamics_single_df(df_profiles_clustered_collection, center_name, df_prices_pct_centers,
                                   period='AS',  borders='q25-q75', sigma_coef=1):
    """
    plot all composites dynamics 
    rescaled by `period`:'D'-daily, 'MS'-monthly, 'AS'-annualy
    and with optional bool param `plot_std`;
    works same as method in `GroupedDFCollection` class (see in S4_EDA_Clustering.ipynb)
    
    Parameters
    ------------------------------------------------------------------------------------
    `df_profiles`: DICT with DataFrames where df correspond to a cluster
    `center_name`: STR (ticker name for selected cluster)
    `df_prices_pct_centers`: pd.DataFrame (which consist of only cluster centers)
    `period`: STR ('D','MS' or 'AS') - scaling parameter
    `borders`: STR (borders definition, see: `GroupedDFCollection` for details)
    `sigma_coef`: INT (see: `GroupedDFCollection` for details)
    
    Returns
    -------
    None
    """    

    import seaborn as sns    
    timestamp = pd.to_datetime(df_prices_pct_centers.index, format='%Y-%m-%d')
    df_comps = df_profiles_clustered_collection[center_name].set_index(timestamp)
    df_composite = pd.DataFrame(df_prices_pct_centers[center_name], columns=[center_name])
    df_composite = df_composite.set_index(timestamp)
    df_comps = df_comps.resample(period).mean()
    df_composite = df_composite.resample(period).mean()
    stat_data = df_comps.T.describe().T
    
    fig = plt.figure(figsize=(15,4))
    sns.lineplot(data=df_composite, c='orange', linewidth=1.2)
    title_ending=''
    if borders:
        border_min, border_max = stat_data['min'], stat_data['max']
        border_q25, border_q75 = stat_data['25%'], stat_data['75%']
        border_minus_Xsigma = stat_data['mean'] - sigma_coef*stat_data['std'] 
        border_plus_Xsigma = stat_data['mean'] + sigma_coef*stat_data['std']
        if borders == 'min-max':
            border_low, border_high = border_min, border_max
        elif borders == 'sigma':
            border_low, border_high = border_minus_Xsigma, border_plus_Xsigma
        elif borders == 'q25-q75':
            border_low, border_high = border_q25, border_q75
        sns.lineplot(data=border_low, c='r', linewidth=0.3, style='-')
        sns.lineplot(data=border_high, c='green', linewidth=0.3)
        plt.fill_between(df_composite.index, border_low.values, 
                            border_high.values, alpha=0.1, color='green')
        title_ending = f' with borders {borders}'
        
    plt.legend()
    plt.title(f'{center_name} composite dynamics p.{period[0].lower()}{title_ending}')
    plt.show()


# 4. functions for ARIMA modelling
# -- functions used in S5_ARIMA.ipynb


def rolling_plots(df, col_name, window, figsize=(10,4)):
    """
    plot `df[col_name]` time serie with adding rolling
    mean and std defined by `window` size
    
    Parameters
    ------------------------------------------------
    `df`: pd.DataFrame
    `col_name`: STR (name of target column)
    `window`: INT (window size)
    `figsize`: TUPLE (figure size)
        
    Returns
    -------
    None
    """
    
    print(col_name)
    print(80*'-')
    ts = df[col_name]
    rolmean = ts.rolling(window = window, center = False).mean()
    rolstd = ts.rolling(window = window, center = False).std()
    fig = plt.figure(figsize=(12,5))
    orig = plt.plot(ts, color='blue',label='Original')
    mean = plt.plot(rolmean, color='red', label='Rolling Mean')
    std = plt.plot(rolstd, color='black', label = 'Rolling Std')
    plt.title(f'Rolling Mean & Standard Deviation for {col_name}')
    plt.legend(loc='best')
    plt.show()





def show_acf_pacf(df, col_name, add_delta=False, delta_step=1, figsize=(10,2)):
    """
    plot ACF and PACF for given `df[col_name]` time serie;
    add additional PACF with given `delta_step` if
    `add_delta`=True
        
    Parameters
    ------------------------------------------------
    `df`: pd.DataFrame
    `col_name`: STR (name of target column)
    `add_delta`: BOOL
    `delta_step`: INT
    `figsize`: TUPLE (figure size)
        
    Returns
    -------
    None
    """
    
    print(col_name)
    print(80*'-')
    ts_composite = df[col_name]
    rcParams['figure.figsize'] = figsize
    fig = plt.figure(figsize=figsize)
    ts_acf_plot = plot_acf(ts_composite, title=f'ACF for {col_name}')
    ts_pacf_plot = plot_pacf(ts_composite, title=f'PACF for {col_name} 1 period')
    if add_delta:
        if delta_step:
            ts_pacf_plot = plot_pacf(ts_composite.diff(delta_step).dropna(), 
                                     title=f'PACF for delta({delta_step}) on {col_name} 1 period');
    plt.show()


def show_adf_test(ts):
    """
    calculate and print ADF-test for given `ts` 
    with verdict (alpha = 0.05)
        
    Parameters
    ------------------------------------------------
    `ts`: pd.Series
        
    Returns
    -------
    None
    """
    
    adf_test = adfuller(ts)
    adf_test = pd.Series(adf_test[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
    p_val = adf_test['p-value']
    if p_val < 0.05:
        verdict = f'{ts.name} time series has stationarity'
    else:
        verdict = f'{ts.name} time series has no stationarity!'
    print(f'ADF-test for {ts.name}')
    print(verdict)
    print(f'p-value = {p_val}')
    print(50*'-')
    
    
def find_optimal_params(ts, p_range, d_range, q_range, s, show_best_params=True, verbose=False):
    """
    create grid search loops for given `ts` with initialization 
    of SARIMAX models for each iteration and return pd.Series 
    for optimal params. 
    Best params defined by AIC.
    Hyperparameter space defined by `p_range`, `d_range`,
    `q_range` and single `s`.
    If `show_best_params`=True, display best params.
    If `verbose`=True, show each iteration results (params and AIC)
            
    Parameters
    ------------------------------------------------
    `ts`: pd.Series
    `p_range`, `d_range`, `q_range`: iterable object
    `s`: INT
    `show_best_params`: BOOL
    `verbose`: BOOL
        
    Returns
    -------
    pd.Series
    """
    
    s = s
    p, d, q = p_range, d_range, q_range
    pdq = list(itertools.product(p, d, q))
    pdqs = [(x[0], x[1], x[2], s) for x in list(itertools.product(p, d, q))]
    ans = []
    for comb in pdq:
        for combs in pdqs:
            try:
                model = sm.tsa.statespace.SARIMAX(ts, order=comb, seasonal_order=combs,
                                                enforce_stationarity=False, enforce_invertibility=False)
                output = model.fit()
                ans.append([comb, combs, output.aic])
                if verbose:
                    print('ARIMA {} x {}{} : AIC Calculated ={}'.format(comb, combs, s, output.aic))
            except:
                continue
    ans_df = pd.DataFrame(ans, columns=['pdq', 'pdqs', 'aic'])
    if show_best_params:
        print(ans_df.loc[ans_df['aic'].idxmin()])
    return ans_df.loc[ans_df['aic'].idxmin()]    
    
    
def run_arima_model(ts, order, seasonal_order, show_summary=False, forecast_start_date='2019-05-13',
                    overal_start_date = '2016-01-01', plot_forecast=True, show_r2=True, dynamic=False):
    
    """
    this function initialize single SARIMAX model, fit is with
    given `ts` serie with given `order` (p-d-q) and 
    `seasonal_order` (p-d-q-s) and return trained model instance.
    
    If `show_summary` = True, print model.summary().
    
    if `plot_forecast`=True, plot initial series starting from
    `overal_start_date` till last date and plot forecast time
    serie with significance borders starting from `forecast_start_date`
    
    Type of forecast defined by `dynamic`:
    - if False: 1-step-ahead forecast
    - if True: full-cycle forecast
    
    If `show_r2`: calculate and print r2_score on given forecast
                
    Parameters
    ------------------------------------------------
    `ts`: pd.Series
    `order`: tuple with (p,d,q)
    `seasonal_order`: tuple with (p,d,q,s)
    `show_summary`: BOOL
    `forecast_start_date`: STR in form ("YYYY-MM-DD")
    `overal_start_date`: STR in form ("YYYY-MM-DD")
    `plot_forecast`: BOOL
    `show_r2`: BOOL
    `dynamic`: BOOL
            
    Returns
    -------
    sm.tsa.statespace.SARIMAX object
    """
    
    try:
        ts_model = sm.tsa.statespace.SARIMAX(endog=ts, order=order, seasonal_order=seasonal_order,
                                                enforce_stationarity=False, enforce_invertibility=False)

        ts_output = ts_model.fit(maxiter=300)
        pred = ts_output.get_prediction(start=pd.to_datetime(forecast_start_date), dynamic=dynamic)
        
        if show_summary:
            ts_output.summary()

        prices_forecasted = pred.predicted_mean
        prices_true = ts[forecast_start_date:]
        r2 = r2_score(prices_true,prices_forecasted)
        if show_r2:
            print(f'r2 for forecasted {len(prices_forecasted)} days: {100*r2:.2f}% with dynamic = {dynamic}')

        if dynamic == False:
            label = 'One-step ahead Forecast'
            title_add = '1-step-ahead'
        else:
            label = 'Full forecast'
            title_add = 'fully'
            
        if plot_forecast:
            fig = plt.figure(figsize=(15,5))
            pred_conf = pred.conf_int()
            ax = ts[overal_start_date:].plot(label='observed')
            pred.predicted_mean.plot(ax=ax, label=label, alpha=.9)
            ax.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='g', alpha=.5)
            ax.set_xlabel('Date')
            ax.set_ylabel(ts.name)
            plt.title(f'time serie actual and forecasted {title_add}')
            plt.legend();
    
        return ts_model
    
    except Exception as e:
        print(e)
        return None      
    
    
    
def calculate_r2_on_period(model, forecast_period, ts, ts_name=None, dynamic_for_plot =False, plot_forecast=False,
                          overal_start_year=None):
    
    """
    this function calculate r2 scores on fitted `model` based on
    given `ts` and `forecast_period` (length of period) and return
    - r2_1: r2 for 1-step-ahead forecast
    - r2_2: r2 for full cycle forecast
    
    If `plot_forecast`=True: plot one version of forecast with 
    initial ts from `overal_start_year`
    Version of forecast defined by `dynamic_for_plot`
    
    Due to the fact, that some days could be missed in datetime
    index of ts, function return exception, if last period of 
    forecast_period is not in ts index
    
    Parameters
    --------------------------------------------------------------
    `model`: sm.tsa.statespace.SARIMAX object
    `forecast_period`: INT (number of forecasted periods)
    `ts`: pd.Series
    `ts_name`: STR (label for plot)
    `dynamic_for_plot`: BOOL
    `plot_forecast`: BOOL
    `overal_start_year`: STR in form ("YYYY-MM-DD")
                
    Returns
    -------
    tuple of floats (r2_1, r2_2)
    """    
    
    ts.index = pd.to_datetime(ts.index)
    last_date = ts.index.max()
    start_date = last_date - timedelta(forecast_period)
    model_output = model.fit()
    try:
        cutted_index = ts[str(start_date):str(last_date)].index
        pred_1 = model_output.get_prediction(start=start_date, dynamic=False)
        pred_2 = model_output.get_prediction(start=start_date, dynamic=True)
        true = ts[cutted_index]
        r2_1 = r2_score(true, pred_1.predicted_mean)
        r2_2 = r2_score(true, pred_2.predicted_mean)
        if not overal_start_year:
            overal_start_year = start_date
        
        if plot_forecast:
            if dynamic_for_plot==False:
                pred = pred_1
            else:
                pred = pred_2
            fig = plt.figure(figsize=(15,5))
            pred_conf = pred.conf_int()
            ax = ts[overal_start_year:].plot(label='observed')
            pred.predicted_mean.plot(ax=ax, label=ts_name, alpha=.9)
            ax.fill_between(pred_conf.index, pred_conf.iloc[:, 0], pred_conf.iloc[:, 1], color='g', alpha=.5)
            ax.set_xlabel('Date')
            ax.set_ylabel(ts_name)
            plt.title(f' {ts_name} actual and forecasted for {forecast_period} days')
            plt.legend()
            plt.show()

        return r2_1, r2_2
    
    except Exception as e:
        print(e)
        print('try another period')    
    
    
    
    
# 5. functions for RNN modelling
# -- functions used in S6_RNN_Basic.ipynb and S7_RNN_FS.ipynb      
    
    
def add_period_cols(df):
    """
    add to given `df` columns with month, day of week
    and day of month based on index dates
        
    Parameters
    ------------------------------------------------
    `df`: pd.DataFrame
        
    Returns
    -------
    pd.Dataframe
    """
    
    df_new = df.copy()
    df_new['month'] = df_new.index.month
    df_new['day_of_week'] = df_new.index.dayofweek
    df_new['day_of_month'] = df_new.index.day
    return df_new

def eda_plots(df, target):
    """
    create and display plots for EDA for `df[target]`:
    - daily time serie and rolling average monthly time serie
    - average daily prices in different months
    - average daily prices in different days of month
    - average daily prices in different days of week
        
    Parameters
    ------------------------------------------------
    `df`: pd.DataFrame
    `target`: STR (name of target column)
        
    Returns
    -------
    None
    """
    
    fig, ax = plt.subplots(4,1, figsize = (15,12))
    ax[0].set_title(f'composite dynamics for {target} per day and averaged per month')
    sns.lineplot(x=df.index, y=target, data=df, ax=ax[0], label='daily prices')
    df_pm = pd.DataFrame(df[target].resample('M').mean())
    sns.lineplot(x=df_pm.index, y=target, data=df_pm, ax=ax[0], label = 'monthly averaged prices')
    ax[0].legend(loc='best')
    ax[1].set_title('average daily prices in different months')
    ax[2].set_title('average daily prices in different days of the week')
    ax[3].set_title('average daily prices in different days in the month')
    sns.pointplot(data=df, x='month', y=target, ax=ax[1])
    sns.pointplot(data=df, x='day_of_week', y=target, ax=ax[2])
    sns.pointplot(data=df, x='day_of_month', y=target, ax=ax[3])
    plt.tight_layout()
    plt.show()

    
def split_to_train_test(df, test_portion):
    """
    split dataframe to train and test subsets based on
    test portion with taking to account datetime sequence
        
    Parameters
    ------------------------------------------------------
    `df`: pd.DataFrame
    `test_portion`: FLOAT
        
    Returns
    -------
    tuple with splitted pd.Dataframes (train and test)
    """
    
    
    test_size = int(len(df)*test_portion)
    train_size = len(df) - test_size
    train, test = df.iloc[0:train_size], df.iloc[train_size:]
    return train, test


def scale_data(scaler_class, train_dataset, test_dataset, target_name, show_stats=False):
    """
    Takes `scaler_class`, `train_dataset` and `test_dataset`
    as inputs and return `train_scaled`, `test_scaled` for
    given `target_name` column and fitted scalers for 
    features (`scaler`) and target (`scaler_target`)
    
    If `show_stats`=True, display describe stats for
    generated scaled dataframes
    
    Parameters
    -------------------------------------------------------
    `scaler_class`: sklearn.preprocessing scaler object
    `train_dataset`, `test_dataset`: pd.Dataframes
    `target_name`: STR (name of target column in df)
    `show_stats`: BOOL
        
    Returns
    -------
    tuple with `train_scaled`, `test_scaled`, 
    `scaler`, `scaler_target`
    """
    
    scaler = scaler_class
    scaler.fit(train_dataset.to_numpy())
       
    train_scaled = pd.DataFrame(scaler.transform(train_dataset.to_numpy()), 
                                                 columns=train_dataset.columns,
                                                 index=train_dataset.index)
    test_scaled = pd.DataFrame(scaler.transform(test_dataset.to_numpy()), 
                                                 columns=test_dataset.columns,
                                                 index=test_dataset.index)
    
    scaler_target = scaler_class
    scaler_target.fit(np.array(train_dataset[target_name]).reshape(-1,1))
    
    if show_stats:
        print('train dataset statistics')
        display(train_scaled.describe())
        print('test dataset statistics')
        display(test_scaled.describe())
        
    return train_scaled, test_scaled, scaler, scaler_target


def create_tensor(X, y, window_size, show_shape=False):
    """
    based on given `X`, `y` and `window_size` create
    3D-tensor (np.array) for `X` and adjusted to 
    `window_size` `y` (np.array)
    
    Parameters
    ---------------------------------------------------
    `X`: pd.DataFrame (features)
    `y`: pd.Series (target)
    `window_size`: INT
    `show_shape`: BOLL
        
    Returns
    -------
    tuple with np.arrays (Xs, ys)
    """
    
    Xs, ys = [], []
    for i in range(len(X) - window_size):
        value = X.iloc[i: i+window_size].to_numpy()
        Xs.append(value)
        ys.append(y.iloc[i+window_size])
    if show_shape:
        print(f'shape of input tensor: {np.array(Xs).shape}')    
    return np.array(Xs), np.array(ys)


def build_rnn_multiple_lstm(matrix_shape, lstm_units, lstm_activation='relu', use_dropout=True, 
              dropout_rate=0.15, n_internal_lstm=0, add_internal_dense=False, internal_dense_units=128,
              optimizer='adam', loss_function='mse', show_model_layers=False,
              metrics = ['mse']):
    
    """
    Build and compile tensorflow.keras.Sequential object
    based on given params.
    
    If `show_model_layers`=True, display model layers
            
    Parameters
    ---------------------------------------------------------------------
    `matrix_shape`: tuple with shape of tensor (n_feat, window_size)
     - used as input_shape in Layers
    
    `lstm_units`: INT, number of nodes in LSTM-layers
     - in case of multiple LSTM-layers, all layers will have same
     number of nodes
    
    `lstm_activation`: STR, activation function name for LSTM-layers
    
    `use_dropout`: BOOL, if True, add 1 droupout layer after all
     LSTM layers
    
    `dropout_rate`: FLOAT (used in dropout layer)
    
    `n_internal_lstm`: INT, number of additional LSTM-layers added
    to the model
    - affordable numbers: 1 and 2 only
    
    `add_internal_dense`: BOOL, if True, 1 additional FC-layer
    added before final FC-layer
    
    `internal_dense_units`: INT, number of nodes for additional FC-layer
    
    `optimizer`: STR, name of optimizer in model.compile method
    
    `loss_function`: STR, name of loss-function in model.compile method
    
    `metrics`: list of STR, names of metrics
            
    Returns
    -------
    tensorflow.keras.model.Sequential instance
    """
    
    # 1. lstm-type selection
    if n_internal_lstm == 0:
        ret_sequence=False
    else:
        ret_sequence=True
    
    first_lstm_layer = keras.layers.LSTM(units=lstm_units, activation=lstm_activation, 
                                         return_sequences=ret_sequence, input_shape=matrix_shape)
    dropout_layer = keras.layers.Dense(units=internal_dense_units, activation=lstm_activation)
    
    # 2. Model init
    model = keras.Sequential()
    # 3. first layer with dropout
    model.add(first_lstm_layer)
    # 4 Optional LSTM layer
    if n_internal_lstm == 2:
        layer_1 = keras.layers.LSTM(units=lstm_units, activation=lstm_activation, return_sequences=True)
        model.add(layer_1)
        layer_2 = keras.layers.LSTM(units=lstm_units, activation=lstm_activation)
        model.add(layer_2)
    elif n_internal_lstm == 1:
        layer_1 = keras.layers.LSTM(units=lstm_units, activation=lstm_activation)
        model.add(layer_1)
    if n_internal_lstm > 2:
        pass
    if use_dropout:
        model.add(keras.layers.Dropout(rate=dropout_rate))
    # 5 Optional dense layer
    if add_internal_dense:
        model.add(keras.layers.Dense(units=internal_dense_units, activation=lstm_activation))
    # 6. Output dense layer
    model.add(keras.layers.Dense(units=1))
    # 7. Model compilation
    model.compile(loss=loss_function, optimizer=optimizer, metrics = metrics)
    # 8. Model layers
    if show_model_layers:
        pprint(model.layers)
    
    return model


def fit_run_plot(model_instance, X, y, n_epochs, batch_size=8, validation_split=0.1, verbose_mode=0,
                show_chart=True, figsize=(14, 5), display_total_time=True, display_model_summary=False, 
                 return_history=True, use_early_stopping=False):
    
    """
    Run and create `history` object based on compiled `model_instance`,
    given `X` and `y` datasets and other params    
    
    If `show_chart`=True, show learning curves for train and valid
    
    If `display_total_time`=True, print total time for learning
    
    If `display_model_summary`=True, print model.summary()
    
    If `return_history`=False, return model instance instead of history
    
    If `use_early_stopping`=True, use EarlyStopping callback in fit stage
        
    Parameters
    ---------------------------------------------------------------------
    `model_instance`: compiled tf.keras.Sequential instance    
    `X`: np.array with features (3D-tensor)    
    `y`: np.array with target    
    `batch_size`: INT, size for batches    
    `validation_split`: FLOAT, validation portion in history
    `verbose_mode`: INT, modes as for keras.model.fit() 
    `show_chart`: BOOL
    `figsize`: TUPLE, figure size for learning curves
    `display_total_time`: BOOL
    `display_model_summary`: BOOL
    `return_history`: BOOL
    `use_early_stopping`: BOOL
    ``
    
    Returns
    -------
    history or model object
    """
    
    start_time = time.time()
    
    if use_early_stopping:
        es = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=0.2*n_epochs)
        callbacks = [es]
    else:
        callbacks = None
    
    history = model_instance.fit(X, y, epochs=n_epochs, batch_size=batch_size, validation_split=validation_split, 
                                 shuffle=False, verbose=verbose_mode, callbacks=callbacks)
    if display_total_time:
        print(f'total time for learning: {time.time() - start_time} sec')
    
    if show_chart:
        fig = plt.figure(figsize=figsize)
        plt.plot(history.history['loss'], label='train')
        plt.plot(history.history['val_loss'], label='valid')
        plt.legend()
        plt.show()
    
    if display_model_summary:
        display(model_instance.summary())
    
    if return_history:
        return history
    else:
        return model_instance

    
def forecast_1step_rnn(fitted_model, scaler_target, test_tensor, test_scaled, target_name,
                       plot_chart=False):
    
    """
    Create one-step-predict based on given `fitted_model` and `test_tensor`
    for timeline from `test_tensor`,
    makes inverse transformations of predict and true values
    based on `scaler_target` and return created `test_pred_inverse`
    and `test_true_inverse` np.arrays
    
    If `plot_chart`=True, plot transformed np.arrays
            
    Parameters
    ------------------------------------------------------------------------
    `fitted_model`: tf.keras.Sequential fitted model
    `scaler_target`: sklearn.preprocessing fitted scaler
    `test_tensor`: test np.array in tensor form
    `test_scaled`: initial pd.Dataframe with features
    `target_name`: STR, name of target 
    `plot_chart`: BOOL
    
    Returns
    -------
    tuple of np.arrays (`test_pred_inverse`, `test_true_inverse`)
    """
    
    test_pred = fitted_model.predict(test_tensor)
    test_pred_inverse = scaler_target.inverse_transform(test_pred)
    test_true_inverse = scaler_target.inverse_transform(np.array(test_scaled[target_name]).reshape(-1,1))
    
    window_size = len(test_true_inverse) - len(test_pred_inverse)
    test_true_inverse = test_true_inverse[window_size:]
    
    if plot_chart:
        plt.title(f'1-step forecast for {target_name}')
        plt.plot(test_true_inverse, c='g', label='true') # add xticks
        plt.plot(test_pred_inverse, c='b', label='pred')
        plt.legend()
        plt.show()
        
    return test_pred_inverse, test_true_inverse

def forecast_Nsteps_rnn(N_periods, fitted_model, scaler_target, test_scaled, target_name,
                        plot_chart=True):
    
    """
    Create full-cycle predict based on given `fitted_model` and `test_tensor`
    for given `N_periods`,  makes inverse transformations of predict and 
    true values based on `scaler_target` and return created 
    `test_pred_inverse` and `test_true_inverse` np.arrays
        
    If `plot_chart`=True, plot transformed np.arrays
            
    Parameters
    ----------------------------------------------------------------
    `N_periods`: INT, number of periods for forecast
    `fitted_model`: tf.keras.Sequential fitted model
    `scaler_target`: sklearn.preprocessing fitted scaler
    `test_tensor`: test np.array in tensor form
    `test_scaled`: initial pd.Dataframe with features
    `target_name`: STR, name of target 
    `plot_chart`: BOOL
    
    Returns
    -------
    tuple of np.arrays (`test_pred_inverse`, `test_true_inverse`)
    """
    
    WINDOW_SIZE = 90
    forecast_limit = len(test_scaled) - WINDOW_SIZE
    N_periods = min(N_periods, forecast_limit)
    
    test_first_step = test_scaled.iloc[:WINDOW_SIZE+1]
    forecast_df = test_scaled.iloc[WINDOW_SIZE+1:]
    test_f = test_first_step.copy()
    
    for i in range(N_periods):
        test_tensor, test_target = create_tensor(test_f, test_f[target_name], 
                                                 window_size=WINDOW_SIZE, show_shape=False)
        test_pred = fitted_model.predict(test_tensor)[-1][0]
        new_row = pd.DataFrame(forecast_df.iloc[i]).T
        new_row[target_name] = test_pred
        test_f = pd.concat([test_f, new_row])
    
    test_f = test_f.iloc[WINDOW_SIZE+1:]
    
    test_pred = np.array(test_f[target_name]).reshape(-1,1)
    test_pred_inverse = scaler_target.inverse_transform(test_pred)
    test_true_inverse = scaler_target.inverse_transform(np.array(test_scaled[target_name]).reshape(-1,1))
    shift_size = len(test_true_inverse) - len(test_pred_inverse)
    test_true_inverse = test_true_inverse[shift_size:]
    if plot_chart:
        plt.title(f'1-step forecast for {target_name}')
        plt.plot(test_true_inverse, c='g', label='true')
        plt.plot(test_pred_inverse, c='b', label='pred')
        plt.legend()
        plt.show()
        
    return test_pred_inverse, test_true_inverse

def create_binnary_vector(array):
    """
    takes LIST or np-array and return LIST with dummy
    values, which are 1 if change of current and 
    previous item in array positive otherwise 0
    
    Parameters
    ------------------------------------------------
    `array`: np.array or list
        
    Returns
    -------
    LIST
    """
    
    if type(array) != list:
        if len(array.shape) == 2 and array.shape[1] == 1:
            array = array.reshape(1,-1)
            array = array[0].tolist()
        elif len(array.shape) == 2 and array.shape[0] == 1:
            array = array[0].tolist()
    
    cl_array = pd.Series(array).diff(1)
    cl_array = [1 if x>0 else 0 for x in cl_array]
    return cl_array    
    
    
def plot_corr_map(corr_matrix):
    """
    plot given correlation matrix in heatmap style
    
    Parameters
    ------------------------------------------------
    `corr_matrix`: pd.DataFrame
        
    Returns
    -------
    None
    """
    fig = plt.figure(figsize=(8,6))
    ax = plt.axes()
    ax.set_title('HEATMAP for Pearson correllation coef')
    sns.heatmap(corr_matrix, square=True,ax=ax,cmap='PuBu');    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
target_cols_is = ['date', 'Revenue', 'Interest Expense', 'EBITDA Margin', 'Profit Margin']

target_cols_cf = ['date', 'Operating Cash Flow', 'Capital Expenditure', 'Acquisitions and disposals',
                  'Issuance (repayment) of debt', 'Issuance (buybacks) of shares', 'Dividend payments', 'Free Cash Flow']


