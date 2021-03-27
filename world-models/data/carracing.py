"""
Generating data from the CarRacing gym environment.
!!! DOES NOT WORK ON TITANIC, DO IT AT HOME, THEN SCP !!!
"""
import argparse
from os import makedirs
from os.path import join, exists
import gym
import numpy as np

from utils.misc import sample_continuous_policy


import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
# matplotlib.use('Agg')
import datetime

import sys
sys.path.append("..")
from finrl.config import config
from finrl.marketdata.yahoodownloader import YahooDownloader
from finrl.preprocessing.preprocessors import FeatureEngineer
from finrl.preprocessing.data import data_split
from finrl.env.env_stocktrading import StockTradingEnv
from finrl.model.models import DRLAgent
# from finrl.trade.backtest import backtest_stats, backtest_plot, get_daily_return, get_baseline



import itertools




def generate_data(rollouts, data_dir, noise_type): # pylint: disable=R0914
    """ Generates data """
    assert exists(data_dir), "The data directory does not exist..."


    df = YahooDownloader(start_date = '2009-01-01',
                        end_date = '2021-01-01',
                       ticker_list = ['AAPL']).fetch_data()

    df.sort_values(['date','tic'],ignore_index=True)

    fe = FeatureEngineer(
                        use_technical_indicator=True,
                        tech_indicator_list = config.TECHNICAL_INDICATORS_LIST,
                        use_turbulence=True,
                        user_defined_feature = False)

    processed = fe.preprocess_data(df)

    
    list_ticker = processed["tic"].unique().tolist()
    list_date = list(pd.date_range(processed['date'].min(),processed['date'].max()).astype(str))
    combination = list(itertools.product(list_date,list_ticker))

    processed_full = pd.DataFrame(combination,columns=["date","tic"]).merge(processed,on=["date","tic"],how="left")
    processed_full = processed_full[processed_full['date'].isin(processed['date'])]
    processed_full = processed_full.sort_values(['date','tic'])

    processed_full = processed_full.fillna(0)


    processed_full.sort_values(['date','tic'],ignore_index=True)

    train = data_split(processed_full, '2009-01-01','2019-01-01')
    trade = data_split(processed_full, '2019-01-01','2021-01-01')
    stock_dimension = len(train.tic.unique())
    state_space = 1 + 2*stock_dimension + len(config.TECHNICAL_INDICATORS_LIST)*stock_dimension
    env_kwargs = {
                "hmax": 100, 
                    "initial_amount": 1000000, 
#                         "buy_cost_pct": 0.001i,
#                             "sell_cost_pct": 0.001,
                             "transaction_cost_pct": 0.001, 
                                "state_space": state_space, 
                                    "stock_dim": stock_dimension, 
                                        "tech_indicator_list": config.TECHNICAL_INDICATORS_LIST, 
                                            "action_space": stock_dimension, 
                                                "reward_scaling": 1e-4
                                                }

    e_train_gym = StockTradingEnv(df = train, **env_kwargs)
    env_train, _ = e_train_gym.get_sb_env()

    env = env_train

#     env = gym.make("CarRacing-v0")

    seq_len = 10000

    for i in range(rollouts):

        env.reset()

#         env.env.viewer.window.dispatch_events()
        if noise_type == 'white':
            a_rollout = [env.action_space.sample() for _ in range(seq_len)]
        elif noise_type == 'brown':
            a_rollout = sample_continuous_policy(env.action_space, seq_len, 1. / 50)

        s_rollout = []
        r_rollout = []
        d_rollout = []


        t = 0
        while True:
            action = a_rollout[t]
            t += 1

            s, r, done, _ = env.step(action)
#             env.env.viewer.window.dispatch_events()
            s_rollout += [s]
            r_rollout += [r]
            d_rollout += [done]
            if done:
                print("> End of rollout {}, {} frames...".format(i, len(s_rollout)))
                np.savez(join(data_dir, 'rollout_{}'.format(i)),
                         observations=np.array(s_rollout),
                         rewards=np.array(r_rollout),
                         actions=np.array(a_rollout),
                         terminals=np.array(d_rollout))
                break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--rollouts', type=int, help="Number of rollouts")
    parser.add_argument('--dir', type=str, help="Where to place rollouts")
    parser.add_argument('--policy', type=str, choices=['white', 'brown'],
                        help='Noise type used for action sampling.',
                        default='brown')
    args = parser.parse_args()
    makedirs(args.dir, exist_ok=True)
    generate_data(args.rollouts, args.dir, args.policy)
