import numpy as np
import pandas as pd
import talib

from catalyst import run_algorithm
from catalyst.api import record, symbol, order_target_percent

# 需要先加载数据
# catalyst ingest-exchange -x binance -i btc_usdt -f daily

NAMESPACE = 'bollinger_bands'
SIGNAL_BUY = 'buy'      # 买入信号
SIGNAL_SELL = 'sell'    # 卖出信号
SIGNAL_INIT = ''        # 观望信号

TRADE_WIN = 60

def initialize(context):
    """
        初始化
    """
    context.i = 0                       # 经历过的交易周期
    context.asset = symbol('bnb_usdt')  # 交易对
    context.base_price = None           # 初始价格
    context.signal = SIGNAL_INIT  # 交易信号
    context.set_commission(maker=0.001, taker=0.001)    # 设置手续费
    context.set_slippage(slippage=0.001)                # 设置滑点


def handle_data(context, data):
    """
        在每个交易周期上运行的策略
    """
    context.i += 1  # 记录交易周期
    if context.i < BOLL_N + 2:
        # 如果交易周期过短，无法计算BBands，则跳过循环
        return

    # 获得历史价格
    frequency = '{}T'.format(TRADE_WIN)  # '5T'
    hitory_data = data.history(context.asset,
                               'close',
                               bar_count=BOLL_N + 2,
                               frequency=frequency,
                               )
    # 获取当前持仓数量
    pos_amount = context.portfolio.positions[context.asset].amount

    # 计算BBands
    uppers, middles, lowers = talib.BBANDS(hitory_data, timeperiod=BOLL_N, nbdevdn=BOLL_M, nbdevup=BOLL_M)

    # BBands 交易策略
    if (hitory_data[-3] <= lowers[-3]) and (hitory_data[-2] >= lowers[-2]) and pos_amount == 0:
        # K线上穿下轨，买入
        order_target_percent(context.asset, target=1)
        context.signal = SIGNAL_BUY

    if (hitory_data[-3] >= uppers[-3]) and (hitory_data[-2] <= uppers[-2]) and pos_amount > 0:
        # K线下穿上轨，卖出
        order_target_percent(context.asset, target=0)
        context.signal = SIGNAL_SELL

    # 获取当前的价格
    price = data.current(context.asset, 'price')
    if context.base_price is None:
        # 如果没有设置初始价格，将第一个周期的价格作为初始价格
        context.base_price = price

    # 计算价格变化百分比，作为基准
    price_change = (price - context.base_price) / context.base_price

    # 记录每个交易周期的信息
    # 1. 价格, 2. 现金, 3. 价格变化率, 4. 上轨, 5. 中轨，6. 下轨
    record(price=price,
           cash=context.portfolio.cash,
           price_change=price_change,
           lower=lowers[-1],
           middle=middles[-1],
           upper=uppers[-1],
           signal=context.signal)

    # 进行下一次交易前重置交易信号
    context.signal = SIGNAL_INIT


if __name__ == '__main__':
    n_range = np.arange(10, 21, 1)  # BBands参数n候选区间
    m_range = np.arange(1, 2.1, 0.1)  # BBands参数m候选区间

    # 记录参数选择结果
    param_results = pd.DataFrame(columns=['n', 'm', 'portfolio'])

    for n in n_range:
        for m in m_range:
            BOLL_N = int(n)
            BOLL_M = float(m)

            perf = run_algorithm(
                capital_base=1000,
                data_frequency='minute',
                initialize=initialize,
                handle_data=handle_data,
                analyze=None,
                exchange_name='binance',
                algo_namespace=NAMESPACE,
                quote_currency='usdt',
                start=pd.to_datetime('2019-01-01', utc=True),
                end=pd.to_datetime('2019-01-28', utc=True)
            )

            portfolio = perf['portfolio_value'][-1]
            print('n={}, m={:.2f}, portfolio={:.2f}'.format(BOLL_N, BOLL_M, portfolio))
            param_results = param_results.append({'n': BOLL_N, 'm': BOLL_M, 'portfolio': portfolio}, ignore_index=True)

    param_results.to_csv('./bbands_param.csv', index=False)
