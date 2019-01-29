import pandas as pd
import matplotlib.pyplot as plt

from catalyst import run_algorithm
from catalyst.api import order, record, symbol


def initialize(context):
    """
        初始化
    """
    context.asset = symbol('btc_usdt')


def handle_data(context, data):
    """
        循环运行策略
    """
    # 每个交易周期买入1个比特币
    order(context.asset, 1)

    # 记录每个交易周期的比特币价格
    record(btc=data.current(context.asset, 'price'))


def analyze(context, perf):
    """
        策略分析
    """
    # print(perf.portfolio_value)
    # 每日资产
    ax1 = plt.subplot(211)
    perf.portfolio_value.plot(ax=ax1)
    ax1.set_ylabel('portfolio value')

    # 比特币价格
    ax2 = plt.subplot(212, sharex=ax1)
    perf.btc.plot(ax=ax2)
    ax2.set_ylabel('bitcoin price')
    plt.show()


if __name__ == '__main__':
    run_algorithm(
        capital_base=10000,
        data_frequency='daily',
        initialize=initialize,
        handle_data=handle_data,
        analyze=analyze,
        exchange_name='binance',
        quote_currency='usdt',
        start=pd.to_datetime('2018-01-01', utc=True),
        end=pd.to_datetime('2018-10-01', utc=True)
    )
