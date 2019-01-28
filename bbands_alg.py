# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import talib

from catalyst import run_algorithm
from catalyst.api import record, symbol, order_target_percent, order_percent
from catalyst.exchange.utils.stats_utils import extract_transactions
from logbook import Logger

# 需要先加载数据
# catalyst ingest-exchange -x binance -i btc_usdt -f daily

NAMESPACE = 'bbands_alg'
log = Logger(NAMESPACE)

NAMESPACE = 'bollinger_bands'
SIGNAL_BUY = 'buy'      # 买入信号
SIGNAL_SELL = 'sell'    # 卖出信号
SIGNAL_INIT = ''        # 观望信号

BOLL_N = 15            # BBands参数n
BOLL_M = 2              # BBands参数m
TRADE_WIN = 60

def get_risk_indices(perf):
    """
        计算风险指标，包括：
        1. 策略收益
        2. 策略年化收益
        3. 策略波动率
        4. 夏普比率
        5. 最大回撤
    """
    # 策略执行天数
    n = len(perf)

    # 1. 策略收益
    total_returns = perf.iloc[-1]['algorithm_period_return']

    # 2. 策略年化收益
    total_ann_returns = (1 + total_returns) ** (250 / n) - 1

    # 3. 策略波动率
    algo_volatility = perf.iloc[-1]['algo_volatility']

    # 4. 夏普比率
    sharpe = perf.iloc[-1]['sharpe']

    # 5. 最大回撤
    max_drawdown = np.abs(perf.iloc[-1]['max_drawdown'])

    return total_returns, total_ann_returns, algo_volatility, sharpe, max_drawdown

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

    frequency = '{}T'.format(TRADE_WIN)  # '5T'
    # 获得历史价格
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
    # 输出信息
    print('日期：{}，价格：{:.4f}，资产：{:.2f}，持仓量：{:.8f}, {}'.format(
        data.current_dt, price, context.portfolio.portfolio_value, pos_amount, context.signal))

    # 进行下一次交易前重置交易信号
    context.signal = SIGNAL_INIT


def analyze(context, perf):
    # 保存交易记录
    perf.to_csv('./bbands_performance.csv')

    # 获取交易所的计价货币
    exchange = list(context.exchanges.values())[0]
    quote_currency = exchange.quote_currency.upper()

    # # 图1：可视化资产值
    # ax1 = plt.subplot(411)
    # perf['portfolio_value'].plot(ax=ax1)
    # ax1.set_ylabel('Portfolio Value\n({})'.format(quote_currency))
    # start, end = ax1.get_ylim()
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # # 图2：可视化货币价格，布林带和买入卖出点
    # ax2 = plt.subplot(412, sharex=ax1)
    # perf[['price', 'lower', 'middle', 'upper']].plot(ax=ax2)
    # ax2.set_ylabel('{asset}\n({quote})'.format(
    #     asset=context.asset.symbol,
    #     quote=quote_currency
    # ))
    # start, end = ax2.get_ylim()
    # ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # # 提取交易时间点
    # transaction_df = extract_transactions(perf)
    # if not transaction_df.empty:
    #     buy_df = transaction_df[transaction_df['amount'] > 0]   # 买入点
    #     sell_df = transaction_df[transaction_df['amount'] < 0]  # 卖出点
    #     ax2.scatter(
    #         buy_df.index.to_pydatetime(),
    #         perf.loc[buy_df.index, 'price'],
    #         marker='^',
    #         s=100,
    #         c='green',
    #         label=''
    #     )
    #     ax2.scatter(
    #         sell_df.index.to_pydatetime(),
    #         perf.loc[sell_df.index, 'price'],
    #         marker='v',
    #         s=100,
    #         c='red',
    #         label=''
    #     )
    #
    # # 图3：比较价格变化率和资产变化率
    # ax3 = plt.subplot(413, sharex=ax1)
    # perf[['algorithm_period_return', 'price_change']].plot(ax=ax3)
    # ax3.set_ylabel('Percent Change')
    # start, end = ax3.get_ylim()
    # ax3.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # # 图4：可视化现金数量
    # ax4 = plt.subplot(414, sharex=ax1)
    # perf['cash'].plot(ax=ax4)
    # ax4.set_ylabel('Cash\n({})'.format(quote_currency))
    # start, end = ax4.get_ylim()
    # ax4.yaxis.set_ticks(np.arange(0, end, end / 5))
    #
    # plt.tight_layout()
    # plt.show()

    # 评价策略
    total_returns, total_ann_returns, algo_volatility, sharpe, max_drawdown = get_risk_indices(perf)
    log.info('策略收益: {:.3f}%, 策略年化收益: {:.3f}%, 策略波动率: {:.3f}%, 夏普比率: {:.3f}, 最大回撤: {:.3f}%'.format(
        total_returns * 100, total_ann_returns * 100, algo_volatility * 100, sharpe, max_drawdown * 100
    ))


if __name__ == '__main__':
    run_modes = ['backtesting', 'paper trading', 'live']
    run_mode = run_modes[2]

    if run_mode == 'backtesting':
        run_algorithm(
            capital_base=100,
            data_frequency='minute',
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            quote_currency='usdt',
            start=pd.to_datetime('2019-01-01', utc=True),
            end=pd.to_datetime('2019-01-27', utc=True)
        )

    elif run_mode == 'live':
        # 实盘交易
        run_algorithm(
            live=True,
            simulate_orders=False,
            capital_base=90,
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            quote_currency='usdt'
        )