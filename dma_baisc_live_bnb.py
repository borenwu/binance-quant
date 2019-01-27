"""
    双均线基准策略
    - 实盘交易
"""

# import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from catalyst import run_algorithm
from catalyst.api import record, symbol, order_target, order
from logbook import Logger


# 需要先加载数据
# catalyst ingest-exchange -x binance -i btc_usdt -f minute
# catalyst ingest-exchange -x binance -i eth_usdt -f minute
# catalyst ingest-exchange -x binance -i ltc_usdt -f minute
# catalyst ingest-exchange -x binance -i eos_usdt -f minute


NAMESPACE = 'dma_basic_live'
log = Logger(NAMESPACE)

# SHORT_WIN = 15               # 短周期窗口
# LONG_WIN = 100               # 长周期窗口

# SHORT_WIN = 50               # 短周期窗口
# LONG_WIN = 200               # 长周期窗口
# TRADE_WIN = 15            # 每个交易周期包含的分钟数 7%

SHORT_WIN = 30               # 短周期窗口
LONG_WIN = 240               # 长周期窗口
TRADE_WIN = 15            # 每个交易周期包含的分钟数 11%

# SHORT_WIN = 30               # 短周期窗口
# LONG_WIN = 240               # 长周期窗口
# TRADE_WIN = 30            # 每个交易周期包含的分钟数 9.8%

# SHORT_WIN = 30               # 短周期窗口
# LONG_WIN = 360               # 长周期窗口
# TRADE_WIN = 15            # 每个交易周期包含的分钟数 3.5%

# SHORT_WIN = 30               # 短周期窗口
# LONG_WIN = 360               # 长周期窗口
# TRADE_WIN = 30            # 每个交易周期包含的分钟数 11%


def get_available_cash(context, use_compound_interest=False):
    """
        获取当前可用资金
        use_compound_interest: 是否使用复利
    """
    if use_compound_interest:
        # 使用复利
        available_cash = context.portfolio.cash
    else:
        available_cash = min(context.portfolio.starting_cash, context.portfolio.cash)
    return available_cash


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

# symbol('eth_usdt'),
# symbol('ltc_usdt'),
# symbol('eos_usdt')

def initialize(context):
    """
        初始化
    """
    log.info('策略初始化')
    context.i = 0                       # 经历过的交易周期
    # 设置加密货币池
    context.asset_pool = [symbol('bnb_usdt')]
    context.set_commission(maker=0.001, taker=0.001)    # 设置手续费
    context.set_slippage(slippage=0.001)                # 设置滑点


def handle_data(context, data):
    """
        在每个交易周期上运行的策略
    """
    context.i += 1  # 记录交易周期
    if context.i < LONG_WIN + 1:
        # 如果交易周期过短，无法计算均线，则跳过循环
        log.warning('交易周期过短，无法计算指标')
        return

    if context.i % TRADE_WIN != 0:
        return

    # 获取当前周期内有效的加密货币
    context.available_asset_pool = [asset
                                    for asset in context.asset_pool
                                    if asset.start_date <= data.current_dt]

    context.up_cross_signaled = set()   # 初始化金叉的交易对集合
    context.down_cross_signaled = set()  # 初始化死叉的交易对集合

    for asset in context.available_asset_pool:
        # 遍历每一个加密货币对
        # 获得历史价格
        frequency = '{}T'.format(TRADE_WIN)     # '5T'
        hitory_data = data.history(asset,
                                   'close',
                                   bar_count=LONG_WIN + 1,
                                   frequency=frequency,
                                   )
        if len(hitory_data) >= LONG_WIN + 1:
            # 保证新的货币有足够的时间计算均线
            # 计算双均线
            short_avgs = hitory_data.rolling(window=SHORT_WIN).mean()
            long_avgs = hitory_data.rolling(window=LONG_WIN).mean()

            # 双均线策略
            # 短期均线上穿长期均线
            if (short_avgs[-2] < long_avgs[-2]) and (short_avgs[-1] >= long_avgs[-1]):
                # 形成金叉
                context.up_cross_signaled.add(asset)

            # 短期均线下穿长期均线
            if (short_avgs[-2] > long_avgs[-2]) and (short_avgs[-1] <= long_avgs[-1]):
                # 形成死叉
                context.down_cross_signaled.add(asset)

    # 卖出均线死叉信号的持仓交易对
    for asset in context.portfolio.positions:
        if asset in context.down_cross_signaled:
            order_target(asset, 0)

    # 买入均线金叉信号的持仓股
    for asset in context.up_cross_signaled:
        if asset not in context.portfolio.positions:
            close_price = data.current(asset, 'close')

            available_cash = get_available_cash(context)
            if available_cash > 0:
                # 如果有可用现金
                # 每个交易对平均分配现金
                cash_for_each_asset = available_cash / len(context.available_asset_pool)

                amount_to_buy = cash_for_each_asset / close_price    # 计算购买的数量
                if amount_to_buy >= asset.min_trade_size:
                    # 购买的数量大于最小购买数量
                    order(asset, amount_to_buy)

    # 持仓比例
    pos_level = context.portfolio.positions_value / context.portfolio.portfolio_value

    # 记录每个交易周期的现金
    record(cash=context.portfolio.cash, pos_level=pos_level)

    # 输出信息
    log.info('日期：{}，资产：{:.2f}，持仓比例：{:.6f}%，持仓产品：{}'.format(
        data.current_dt, context.portfolio.portfolio_value, pos_level * 100,
        ', '.join([asset.asset_name for asset in context.portfolio.positions]))
    )


def analyze(context, perf):
    # 保存交易记录
    perf.to_csv('./dma_baisc_live_performance.csv')

    # 获取交易所的计价货币
    exchange = list(context.exchanges.values())[0]
    quote_currency = exchange.quote_currency.upper()

    # 图1：可视化资产值
    # ax1 = plt.subplot(311)
    # perf['portfolio_value'].plot(ax=ax1)
    # ax1.set_ylabel('Portfolio Value\n({})'.format(quote_currency))
    # start, end = ax1.get_ylim()
    # ax1.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # 图2：可视化仓位
    # ax2 = plt.subplot(312)
    # perf['pos_level'].plot(ax=ax2)
    # ax2.set_ylabel('Position Level')
    # start, end = 0, 1
    # ax2.yaxis.set_ticks(np.arange(start, end, (end - start) / 5))

    # 图3：可视化现金数量
    # ax3 = plt.subplot(313, sharex=ax1)
    # perf['cash'].plot(ax=ax3)
    # ax3.set_ylabel('Cash\n({})'.format(quote_currency))
    # start, end = ax3.get_ylim()
    # ax3.yaxis.set_ticks(np.arange(0, end, end / 5))

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
        # 回测
        run_algorithm(
            live=False,
            simulate_orders=True,
            capital_base=100,
            data_frequency='minute',
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            quote_currency='usdt',
            start=pd.to_datetime('2019-01-01', utc=True),
            end=pd.to_datetime('2019-01-25', utc=True)
        )
    elif run_mode == 'paper trading':
        # 实盘模拟
        run_algorithm(
            live=True,
            simulate_orders=True,
            capital_base=100,
            initialize=initialize,
            handle_data=handle_data,
            analyze=analyze,
            exchange_name='binance',
            algo_namespace=NAMESPACE,
            quote_currency='usdt'
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
