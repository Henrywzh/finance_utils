from finance_utils.strategy import Strategy
from finance_utils.backtest import *


class StrategyOptimisation:
    def __init__(self, class_strategy: Strategy):
        self.strategy = class_strategy
        # TODO: Add optimisation to parameters
