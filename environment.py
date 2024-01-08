from gym_anytrading.envs import TradingEnv, Actions, Positions,StocksEnv
from base_environment import BaseEnv


class CustomTradingEnv(BaseEnv):
    def __init__(self, df, window_size, frame_bound,balance=100000, render_mode=None):
        assert len(frame_bound) == 2

        self.frame_bound = frame_bound
        super().__init__(df, window_size,balance, render_mode)

        self.trade_fee_bid_percent = 0.01  # unit
        self.trade_fee_ask_percent = 0.005  # unit
        
    def _process_data(self):
        start = self.frame_bound[0] - self.window_size
        end = self.frame_bound[1]
        prices = self.df.loc[:, 'Close'].to_numpy()[start:end]
        signal_features = self.df.loc[:, ['Close', 'Volume','Open','High','Low','TEMA','ER','RSI','OBV','STOCH']].to_numpy()[start:end]
        return prices, signal_features
  
    # de schimbat
    def _calculate_reward(self, action):
        step_reward = 0
        current_price = self.prices[self._current_tick]
        trade = False
        
        if (
            (action == Actions.Buy.value and self._position.value == Positions.Short.value) or
            (action == Actions.Sell.value and self._position.value == Positions.Long.value)
        ):
            trade = True
        
        
        # foarte bine daca a cumparat cand era mai mic(Long) si a vandut cand era mai mare 
        # si foarte bine ca a vandut cand era mai mare si a cumparat cand era mai mic
        # invers e naspa si scade reward-ul
        if trade:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]
            if self._position.value == Positions.Long.value:
                step_reward += (current_price - last_trade_price)
            else:
                step_reward += (last_trade_price - current_price)
        
        
        # daca era pe Long si a mai cumparat dar ziua urmatoare creste pretul tot e bine
        # daca era pe Short si a mai vandut dar ziua urmatoare scade pretul tot e bine
        # invers e naspa si scade reward-ul
        if not trade:
            try:
                next_price = self.prices[self._current_tick + 1]
                if self._current_tick == self._end_tick:
                    pass
                elif action == Actions.Buy.value and self._position.value == Positions.Long.value:
                    step_reward += (next_price - current_price)
                elif action == Actions.Sell.value and self._position.value == Positions.Short.value:
                    step_reward += (current_price - next_price)
            except Exception as e:
                step_reward += 0
        
        
        if trade:
            try:
                next_price = self.prices[self._current_tick + 1]
                if self._current_tick == self._end_tick:
                    pass
                # trebuia sa mai astepte putin deoaerece pretul a scazut ziua urmatoare
                elif action == Actions.Buy.value and self._position.value == Positions.Short.value:
                    if next_price < current_price:
                        step_reward = step_reward * .95
                    # bonus ca maine crestea pretul
                    else:
                        step_reward = step_reward * 1.05
                # trebuia sa mai astepte putin deoaerece pretul a crescut ziua urmatoare
                elif action == Actions.Sell.value and self._position.value == Positions.Long.value:
                    if next_price > current_price:
                        step_reward = step_reward * .95
                    else:
                        #bonus ca maine scadea pretul
                        step_reward = step_reward * 1.05
                    
            except Exception as e:
                step_reward += 0
                    
            
        return step_reward

    # de schimbat
    def _update_profit(self, action):
        trade = False
        if (
            (action == Actions.Buy.value and self._position.value == Positions.Short.value) or
            (action == Actions.Sell.value and self._position.value == Positions.Long.value)
        ):
            trade = True

        if trade or self._truncated:
            current_price = self.prices[self._current_tick]
            last_trade_price = self.prices[self._last_trade_tick]

            if self._position.value == Positions.Long.value:
                shares = (self._total_profit * (1 - self.trade_fee_ask_percent)) / last_trade_price
                self._total_profit = (shares * (1 - self.trade_fee_bid_percent)) * current_price

    def max_possible_profit(self):
        current_tick = self._start_tick
        last_trade_tick = current_tick - 1
        profit = 1.

        while current_tick <= self._end_tick:
            position = None
            if self.prices[current_tick] < self.prices[current_tick - 1]:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] < self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Short
            else:
                while (current_tick <= self._end_tick and
                       self.prices[current_tick] >= self.prices[current_tick - 1]):
                    current_tick += 1
                position = Positions.Long

            if position == Positions.Long:
                current_price = self.prices[current_tick - 1]
                last_trade_price = self.prices[last_trade_tick]
                shares = profit / last_trade_price
                profit = shares * current_price
            last_trade_tick = current_tick - 1

        return profit
    
    def _update_balance(self,action):
        if action == Actions.Buy.value:
            self.balance -= self.prices[self._current_tick]
            self.balance -= self.prices[self._current_tick] * self.trade_fee_ask_percent
        elif action == Actions.Sell.value:
            self.balance += self.prices[self._current_tick]
            self.balance -= self.prices[self._current_tick] * self.trade_fee_bid_percent