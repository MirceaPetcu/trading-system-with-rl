
# testate pe un RecurentPPO MlpLstmPolicy cu 20000 timesteps cu acelasi seed si cu parametrii default

# 0.8215623248506755 , cele mai mari profituri, kl-divergence :
def reward1(self, action):
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
            if current_price > last_trade_price:
                step_reward += 1
            elif current_price < last_trade_price:
                step_reward -= 1
            else:
                step_reward -= 0.1
        else:
            if last_trade_price > current_price:
                step_reward += 1
            elif last_trade_price < current_price:
                step_reward -= 1
            else:   
                step_reward -= 0.1
        
    return step_reward


# corelatie 0.04885196842947155, profituri ok,corelatie slaba, kl-divergence: 
def reward2(self,action):
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
            if current_price > last_trade_price:
                step_reward += 1
            elif current_price < last_trade_price:
                step_reward -= 1
            else:
                step_reward -= 0.1
        else:
            if last_trade_price > current_price:
                step_reward += 1
            elif last_trade_price < current_price:
                step_reward -= 1
            else:   
                step_reward -= 0.1
    
    
    # daca era pe Long si a mai cumparat dar ziua urmatoare creste pretul tot e bine
    # daca era pe Short si a mai vandut dar ziua urmatoare scade pretul tot e bine
    # invers e naspa si scade reward-ul
    if not trade:
        try:
            next_price = self.prices[self._current_tick + 1]
            if self._current_tick == self._end_tick:
                pass
            elif action == Actions.Buy.value and self._position.value == Positions.Long.value:
                if (next_price > current_price):
                    step_reward += 0.5
                else:
                    step_reward -= 0.5
            elif action == Actions.Sell.value and self._position.value == Positions.Short.value:
                if (current_price > next_price):
                    step_reward += 0.5
                else:
                    step_reward -= 0.5
        except Exception as e:
            step_reward += 0
    

# kl-divergence: 
def reward3(self, action):
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
            if current_price > last_trade_price:
                step_reward += 1
            elif current_price < last_trade_price:
                step_reward -= 1
            else:
                step_reward -= 0.1