# TRADING SYSTEM WITH REINFORCEMENT LEARNING

## Project Description
The proposed project aims to develop a trading system based on Reinforcement Learning. The trading system will make daily decisions to buy or sell, depending on the price movements and technical indicators of financial assets, in order to maximize profit over periods of at least 30 days.

## Implementation Methods
To implement the trading system, I trained agents using Deep Reinforcement Learning. Since the states are numerous (prices and indicators are real, fluctuating numbers, resulting in an extremely high number of states that cannot be stored in a tabular format), they cannot be stored in a matrix format. Thus, our project presents Function Approximation algorithms. The financial world is influenced by many external factors and is also very dynamic and noisy. Our intuition is that a stochastic policy will perform better for the presented problem. Not every given action is the best considering the information available to the system, so this type of policy better manages the trade-off between exploration and exploitation. Therefore, in light of a more drastic change in the market, the system can adapt more effectively. Our intuition proves to be correct and will be demonstrated in the results section.

## Data Used
For training, we used a dataset from Google, collected by the author of the environment we used, which contains 2,335 examples from the period 2009-2018. The data format includes the date of the recorded price, opening and closing prices, closing price after dividends, the lowest and highest price of the day, and the volume of shares traded that day. For testing, we obtained several datasets from Yahoo Finance, from 2015 (for companies listed on the stock exchange up to and including 2015) to 2023 (01.01.2015-31.12.2023), following the same format as the training data. The assets for which we obtained test data include: Tesla, Ethereum, Bitcoin, Google, Amazon, Apple, Microsoft, Netflix, Spotify, PayPal, Nike, Disney, AMD, Intel, Nvidia, IBM, Alibaba, Uber, Airbnb.

## Environment Description
The trading environment is defined using a customized version of the Anytrading Gym trading environment, specifically the stockenv environment from this framework. We maintained the profit calculation (calculated as a percentage), positions (long when the agent buys low to sell high, and short when selling high to buy low), actions (buying and selling), window size (the number of days in an observation/state), and a selling commission of 1% of the sale price and a buying commission of 0.5% as requested by the trading platform. Our modifications are as follows:

### Defining the Environment State
The state where the agent can be is represented by: price (closing price, standard in this context), volume of shares traded, opening price, closing price, lowest and highest price, Triple Exponential Moving Average (TEMA 30, indicating the trend direction over 30 days, with greater emphasis on recent days. Making buying and selling decisions daily, we decided to place greater emphasis on the last trading days), Kaufman Efficiency Indicator (ER, an indicator for trend confirmation, ranges between -1 and 1, -1 = downtrend, 1 = uptrend, approx. 0 = random), Relative Strength Index (RSI, an indicator for trend confirmation, ranges from 0 to 100, 30-70 = area of interest, 0-30 = oversold, 70-100 = overbought), On-Balance Volume (OBV, an indicator for trend confirmation based on the volume of trades that day), Stochastic Oscillator (STOCH, an indicator for trend confirmation, ranges between 0 and 100, 20-80 = area of interest, 0-20 = oversold, 80-100 = overbought), all expressed at the daily level.

### Calculating Rewards
To calculate rewards, we tried several approaches:
1. The original reward from the environment, calculated only when a long trade is made (when the agent buys to sell at a higher price), defined as the difference between the current price and the price on the last purchase day; [figure 1]
2. The original reward, but instead of the price difference, a reward of +1 is granted if the selling price is higher than the last purchase price, and -1 otherwise; [figure 2]
3. Similar to the reward function in point 2, but a reward of +1 is given if on that day the agent bought at a lower price and previously sold at a higher price than the current purchase price, otherwise -1. Additionally, in both cases, 0.1 is deducted if prices are equal; [figure 3]
4. The reward as in point 3, but following the reward shaping principle, intermediate rewards are granted for faster convergence, especially helpful at the beginning when the agent is not very intelligent, having a trajectory that does not yield very good rewards and when final rewards are quite rare. Thus, if the agent buys, and previously was long (bought) and the next day's price is higher than the current day's price, it receives a reward of +0.5; otherwise, it receives -0.5. If the agent sells, and previously was short (sold) and the current price is higher than the next day's price, it receives a reward of +0.5; otherwise, it receives -0.5; [figure 4]
5. The reward function number 5 is similar to point 4, but in the case of intermediate rewards, the fixed rewards of +0.5 or -0.5 are no longer given; instead, if the agent is long and buys that day, it will receive the difference between the next day's price and the current price, and if the agent is short and sells that day, it will receive the difference between the current price and the next day's price. [figure 5]

#### Evaluating Reward Functions
We tested each reward function using a recurrent Proximal Policy Optimization model (which will be detailed in the agents section), over a total of 20,000 timesteps with default parameters, across 100 episodes, on the training dataset (80% of the dataset was used for training, and 20% for evaluation—approximately 450 days). We obtained the following results, which will be displayed in the graphs below. As observed, the best reward that generates good profits while also correlating profit and reward is from point (5). This is evident both empirically and intuitively, as we applied the concept of reward shaping, and final rewards are normalized. Even though intermediate rewards are not normalized, they are not substantial enough to negatively impact the algorithms, introducing a stochastic component to move past local optima.

![Alt text](/plots/original_reward.png)

*figure 1*

![Alt text](/plots/trade_normal_doar_long.png)

*figure 2*

![Alt text](/plots/trade.png)

*figure 3*

![Alt text](/plots/trade_si_non_trade.png)

*figure 4*

![Alt text](/plots/trade_normal_nontrade_nenormal.png)

*figure 5*

## Agents
### DQN
For implementing the trading system, I used two algorithms. 
The first algorithm is Deep Q Network, where instead of a classic MLP or a convolutional network, I used a network formed of two layers of Long Short Term Memory (LSTM), with a hidden size of 64, one fully connected layer of hidden size 64, and a dropout of 0.2 for both LSTM and fully connected layers, as the data is sequential and the order in which it is presented is important. I used a gamma of 0.95, epsilon of 0.99, a minimum epsilon of 0.05, epsilon decay of 0.9 / 100,000, and a learning rate of 0.001. The algorithm uses a replay buffer, with a minimum number of states of 50,000 to start training and a maximum memory size of 1,000,000 states. I trained over 400 episodes. The algorithm follows the structure from the lab, with our modification being the introduction of recurrent networks.

### PPO
The second algorithm I applied to the environment is a recurrent version of the Proximal Policy Optimization (PPO) algorithm, implemented in stable_baselines3 (the contributed version). 
The presented algorithm is a simplified version of the Trust Region Policy Optimization (TRPO). TRPO is a more stable adjustment of the Vanilla Policy Gradient algorithm. PPO is a policy-based algorithm (it directly assists the policy, not through a value function or action value function) and on-policy (as it samples trajectories from its own policy and chooses the optimal action based on its policy). This algorithm assumes a stochastic policy, which greatly aids us in our situation. I used the actor-critic variant of the algorithm to combine the advantages of policy-based methods with those of value-based methods, bringing various benefits, such as faster convergence and stability, though increasing complexity. 
Initially, the algorithm samples its trajectories and calculates the advantage for each timestamp, based on the information available at that moment. This information, in the case of the actor-critic variant, represents the difference between discounted rewards (calculated starting from the current timestamp) and the value function in the current state (calculated by the model's critic network). As the algorithm is based on policy gradients, it uses part of its loss function: $E_t[\log \pi_\theta (a_t|s_t) \cdot A_t]$, where $\pi_\theta$ is the policy, $a_t$ is the action, $s_t$ is the state, and $A_t$ is the advantage. This function can be interpreted as follows: when the gradient is positive, the probability

 of that action is increased, while when it is negative, it is decreased. This allows us to adapt the algorithm more effectively to our environment.

## Performance of the System
In order to validate the performance of the system and to illustrate that the modifications we introduced had a positive impact on the results, we conducted tests using a simple trading algorithm that makes decisions based on the same indicators, buying assets at a low price and selling them when they are higher. We recorded the trading history and the final profit after closing all positions at the end of the test. Below are the results we obtained from the DQN agent and the PPO agent after a series of tests:

### DQN vs Simple Trading System
#### DQN Performance
- Profit: 21,750.34 (approximately 27.2% growth over 30 days)
- Maximum Drawdown: 7.3%
- Days Closed: 18
- Average Daily Profit: 721.5

#### Simple Trading System Performance
- Profit: 6,113.14 (approximately 8.3% growth over 30 days)
- Maximum Drawdown: 7.3%
- Days Closed: 12
- Average Daily Profit: 509.4

### PPO vs Simple Trading System
#### PPO Performance
- Profit: 12,975.74 (approximately 17.4% growth over 30 days)
- Maximum Drawdown: 7.4%
- Days Closed: 11
- Average Daily Profit: 1188.2

#### Simple Trading System Performance
- Profit: 6,113.14 (approximately 8.3% growth over 30 days)
- Maximum Drawdown: 7.3%
- Days Closed: 12
- Average Daily Profit: 509.4

## Conclusion
In conclusion, I would like to present the final results, which demonstrated that the proposed methods outperformed the simple trading algorithm in terms of profit. The main contribution is the definition of the reward function, using value shaping and additional indicators to improve the agent's decision-making in a dynamic environment. Both agents achieved profits that exceeded 17% over 30 days, with DQN performing better in terms of return and days closed.

## Libraries Used
- pandas
- numpy
- matplotlib
- stable-baselines3
- tensorflow
- gym

## To-Do
- Performance improvements
- Add more indicators
- Integrate additional risk measures
- Extend the trading horizon to allow for long-term investments
- Explore the potential of other learning methods such as A3C or TRPO.

## Run Instructions
1. Clone the repository
```bash
git clone <repository_url>
```
2. Navigate to the project directory
```bash
cd <project_directory>
```
3. Install the required libraries
```bash
pip install -r requirements.txt
```
4. Run the main script
```bash
python main.py
``` 

This way, the original structure of the README is preserved while translating the text. If you need further adjustments or formatting, just let me know!
