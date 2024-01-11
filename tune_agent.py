""" Optuna example that optimizes the hyperparameters of
a reinforcement learning agent using A2C implementation from Stable-Baselines3
on a Gymnasium environment.

This is a simplified version of what can be found in https://github.com/DLR-RM/rl-baselines3-zoo.

You can run this example as follows:
    $ python sb3_simple.py

"""
from typing import Any
from typing import Dict

import gymnasium
import optuna
from optuna.pruners import MedianPruner
from optuna.samplers import TPESampler
from stable_baselines3 import A2C
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.monitor import Monitor
import torch
import torch.nn as nn
from sb3_contrib import RecurrentPPO
from environment import CustomTradingEnv
from eda import get_data
from stable_baselines3 import PPO

N_TRIALS = 100
N_STARTUP_TRIALS = 0
N_EVALUATIONS = 2
N_TIMESTEPS = int(2e4)
EVAL_FREQ = int(N_TIMESTEPS / N_EVALUATIONS)
N_EVAL_EPISODES = 3


def create_environment() -> gymnasium.Env:
    df = get_data()
    env = CustomTradingEnv(df=df, frame_bound=(30, int(0.8*len(df))), window_size=30)
    return env

DEFAULT_HYPERPARAMS = {
    "policy": "MlpLstmPolicy",
    "env": create_environment()
}

initial_hyperparameters = {
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "learning_rate": 3e-4,
    # "ent_coef": 0.0,
    "max_grad_norm": 0.5,
    "vf_coef": 0.5,
    "clip_range": 0.2,
    "policy_kwargs": {
        "lstm_hidden_size": 256,
        "n_lstm_layers": 1,
    }
}

# evaluate the agent
from environment import CustomTradingEnv
def evaluate_agent(model,episodes=10):
    df = get_data()
    test_env = CustomTradingEnv(df=df, frame_bound=(int(0.8*len(df)), len(df)), window_size=30)
    mean_reward, mean_profit = 0, 0
    for episode in range(episodes):
        state = test_env.reset()[0]
        while True:
            action, x = model.predict(state)
            state, reward, deprecated,done, info = test_env.step(action)
            if done:
                print(info)
                break
        mean_profit += info['total_profit']
        mean_reward += info['total_reward']
        
    mean_reward /= episodes
    mean_profit /= episodes
    
    test_env.close()
    return mean_reward, mean_profit


def sample_reccurent_ppo_params(trial: optuna.Trial) -> Dict[str, Any]:
    """Sampler for RecurrentPPO hyperparameters."""
    gamma = trial.suggest_float("gamma", 0.9, 0.9999, log=True)
    max_grad_norm = trial.suggest_float("max_grad_norm", 0.3, 5.0, log=True)
    gae_lambda = trial.suggest_float("gae_lambda", 0.8, 0.999, log=True)
    learning_rate = trial.suggest_float("learning_rate", 1e-5, 1, log=True)
    ent_coef = trial.suggest_float("ent_coef", 0.00000001, 0.1, log=True)
    clip_range = trial.suggest_float("clip_range", 0.1, 0.4, log=True)
    vf_coef = trial.suggest_float("vf_coef", 0.1, 0.9, log=True)
    lstm_hidden_size = trial.suggest_categorical("lstm_hidden_size", [32, 64, 128, 256, 512,1024])
    n_lstm_layers = trial.suggest_categorical("n_lstm_layers", [1, 2, 3, 4, 5, 6])
    
    # Display true values.
    trial.set_user_attr("gamma_", gamma)
    trial.set_user_attr("gae_lambda_", gae_lambda)
    trial.set_user_attr("learning_rate", learning_rate)
    # trial.set_user_attr("ent_coef", ent_coef)
    trial.set_user_attr("max_grad_norm", max_grad_norm)
    trial.set_user_attr("vf_coef", vf_coef)
    trial.set_user_attr("clip_range", clip_range)
    



    return {
        "gamma": gamma,
        "gae_lambda": gae_lambda,
        "learning_rate": learning_rate,
        "ent_coef": ent_coef,
        "max_grad_norm": max_grad_norm,
        "vf_coef": vf_coef,
        "clip_range": clip_range,
        "policy_kwargs": {
            "lstm_hidden_size": lstm_hidden_size,
            "n_lstm_layers": n_lstm_layers,
        }
    }


def objective(trial: optuna.Trial) -> float:
    kwargs = DEFAULT_HYPERPARAMS.copy()
    # Sample hyperparameters.
    kwargs.update(sample_reccurent_ppo_params(trial))
    # Create the RL model.
    model = RecurrentPPO(**kwargs)
    

    nan_encountered = False
    try:
        print('model training')
        model.learn(N_TIMESTEPS)
        # Evaluate for 10 episodes.
        mean_reward, mean_profit = evaluate_agent(model,5)
    except AssertionError as e:
        # Sometimes, random hyperparams can generate NaN.
        print(e)
        nan_encountered = True
    finally:
        # Free memory.
        model.env.close()
        # eval_env.close()

    # Tell the optimizer that the trial failed.
    if nan_encountered:
        return float("nan")


    return mean_profit


if __name__ == "__main__":
    # Set pytorch num threads to 1 for faster training.
    torch.set_num_threads(1)

    sampler = TPESampler(n_startup_trials=N_STARTUP_TRIALS)

    study = optuna.create_study(sampler=sampler, direction="maximize", storage="sqlite:///db.sqlite3",  # Specify the storage URL here.
        study_name="recurrent-ppo-study",load_if_exists=True)
    study.enqueue_trial(initial_hyperparameters)
    try:
        study.optimize(objective, n_trials=N_TRIALS)
    except KeyboardInterrupt:
        pass

    print("Number of finished trials: ", len(study.trials))

    print("Best trial:")
    trial = study.best_trial

    print("  Value: ", trial.value)

    print("  Params: ")
    for key, value in trial.params.items():
        print("    {}: {}".format(key, value))

    print("  User attrs:")
    for key, value in trial.user_attrs.items():
        print("    {}: {}".format(key, value))