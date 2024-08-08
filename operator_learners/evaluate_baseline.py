from stable_baselines3 import SAC
from datetime import datetime
import numpy as np

def to_datestring(unixtime: int, format='%Y-%m-%d_%H:%M:%S'):
	return datetime.utcfromtimestamp(unixtime).strftime(format)

def eval_success(env, model, eval_num=100):
    if type(model) is str:
        model = SAC.load(model)
    succ = 0
    tout = 0
    for i in range(eval_num):
        print(f"EVAL: iteration {i}, successes so far {succ}, time limit {tout}")
        env.seed(i)
        obs = env.reset()
        t=0
        while True:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
            t+=1
            if done or t >=200:
                if info['success']:
                    succ += 1
                break
        if t>=200:
            tout += 1
    return succ/eval_num, tout
