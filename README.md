# Simple-Rl-Trading Bot
Simple trading reinforcement learning with the help of simple-baselines and gym env.

<br>

<div align="center">
<img hight="300" width="700" alt="PNG" align="center" src="https://github.com/bhavithran1/bhavithran1/blob/main/assets/rl.png">
</div>
<br>

Reinforment Learning is about learning the optimal behavior in an environment to obtain maximum reward. This optimal behavior is learned through interactions with the environment and observations of how it responds, similar to children exploring the world around them and learning the actions that help them achieve a goal.

I thought it would be a fun idea to apply this method to the world of finance.

<br>

# How To Use ?

1. Open main.ipynb, Replace the PATH with the path to your own dataset.

```
## place your custom csv file here!
PATH = 'raw_data/BTC_USD/train_5m.csv'

data = pd.read_csv(PATH)
#data.drop(columns=['tic'], inplace=True)
data = pipeline(data, 'data/data.csv')
```

2. Run main.ipynb to generate all needed features and split the data into train,test, validation sets.

<br>
<div align="center">
<img hight="300" width="700" alt="PNG" align="center" src="https://github.com/bhavithran1/bhavithran1/blob/main/assets/feats.png">
</div>

<br>

3.Run alpha.ipynb to train the reinforment learning model and the result will be returned at the end of the test.
<br>
<br>
# Dependencies

- `gym`
- `stable_baselines3`
- `numpy`

For the dependecies for the features used:

```
%pip install ccxt
%pip install pandas
%pip install jupyterlab
%pip install gym==0.22
%pip install feature_engine
%pip install tensorforce
%pip install scikit-learn
%pip install optuna
%pip install quantstats==0.0.50
%pip install ta
%pip install TA-Lib
%pip install git+https://github.com/twopirllc/pandas-ta@development

# If using conda 
%conda install -c conda-forge ta
%conda install -c conda-forge ta-lib
```

<br>
<br>
<br>




