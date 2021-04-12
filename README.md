# wavelets_STP
This is an attempt to use wavelet coefficients obtained from discrete wavelet transform (DWT) as engineered features on stock trend prediction (STP), inspired by Multi-scale Two-way Deep Neural Network [paper](https://www.ijcai.org/Proceedings/2020/0628.pdf). 

## Requirment
```
Python 
Numpy
Pandas
Sklearn 
... 
```

## Data Digestion 
- 510050_1m.csv: mintue quote of 510050, a major index in mainland China, from 2015 to 2020. Each row contains open, close, high, low, and trade volume. 

## File Digestion
- preprocess.py: standardize each column and obtain their wavelet coefficients; assign -1, 0, and 1 to each row as stock trend label. 
- train_test.py: fit a couple of classic ML models and track their accuracies; 
- naive_trading.py: develop a simple long short trading strategy using the model with the best accuracy, and report some metrics on the strategies (annulized return, Sharpe Ratio, Calmar Ratio, maximum drawdown, and etc...)

