import pandas as pd
import numpy as np 
import requests
import time

class TraderBot: 
    def __init__(self, api_key, api_secret):
        self.exchange = ''

    def place_order(self, crypto_pair, order_type, amount, price):
        payload = {
            'pair': crypto_pair,
            'type': order_type,
            'amount': amount,
            'price': price
        }
        headers = {
            'API-KEY': self.api_key,
            'API-SECRET': self.api_secret
        }
        response = requests.post(f"{self.api_url}/order", json=payload, headers=headers)
        return response.json()
    
    def execute_trade(self, crypto_pair, forecasted_price, buy_threshold, sell_threshold, amount):
        if forecasted_price <= buy_threshold:
            self.place_order(crypto_pair, 'buy', amount, forecasted_price)
        elif forecasted_price >= sell_threshold:
            self.place_order(crypto_pair, 'sell', amount, forecasted_price)

    def run(self, crypto_pair, model_fit, buy_threshold, sell_threshold, amount, interval=60):
        while True:
            forecasted_price = model_fit.forecast(steps=1)[0]
            self.execute_trade(crypto_pair, forecasted_price, buy_threshold, sell_threshold, amount)
            time.sleep(interval)