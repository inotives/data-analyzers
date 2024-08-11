import requests 

class TradeAPI():
    def __init__(self, apikey, apisecret, apiurl):
        self.apikey = apikey
        self.apisecret = apisecret
        self.apiurl = apiurl 
    
    