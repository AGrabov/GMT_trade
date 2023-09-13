class CryptoDataNormalizer:
    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, data):
        self.mean = data.mean()
        self.std = data.std()

    def transform(self, data):
        return (data - self.mean) / self.std

    def inverse_transform(self, normalized_data):
        return normalized_data * self.std + self.mean

    def normalize_returns(self, data):
        returns = data.pct_change().dropna()
        self.fit(returns)
        return self.transform(returns)

    def inverse_transform_returns(self, normalized_returns, previous_price):
        returns = self.inverse_transform(normalized_returns)
        return (1 + returns) * previous_price
