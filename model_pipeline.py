import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

class CancellationModelPipeline:
    def __init__(self, model, training_columns, features_list):
        self.model = model
        self.training_columns = training_columns
        self.features_list = features_list

    def _transform(self, input_df):
        X_raw = input_df[self.features_list]
        X_encoded = pd.get_dummies(X_raw, columns=['customer_state', 'customer_city'], dtype=int)
        X_matched = X_encoded.reindex(columns=self.training_columns, fill_value=0)
        return csr_matrix(X_matched.to_numpy(dtype=np.float32))

    def predict(self, input_df):
        return self.model.predict(self._transform(input_df))

    def predict_proba(self, input_df):
        return self.model.predict_proba(self._transform(input_df))[:, 1]
