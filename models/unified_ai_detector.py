import os
import joblib
import numpy as np
import tensorflow as tf


class UnifiedAIDetector:
    def __init__(self, models_dir='models'):
        self.models_dir = models_dir
        self.scaler = joblib.load(os.path.join(models_dir, 'scaler.pkl'))
        self.label_encoder = joblib.load(os.path.join(models_dir, 'label_encoder.pkl'))

    def _load_model(self, fname):
        path = os.path.join(self.models_dir, fname)
        if fname.endswith('.keras'):
            return tf.keras.models.load_model(path), 'keras'
        else:
            return joblib.load(path), 'sklearn'

    def predict_best_proba(self, X_new):
        '''Return AI probability using the best model only.'''
        X_scaled = self.scaler.transform(X_new)
        model, mtype = self._load_model('random_forest.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        return probs

    def predict_best_label(self, X_new, threshold=0.5):
        '''Return labels (0/1) using the best model only.'''
        probs = self.predict_best_proba(X_new)
        return (probs > threshold).astype(int)

    def predict_all_proba(self, X_new):
        '''Return probabilities from all models as a dict.'''
        X_scaled = self.scaler.transform(X_new)
        results = {}
        model, mtype = self._load_model('naive_bayes.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['naive_bayes'] = probs
        model, mtype = self._load_model('logistic_regression.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['logistic_regression'] = probs
        model, mtype = self._load_model('random_forest.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['random_forest'] = probs
        model, mtype = self._load_model('svm.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['svm'] = probs
        model, mtype = self._load_model('xgboost.pkl')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['xgboost'] = probs
        model, mtype = self._load_model('feedforward_nn.keras')
        if mtype == 'keras':
            probs = model.predict(X_scaled, verbose=0).flatten()
        else:
            probs = model.predict_proba(X_scaled)[:, 1]
        results['feedforward_nn'] = probs
        return results


detector = UnifiedAIDetector()
print('🚀 UnifiedAIDetector ready. Use detector.predict_best_proba(X)')