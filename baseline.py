from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_squared_error
from common import timed_evaluation

@timed_evaluation
def baseline_elastic_net(X_train, y_train, X_test, y_test, **kwargs):
    model = ElasticNetCV(cv=5, **kwargs)
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    mse_test = mean_squared_error(y_test, y_pred_test)
    return model.coef_, mse_test