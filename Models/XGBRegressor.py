from sklearn.model_selection import GridSearchCV
from xgboost import XGBRegressor
import json

def XGB(x_train, y_train, x_test, y_test):

    with open('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/utils/grid_params_XGB.json', 'r') as f:
        grid_params = json.load(f)

    model_XGB = XGBRegressor()

    grid_search = GridSearchCV( estimator = model_XGB, 
                                param_grid = grid_params, 
                                scoring = ['r2','neg_mean_squared_error'],
                                refit = "r2",
                                cv = 5, #crossvalidation
                                verbose = 4
                                )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    result_xgb = best_model.predict(x_test)

    return best_model,  result_xgb

def XGB_metrics_results ( grid_search, best_model ):
    print(grid_search.best_estimator_)
    print(best_model)
    print(grid_search.best_score_)