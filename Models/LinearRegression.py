from sklearn import linear_model 
from sklearn.model_selection import GridSearchCV
import json

def LinearRegression(x_train, y_train, x_test, y_test):

    #with open('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/utils/grid_params_linear.json', 'r') as f:
       # grid_params = json.load(f)
    grid_params = {
        'fit_intercept': [True, False],
        'copy_X': [True, False],
        'n_jobs': [None, -1]
        }

    model_linear = linear_model.LinearRegression()


    grid_search = GridSearchCV( estimator = model_linear, 
                                param_grid = grid_params, 
                                scoring = ['r2','neg_mean_squared_error'],
                                refit = "r2",
                                cv = 5, #crossvalidation
                                verbose = 4
                                )
    
    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    linear_result = best_model.predict(x_test)

    return model_linear, linear_result

