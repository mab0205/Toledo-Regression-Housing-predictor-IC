from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV
import json

def GradientBR(X_train, y_train, X_test, y_test):
    
    with open('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/utils/grid_params_GBR.json', 'r') as f:
        grid_params = json.load(f)

    model_gradient = GradientBoostingRegressor()
    
    grid_search = GridSearchCV( estimator = model_gradient, 
                                param_grid = grid_params, 
                                scoring = ['r2','neg_mean_squared_error'],
                                refit = "r2",
                                cv = 5, #crossvalidation
                                verbose = 4
                                )
    
    grid_search.fit(X_train, y_train)
    best_model = grid_search.best_estimator_
    # Hacer predicciones en los datos de prueba
    result_predictions = best_model.predict(X_test)

    return best_model , result_predictions

def Gradient_metrics_results ( grid_search, best_model ):
    print(grid_search.best_estimator_)
    print(best_model)
    print(grid_search.best_score_)