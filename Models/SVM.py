from sklearn.pipeline import make_pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVR
import json

def SupportVector(x_train, y_train, x_test, y_test):

    with open('C:/Users/gaboh/UTFPR/GitHub/Toledo-Regression-Housing-predictor-IC/utils/grid_params_SVM.json', 'r') as f:
        grid_params = json.load(f)
    
    model_SVM = make_pipeline(StandardScaler(), SVR())

    grid_search = GridSearchCV( estimator = model_SVM, 
                                param_grid = grid_params, 
                                scoring = ['r2','neg_mean_squared_error'],
                                refit = "r2",
                                cv = 5, #crossvalidation
                                verbose = 4
                                )

    grid_search.fit(x_train, y_train)
    best_model = grid_search.best_estimator_
    result_SVR = best_model.predict(x_test)

    return model_SVM, result_SVR 



