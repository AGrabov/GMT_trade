from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.metrics import (explained_variance_score, mean_absolute_error,
                             mean_absolute_percentage_error,
                             mean_squared_error, mean_squared_log_error,
                             r2_score)
from bayes_opt import BayesianOptimization
from pyswarm import pso

class HyperparameterSearch:
    def __init__(self, model, X, y, scoring_function, random_state=None):
        self.model = model
        self.X = X
        self.y = y
        self.scoring_function = scoring_function
        self.random_state = random_state

    def grid_search(self, param_grid, cv=5):
        try:
            grid_search = GridSearchCV(self.model, param_grid, scoring=self.scoring_function, cv=cv)
            grid_search.fit(self.X, self.y)
            best_params = grid_search.best_params_
            best_score = grid_search.best_score_
            return best_params, best_score
        except Exception as e:
            print(f"Error during grid search: {e}")
            return None

    def random_search(self, param_distributions, n_iter=100, cv=5):
        try:
            random_search = RandomizedSearchCV(self.model, param_distributions, n_iter=n_iter, scoring=self.scoring_function, cv=cv, random_state=self.random_state)
            random_search.fit(self.X, self.y)
            best_params = random_search.best_params_
            best_score = random_search.best_score_
            return best_params, best_score
        except Exception as e:
            print(f"Error during random search: {e}")
            return None

    def bayesian_optimization(self, param_bounds, init_points=5, n_iter=50):
        def objective_function(**params):
            self.model.set_params(**params)
            return cross_val_score(self.model, self.X, self.y, cv=5, scoring=self.scoring_function).mean()

        try:
            optimizer = BayesianOptimization(f=objective_function, pbounds=param_bounds, random_state=self.random_state)
            optimizer.maximize(init_points=init_points, n_iter=n_iter)
            best_params = optimizer.max['params']
            best_score = optimizer.max['target']
            return best_params, best_score
        except Exception as e:
            print(f"Error during Bayesian optimization: {e}")
            return None

    def pso_optimization(self, lb, ub, swarmsize=30, maxiter=100):
        """
        lb: list
            The lower bounds of the parameter space.
        ub: list
            The upper bounds of the parameter space.
        swarmsize: int
            The number of particles in the swarm.
        maxiter: int
            The maximum number of iterations for the swarm to search.
        """
        def objective_function(x):
            params = {key: val for key, val in zip(self.model.get_params().keys(), x)}
            self.model.set_params(**params)
            return -cross_val_score(self.model, self.X, self.y, cv=5, scoring=self.scoring_function).mean()

        try:
            xopt, fopt = pso(objective_function, lb, ub, swarmsize=swarmsize, maxiter=maxiter)
            best_params = {key: val for key, val in zip(self.model.get_params().keys(), xopt)}
            best_score = -fopt
            return best_params, best_score
        except Exception as e:
            print(f"Error during PSO optimization: {e}")
            return None
