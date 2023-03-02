import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV

# Load data
data = pd.read_csv('optimParameters.csv')
X = data.iloc[:, :-1]
y = data.iloc[:, -1]

# Create pipeline
pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('ridge', Ridge())
])

# Set parameter grid
param_grid = {
    'scaler__with_mean': [True, False],
    'scaler__with_std': [True, False],
    'ridge__alpha': [0.1, 1.0, 10.0]
}

# Perform grid search
grid_search = GridSearchCV(pipe, param_grid=param_grid, cv=5, n_jobs=-1)
grid_search.fit(X, y)

# Print results
print("Best parameters:", grid_search.best_params_)
print("Best score:", grid_search.best_score_)
