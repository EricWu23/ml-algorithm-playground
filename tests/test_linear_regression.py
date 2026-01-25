from ml_algorithm_playground.linear.linear_regression import LinearRegression
import numpy as np

def test_linear_regression():
    X = np.array([
        [1,2],
        [1,4],
        [2,2],
        [2,4],
        [3,2],
        [3,4]  
    ])
    y= X*10

    model = LinearRegression(lr=0.01, max_iter=1000)
    model.fit(X,y)
    y_pred = model.predict(X)

    mae = np.abs(y_pred - y).mean()
    print(f"Mean Absolute Error: {mae}")
    assert mae < 1.0
