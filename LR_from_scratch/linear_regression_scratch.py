import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

ds = pd.read_csv("Salary_Data.csv")

_input = ds["YearsExperience"].values
_output = ds["Salary"].values

m = len(_input)

def cost_function(inp, out, w, b):
    cost_sum = 0
    for i in range (m):
        f = w * inp[i] + b
        cost = (f - out[i]) ** 2
        cost_sum += cost
    total_cost = (1/(2*m)) * cost_sum
    return total_cost

def calc_derivative(inp, out, w, b):
    dc_dw = 0
    dc_db = 0

    for i in range (m):
        f = w * inp[i] + b
        
        dc_dw += (f - out[i]) * inp[i]
        dc_db += f - out[i]

    dc_dw = (1/m) * dc_dw
    dc_db = (1/m) * dc_db
    return dc_dw, dc_db

def gradient_descent(inp, out, alpha, iterations):
    w = 0
    b = 0
      
    for i in range (iterations):
        dc_dw, dc_db = calc_derivative(inp, out, w, b)
        
        w = w - alpha * dc_dw
        b = b - alpha * dc_db

        print(f"iteration {i}: cost= {cost_function(inp, out, w, b)}")
    return w, b

learning_rate = .0001
iterations = 10000
_w, _b = gradient_descent(_input, _output, learning_rate, iterations)

plt.scatter(_input, _output, label='Data points')
_input_values = np.linspace(min(_input), max(_input), 100)
_output_values = _w * _input_values + _b
plt.plot(_input_values, _output_values, color='green', label="prediction_line")
plt.xlabel("YearsExperience")
plt.ylabel("Salary")
plt.legend()
plt.show()
