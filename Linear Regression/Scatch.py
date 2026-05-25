import pandas as pd
import matplotlib.pyplot as plt

data=pd.read_csv(r'C:\Users\nguye\OneDrive\Documents\Data Analysis\Linear Regression\data.csv')

def loss_function(m,b,points):
    total_error = 0
    for index, row in points.iterrows():
        x = row.studytime
        y = row.Score
        total_error += (y - (m * x + b))**2
    return total_error / len(points)

def gradient_descent(m_now,b_now,points,L):
    m_gradients = 0
    b_gradients = 0

    n=len(points)

    for i in range(n):
        x= points.iloc[i].studytime
        y= points.iloc[i].Score
        
        m_gradients += (-2/n) * x * (y - (m_now * x + b_now))
        b_gradients += (-2/n)  * (y - (m_now * x + b_now))
    
    m = m_now - m_gradients * L
    b = b_now - b_gradients * L
    return m,b

m = 0
b = 0
L= 0.0001
epochs=100
for i in range (epochs):
    if i %50 == 0:
        print(f'Epoch: {i}')
    m, b = gradient_descent(m,b,data,L)

print(m,b)
plt.figure(figsize=(10, 6))
plt.scatter(data.studytime, data.Score, label='Data Points') 
x_line = [data.studytime.min(), data.studytime.max()]
y_line = [m * x + b for x in x_line]
plt.plot(x_line, y_line, color='red', label='Regression Line') 
plt.xlabel("Study Time")
plt.ylabel("Score")
plt.title("Linear Regression with Gradient Descent")
plt.legend()
plt.grid(True) 
plt.show()
