import xlrd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

wb = xlrd.open_workbook("/home/arunkrishna/Desktop/College/soft assignment/datasets.xlsx")
ws_t = wb.sheet_by_name("traditional_mode")
ws_o = wb.sheet_by_name("online_mode")

rows_t = ws_t.get_rows()
next(rows_t)

rows_o = ws_o.get_rows()
next(rows_o)

x_t = []
y_t = []

x_o = []
y_o = []


for row in rows_t:
    x_t.append(row[0].value)
    y_t.append(row[1].value)

for row in rows_o:
    x_o.append(row[0].value)
    y_o.append(row[1].value)

def Model(x_d, y_d, title):
    x = np.array(x_d).reshape(-1,1)
    regression_model = LinearRegression()

    # Fit the data(train the model)
    regression_model.fit(x, y_d)

    # Predict
    y_predicted = regression_model.predict(x)

    # model evaluation
    mse = mean_squared_error(y_d, y_predicted)
    rmse = np.sqrt(mean_squared_error(y_d, y_predicted))
    r2 = r2_score(y_d, y_predicted)


    plt.scatter(x, y_d, color='red')
    plt.plot(x, y_predicted, color='green')
    positions = (1, 2, 3, 4, 5)
    labels = ("0-10", "10-20", "20-30", "30-40", "40-50")

    plt.xticks(positions, labels)
    plt.xlabel('Marks (0 - 50)')
    plt.ylabel('Comprehension (0 - 10)')
    plt.title(title)
    plt.show()

    # printing values
    print('Slope:', regression_model.coef_)
    print('Intercept:', regression_model.intercept_)
    print('MSE:', mse)
    print('Root mean squared error: ', rmse)
    print('R2 score: ', r2)


# Traditional Mode
Model(x_t, y_t, 'Traditional Learning Mode')

# Online Mode
Model(x_o, y_o, 'Online Learning Mode')