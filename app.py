from flask import Flask, render_template, request
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error
from statsmodels.tsa.arima.model import ARIMA
from math import sqrt
import mpld3
import os
import matplotlib
import pmdarima as pm

matplotlib.use('Agg')
app = Flask(__name__)
UPLOAD_FOLDER = 'uploads'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in {'csv'}

def find_arima_parameters(df, di):
    res = pm.auto_arima(df.values, start_p=0, start_q=0,
                        test='adf',
                        max_p=3, max_q=3,
                        m=8,
                        d=di,
                        seasonal=True,
                        start_P=0, 
                        D=di,
                        trace=True,
                        error_action='ignore',  
                        suppress_warnings=True, 
                        stepwise=True)
    return res

def plot_arima_forecast(train_data, test_data, order, seasonal_order):
    # Extract relevant columns and reset the index
    train_data = train_data.reset_index()[['Created Date', 'Cumulative_Tickets_Sold']]
    train_data = train_data.set_index('Created Date')

    try:
        test_data['Created Date'] = pd.to_datetime(test_data['Created Date'], format='%d-%m-%Y')
    except ValueError:
        test_data['Created Date'] = pd.to_datetime(test_data['Created Date'])

    test_data = test_data.set_index('Created Date')

    # Perform ARIMA forecasting
    model = ARIMA(train_data.values, order=order, seasonal_order=seasonal_order)
    results = model.fit()

    # Get the forecast for the original steps
    forecast = results.get_forecast(steps=len(test_data))

    # Set a minimum value for predictions to prevent them from going negative
    min_value = 0
    forecast.predicted_mean = forecast.predicted_mean.clip(min=min_value)
    forecast.conf_int()[:, 0] = forecast.conf_int()[:, 0].clip(min=min_value)
    forecast.conf_int()[:, 1] = forecast.conf_int()[:, 1].clip(min=min_value)

    # Calculate performance metrics
    mse = mean_squared_error(test_data['Cumulative_Tickets_Sold'].values, forecast.predicted_mean)
    rmse = sqrt(mse)
    mape = mean_absolute_percentage_error(test_data['Cumulative_Tickets_Sold'].values, forecast.predicted_mean)

    # Plotting
    plt.figure(figsize=(12, 6))

    train_days = range(1, len(train_data) + 1)
    test_days = range(len(train_data) + 1, len(train_data) + len(test_data) + 1)

    # Training data
    plt.plot(train_days, train_data.values, label='Training', linewidth=2, linestyle='--', color='lightblue', marker='o')

    # Actual test data
    plt.plot(test_days, test_data['Cumulative_Tickets_Sold'].values, label='Actual', linewidth=2, color='green', marker='o')

    # Forecasted data
    plt.plot(test_days, forecast.predicted_mean, color='red', label='Forecast', linewidth=2, marker='o')

    # Confidence interval
    plt.fill_between(test_days, forecast.conf_int()[:, 0], forecast.conf_int()[:, 1], color='red', alpha=0.2)

    plt.title('ARIMA Forecasting - Cumulative Tickets Sold', fontsize=16)
    plt.xlabel('Days', fontsize=12)
    plt.ylabel('Cumulative Tickets Sold', fontsize=12)
    plt.legend(loc='upper left')  # Adjust the legend location
    plt.grid(True, linestyle='--', alpha=0.3)  # Adjust grid lines

    # Convert Matplotlib figure to HTML using mpld3
    plot_html = mpld3.fig_to_html(plt.gcf())

    # Close the Matplotlib figure
    plt.close()

    return plot_html, rmse, mape

def load_and_plot_forecast(file_path):
    df = pd.read_csv(file_path)

    try:
        df['Created Date'] = pd.to_datetime(df['Created Date'], format='%d-%m-%Y')
    except ValueError:
        df['Created Date'] = pd.to_datetime(df['Created Date'])

    train_size = int(len(df) * 0.8)
    train_data, test_data = df.iloc[:train_size][['Created Date', 'Cumulative_Tickets_Sold']].copy(), df.iloc[train_size:][['Created Date', 'Cumulative_Tickets_Sold']].copy()
    train_data = train_data.set_index('Created Date')

    di = 2  # Set the differencing order
    parameters = find_arima_parameters(train_data, di)

    plot_html, rmse, mape = plot_arima_forecast(train_data, test_data, parameters.order, parameters.seasonal_order)

    return plot_html, rmse, mape

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        file = request.files['file']

        if file and allowed_file(file.filename):
            filename = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filename)

            plot_html, rmse, mape = load_and_plot_forecast(filename)

            return render_template('index.html', plot_html=plot_html, rmse=rmse, mape=mape)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
