
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from apscheduler.schedulers.background import BackgroundScheduler


import mysql.connector
import logging
df = pd.DataFrame()
logging.basicConfig(level=logging.INFO)



def fetch_data_from_db() -> pd.DataFrame:
    """Fetch data from the database and return as a DataFrame."""
    global df
    try:
        query = """
            SELECT 
                inventory.id,
                a.OperatorType,
                users.UserName,
                checkout.city,
                users.emailid AS Email, 
                checkout.country_name,
                inventory.PurchaseDate,
                DATE_FORMAT(inventory.PurchaseDate, '%r') AS `time`,   
                checkout.price AS SellingPrice,
                (SELECT planename FROM tbl_Plane WHERE P_id = inventory.planeid) AS PlanName,
                a.vaildity AS vaildity,
                (SELECT CountryName FROM tbl_reasonbycountry WHERE ID = a.country) AS countryname,
                (SELECT Name FROM tbl_region WHERE ID = a.region) AS regionname,
                (CASE 
                    WHEN (inventory.transcation IS NOT NULL OR Payment_method = 'Stripe') 
                    THEN 'stripe' 
                    ELSE 'paypal' 
                END) AS payment_gateway,
                checkout.source,
                checkout.Refsite,
                checkout.accounttype,
                checkout.CompanyBuyingPrice,
                checkout.TravelDate,
                inventory.Activation_Date,
                inventory.IOrderId
            FROM 
                tbl_Inventroy inventory
            LEFT JOIN 
                tbl_plane AS a ON a.P_id = inventory.planeid
            LEFT JOIN 
                User_Login users ON inventory.CustomerID = users.Customerid
            LEFT JOIN 
                Checkoutdata checkout ON checkout.guid = inventory.guid
            WHERE 
                inventory.status = 3 
                AND inventory.PurchaseDate BETWEEN '2022-11-01' AND '2024-11-28'  
            ORDER BY 
                inventory.PurchaseDate DESC;
        """

        # Connect to the MySQL database and fetch data into a DataFrame
        connection = mysql.connector.connect(
            host="34.42.98.10",       
            user="clayerp",   
            password="6z^*V2M9Y(/+", 
            database="esim_local" 
        )

        # Fetch the data into a DataFrame
        df = pd.read_sql(query, connection)

        # Close the connection
        connection.close()

        logging.info("Data successfully loaded from the database.")
        # df = df.drop_duplicates()
        df = df.replace({'\\N': np.nan, 'undefined': np.nan, 'null': np.nan})
        df['SellingPrice'] = pd.to_numeric(df['SellingPrice'], errors='coerce').fillna(0).astype(int)
        df['CompanyBuyingPrice'] = pd.to_numeric(df['CompanyBuyingPrice'], errors='coerce').fillna(0).astype(int)

        # Convert 'PurchaseDate' to datetime format and then format it as required
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        #df['PurchaseDate'] = df['PurchaseDate'].dt.date
        df['TravelDate'] = pd.to_datetime(df['TravelDate'], errors='coerce')
        return df

    except mysql.connector.Error as e:
        logging.error(f"Database error: {str(e)}")
        return pd.DataFrame()  # Return an empty DataFrame in case of error

fetch_data_from_db()

#################################models###########################
# Function to process data and forecast sales
def generate_sales_forecasts(df):
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce').dt.date
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    df['SellingPrice'] = pd.to_numeric(df['SellingPrice'], errors='coerce')
    daily_sales = df.groupby(['PurchaseDate']).agg(
        SalesPrice=('SellingPrice', 'sum'),
        SalesCount=('SellingPrice', 'count')
    ).reset_index()
    daily_sales.set_index('PurchaseDate', inplace=True)
    daily_sales.index = pd.to_datetime(daily_sales.index)

    daily_data = daily_sales.resample('D').sum()
    monthly_data = daily_sales.resample('M').sum()
    yearly_data = daily_sales.resample('Y').sum()

    # Daily Sales Forecast (ARIMA)
    sales_model = ARIMA(daily_data['SalesCount'], order=(5, 2, 0))
    sales_model_fit = sales_model.fit()
    daily_forecast = sales_model_fit.forecast(steps=15)
    df1 = pd.DataFrame({
        'Date': pd.date_range(start=daily_data.index[-1] + pd.Timedelta(days=1), periods=15, freq='D'),
        'SalesForecast': np.ceil(daily_forecast).astype(int)
    })

    # Monthly Sales Forecast (SARIMA)
    sarima_model = SARIMAX(monthly_data['SalesCount'], order=(2, 0, 1), seasonal_order=(1, 1, 1, 12))
    sarima_model_fit = sarima_model.fit()
    monthly_forecast = sarima_model_fit.get_forecast(steps=12).predicted_mean
    df2 = pd.DataFrame({
        'Date': pd.date_range(start=monthly_data.index[-1] + pd.DateOffset(months=1), periods=12, freq='M'),
        'SalesForecast': np.ceil(monthly_forecast).astype(int)
    })

    # Yearly Sales Forecast (ARIMA)
    yearly_sales_model = ARIMA(yearly_data['SalesCount'], order=(5, 1, 0))
    yearly_sales_model_fit = yearly_sales_model.fit()
    yearly_forecast = yearly_sales_model_fit.forecast(steps=5)
    df3 = pd.DataFrame({
        'Date': pd.date_range(start=yearly_data.index[-1] + pd.DateOffset(years=1), periods=5, freq='Y'),
        'SalesForecast': np.round(yearly_forecast).astype(int)
    })
    df1['Date'] = df1['Date'].dt.date
    df2['Date'] = df2['Date'].dt.date
    df3['Date'] = df3['Date'].dt.date

    
    return df1, df2, df3
def schedule_sales_forecast():
    scheduler = BackgroundScheduler()
    
    # Schedule your job after 10 days at midnight (00:00)
    run_time = datetime.now() + timedelta(days=1)
    run_time = run_time.replace(hour=0, minute=0, second=0, microsecond=0)  # Set to midnight
    
    scheduler.add_job(generate_sales_forecasts, 'date', run_date=run_time)
    scheduler.start()

    

if __name__ == '__main__':
    schedule_sales_forecast()
