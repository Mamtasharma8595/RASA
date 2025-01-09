from fuzzywuzzy import process
from apscheduler.schedulers.background import BackgroundScheduler
from tabulate import tabulate
from typing import Any, Text, Dict, Tuple, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.executor import CollectingDispatcher
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import requests
import json
import matplotlib.pyplot as plt
import re
import os
import dateparser
from .utils import df,generate_sales_forecasts
from dateutil.relativedelta import relativedelta
def background_forecast_job():
    generate_sales_forecasts(df)

# Schedule the job to run automatically (e.g., every 1 days)
scheduler = BackgroundScheduler()
scheduler.add_job(background_forecast_job, 'date', run_date=datetime.now() + timedelta(days=1))
scheduler.start()
# import mysql.connector
import logging
# df = pd.DataFrame()

logging.basicConfig(level=logging.INFO)



# def fetch_data_from_db() -> pd.DataFrame:
#     """Fetch data from the database and return as a DataFrame."""
#     global df
#     try:
#         query = """
#             SELECT 
#                 inventory.id,
#                 a.OperatorType,
#                 users.UserName,
#                 checkout.city,
#                 users.emailid AS Email, 
#                 checkout.country_name,
#                 inventory.PurchaseDate,
#                 DATE_FORMAT(inventory.PurchaseDate, '%r') AS `time`,   
#                 checkout.price AS SellingPrice,
#                 (SELECT planename FROM tbl_Plane WHERE P_id = inventory.planeid) AS PlanName,
#                 a.vaildity AS vaildity,
#                 (SELECT CountryName FROM tbl_reasonbycountry WHERE ID = a.country) AS countryname,
#                 (SELECT Name FROM tbl_region WHERE ID = a.region) AS regionname,
#                 (CASE 
#                     WHEN (inventory.transcation IS NOT NULL OR Payment_method = 'Stripe') 
#                     THEN 'stripe' 
#                     ELSE 'paypal' 
#                 END) AS payment_gateway,
#                 checkout.source,
#                 checkout.Refsite,
#                 checkout.accounttype,
#                 checkout.CompanyBuyingPrice,
#                 checkout.TravelDate,
#                 inventory.Activation_Date,
#                 inventory.IOrderId
#             FROM 
#                 tbl_Inventroy inventory
#             LEFT JOIN 
#                 tbl_plane AS a ON a.P_id = inventory.planeid
#             LEFT JOIN 
#                 User_Login users ON inventory.CustomerID = users.Customerid
#             LEFT JOIN 
#                 Checkoutdata checkout ON checkout.guid = inventory.guid
#             WHERE 
#                 inventory.status = 3 
#                 AND inventory.PurchaseDate BETWEEN '2022-11-01' AND '2024-11-28'  
#             ORDER BY 
#                 inventory.PurchaseDate DESC;
#         """

#         # Connect to the MySQL database and fetch data into a DataFrame
#         connection = mysql.connector.connect(
#             host="34.42.98.10",       
#             user="clayerp",   
#             password="6z^*V2M9Y(/+", 
#             database="esim_local" 
#         )

#         # Fetch the data into a DataFrame
#         df = pd.read_sql(query, connection)

#         # Close the connection
#         connection.close()

#         logging.info("Data successfully loaded from the database.")
#         # df = df.drop_duplicates()
#         df = df.replace({'\\N': np.nan, 'undefined': np.nan, 'null': np.nan})
#         df['SellingPrice'] = pd.to_numeric(df['SellingPrice'], errors='coerce').fillna(0).astype(int)
#         df['CompanyBuyingPrice'] = pd.to_numeric(df['CompanyBuyingPrice'], errors='coerce').fillna(0).astype(int)

#         # Convert 'PurchaseDate' to datetime format and then format it as required
#         df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
#         #df['PurchaseDate'] = df['PurchaseDate'].dt.date
#         df['TravelDate'] = pd.to_datetime(df['TravelDate'], errors='coerce')
#         return df

#     except mysql.connector.Error as e:
#         logging.error(f"Database error: {str(e)}")
#         return pd.DataFrame()  # Return an empty DataFrame in case of error

# fetch_data_from_db()
class ActionGenerateSalesForecast(Action):
    def name(self):
        return "action_generate_sales_forecast"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain):
        # Inform the user that the forecast generation has started
        # dispatcher.utter_message(text="Sales forecast generation is running in the background.")

        # Start the background job (not blocking the main thread)
        thread = Thread(target=background_forecast_job)
        thread.start()

        return []

months = { 'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3, 'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7, 'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10, 'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12 }
months_reverse = {v: k for k, v in months.items()}

word_to_num = {
    # 'one': 1, 'two': 2, 'three': 3, 'four': 4, 'five': 5, 'six': 6, 
    # 'seven': 7, 'eight': 8, 'nine': 9, 'ten': 10, 'eleven': 11, 'twelve': 12,
    'first': 1, 'second': 2, 'third': 3, 'fourth': 4,
    'fifth': 5, 'sixth': 6, 'seventh': 7, 'eighth': 8,
    'ninth': 9, 'tenth': 10, 'eleventh': 11, 'twelfth': 12
}

def extract_months_from_text(text):
    # Regular expression to match numbers or words like '5', 'five', etc.
    pattern = r'\b(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        month_str = match.group(1).lower()
        if month_str.isdigit():
            return int(month_str)
        else:
            return word_to_num.get(month_str, 0)  # Convert word to number, default to 0 if not found
    return 0

def get_month_range_from_text(text):
    # Extract the number of months
    num_months = extract_months_from_text(text)
    
    if num_months > 0:
        # Get the current date
        current_date = datetime.now()
        
        # Calculate the date X months ago
        past_date = current_date - relativedelta(months=num_months)
        
        # Return the range of months (past month/year to current month/year)
        month_range = [[past_date.month, past_date.year], [current_date.month, current_date.year]]
        return month_range, num_months
    else:
        return None

def extract_date(text):
    # Clean the text by removing unnecessary ordinal suffixes
    cleaned_message = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)

    # Define a regex to capture likely date patterns
    date_patterns = r'(\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b|\b\d{1,2}\s(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s\d{4}\b|\b(?:Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec)\w*\s\d{1,2},?\s\d{4}\b)'
    matches = re.findall(date_patterns, cleaned_message, re.IGNORECASE)

    # Parse each match into a standard date format
    dates = []
    for match in matches:
        parsed_date = dateparser.parse(match)
        if parsed_date:
            dates.append(parsed_date.strftime('%Y-%m-%d'))  # Format to YYYY-MM-DD

    return dates[0] if dates else None

# def extract_months(text):
#     # Regular expression to find month names or abbreviations (case-insensitive)
#     pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
#     # Find all matches of month names (case-insensitive)
#     matches = re.findall(pattern, text, re.IGNORECASE)
    
#     # Convert matched month names to corresponding digits
#     month_digits = [months[match.lower()] for match in matches]
    
#     return month_digits

# def extract_date_range(text):
   
#     try:
#         # Define patterns for start_date and end_date
#         pattern = r"from\s+([\w\s,]+)\s+to\s+([\w\s,]+)|between\s+([\w\s,]+)\s+and\s+([\w\s,]+)|from\s+([\w\s,]+)\s+through\s+([\w\s,]+)"
#         match = re.search(pattern, text, re.IGNORECASE)
        
#         if match:
#             # Extract start_date and end_date
#             start_date_str = match.group(1) or match.group(3) or match.group(5)
#             end_date_str = match.group(2) or match.group(4) or match.group(6)
            
#             # Parse dates
#             start_date = pd.to_datetime(start_date_str, errors='coerce')
#             end_date = pd.to_datetime(end_date_str, errors='coerce')
            
#             # Validate parsed dates
#             if pd.isnull(start_date) or pd.isnull(end_date):
#                 return None, None, "Error: One or both dates could not be parsed. Please provide valid dates."
            
#             return start_date.date(), end_date.date(), None
        
#         return None, None, "Error: No valid date range found in the query."
    
#     except Exception as e:
#         return None, None, f"Error occurred while parsing date range: {str(e)}"

def extract_months(text):
    """Extracts month numbers from user input based on month names or numeric/ordinal representations."""
    
    # Regular expression to find month names or abbreviations (case-insensitive)
    month_pattern = r'\b(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\b'
    
    # Regular expression to find numeric months (1-12)
    #numeric_pattern = r'\b(1[0-2]|[1-9])\b'
    
    # Regular expression to find ordinal months (first to twelfth)
    ordinal_pattern = r'\b(first|second|third|fourth|fifth|sixth|seventh|eighth|ninth|tenth|eleventh|twelfth)\b'
    
    # Find all matches of month names
    matches = re.findall(month_pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits
    month_digits = [months[match.lower()] for match in matches]

    # # Check for numeric months
    # numeric_match = re.search(numeric_pattern, text)
    # if numeric_match:
    #     month_digits.append(int(numeric_match.group(0)))

    # Check for ordinal months
    ordinal_match = re.search(ordinal_pattern, text)
    if ordinal_match:
        month_digits.append(word_to_num.get(ordinal_match.group(0), None))

    return list(set(month_digits))
def extract_today(text):
    # Regular expression to match the word 'today'
    pattern = r'\btoday\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)

def extract_last_day(text):
    pattern = r'\blast\sday\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)

def extract_monthwise(text):
    pattern = r'\bmonthwise\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)
def extract_yearwise(text):
    pattern = r'\byearwise\b'
    matches = re.findall(pattern, text, re.IGNORECASE)
    return bool(matches)

def extract_years(text):
    # Regular expression to match years in YYYY format without capturing the century separately
    pattern = r'\b(?:19|20)\d{2}\b'
    
    # Find all matches of the pattern
    years = re.findall(pattern, text)
    
    return [int(year) for year in years]
    

def extract_month_year(text):    
    pattern = r'\b(jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)\s*(?:of\s*)?(\d{4})\b'
    
    # Find all matches of the pattern (month and year)
    matches = re.findall(pattern, text, re.IGNORECASE)
    
    # Convert matched month names to corresponding digits and pair them with the year as arrays
    month_year_pairs = [[months[month.lower()], int(year)] for month, year in matches]
    
    return month_year_pairs

def extract_quarters_from_text(text):
    # Regular expression to match quarter-related terms
    pattern = r'\b(?:quarter\s*(1|2|3|4)|q(1|2|3|4)|first|second|third|fourth)\b'
    year_pattern = r'\b(20\d{2})\b' 
    match_quarter = re.search(pattern, text, re.IGNORECASE)
    match_year = re.search(year_pattern, text)
    
    if match_quarter:
        # Normalize the matched group into a quarter number
        if match_quarter.group(1):  # Matches "quarter 1", "quarter 2", etc.
            quarter = int(match_quarter.group(1))
        elif match_quarter.group(2):  # Matches "q1", "q2", etc.
            quarter = int(match_quarter.group(2))
        else:  # Matches "first", "second", "third", "fourth"
            quarter_name = match_quarter.group(0).lower()
            quarter_map = {
                'first': 1,
                'second': 2,
                'third': 3,
                'fourth': 4
            }
            quarter = quarter_map.get(quarter_name, 0)

        
        year = int(match_year.group(0)) if match_year else pd.to_datetime('today').year

        # Return the corresponding month range for the identified quarter
        quarters = {
            1: (1, 3),   # Q1: January to March
            2: (4, 6),   # Q2: April to June
            3: (7, 9),   # Q3: July to September
            4: (10, 12)  # Q4: October to December
        }
        return (quarters.get(quarter), year)

    return None


def extract_half_year_from_text(text):
    pattern1 = r'\b(first|second|sec|1st|2nd|last)\s*(half|half\s+year|half\s+yearly)\b'
    pattern2 = r'\b(h1|h2|H1|H2)\b'


    # pattern = r'\b(first|second|sec|1st|2nd|last|h1|h2|H1|H2)\s+(half|half\s+year|half\s+yearly|half\s+year\s+report|half\s+year\s+analysis|half\s+yearly\s+summary)\b'
    year_pattern = r'\b(20\d{2})\b' 
    match_year = re.search(year_pattern, text)
    if match_year:
        year = int(match_year.group(1))
    else:
        year = datetime.now().year
    match1 = re.search(pattern1, text, re.IGNORECASE)
    match2 = re.search(pattern2, text, re.IGNORECASE)

    if match1:
        half = match1.group(1).lower()
        
        # Determine the months based on the half-year mentioned
        if half in ['first', '1st']:
            return  year, (1, 6)  # January to June (First half)
        elif half in ['second', 'sec', '2nd','last' ]:
            return year,(7, 12)  # July to December (Second half)
    if match2:
        half_term = match2.group(0).lower()  # Extract H1 or H2
        if half_term in ['h1']:
            return year, (1, 6)  # First half of the year
        elif half_term in ['h2']:
            return year, (7, 12)

    return None


def extract_fortnight(text):
    pattern = r'\b(fortnight|two\s+weeks|last\s+fortnight|last\s+two\s+weeks|last\s+14\s+days)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    if match:
        today = datetime.now()
        start_date = today - timedelta(days=14)
        return start_date, today
    return None


# def extract_last_n_months(text):
#     pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?)\b'
#     match = re.search(pattern, text, re.IGNORECASE)
    
#     if match:
#         num_months_str = match.group(2).lower()
#         num_months = int(num_months_str) if num_months_str.isdigit() else word_to_num.get(num_months_str, 0)
        
#         # Get the current date
#         today = datetime.now()

        
#         start_date = today - relativedelta(months=num_months)
        
#         return start_date, today  , num_months
#     return None
def extract_last_n_months(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_months_str = match.group(2).lower()
        if not num_months_str:
            num_months = 1
        else:
            num_months = int(num_months_str) if num_months_str.isdigit() else word_to_num.get(num_months_str, 1)
            
        #     num_months_str = 1
        # num_months = int(num_months_str) if num_months_str.isdigit() else word_to_num(num_months_str)
        
        # Get the current date and calculate the first day of the current month
        today = datetime.now()
        first_of_current_month = today.replace(day=1)
        
        # Calculate the end date (start of the current month)
        end_date = first_of_current_month - relativedelta(days=1)
        
        # Calculate the start date for the last N full months
        start_date = first_of_current_month - relativedelta(months=num_months)
        
        # Calculate total days in these months
        total_days = 0
        current_month_start = start_date
        for _ in range(num_months):
            next_month_start = current_month_start + relativedelta(months=1)
            total_days += (next_month_start - current_month_start).days
            current_month_start = next_month_start

        return start_date, end_date, num_months

    return None
    
def extract_last_n_hours(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+hours|hr|hour|hrs?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_hours_str = match.group(2).lower()
        if not num_hours_str:
            num_hours_str = 1
        num_hours = int(num_hours_str) if num_hours_str.isdigit() else word_to_num(num_hours_str)
        
        # Get the current time
        now = datetime.now()
        
        # Exclude the current hour by subtracting one hour
        end_time = now - timedelta(hours=1)
        
        # Calculate the start time for the last N hours
        start_time = end_time - timedelta(hours=num_hours - 1)
        start_time_str = start_time.strftime("%Y-%m-%d %H:%M:%S")
        
        # Format the end time to include only the time (no date)
        end_time_str = end_time.strftime("%Y-%m-%d %H:%M:%S")
        
        return start_time_str, end_time_str, num_hours

    

    return None
def extract_last_n_weeks(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+weeks?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_weeks_str = match.group(2).lower()
        if not num_weeks_str:
            num_weeks_str = 1
        num_weeks = int(num_weeks_str) if num_weeks_str.isdigit() else word_to_num(num_weeks_str)
        
        # Get the current date
        today = datetime.now()
        
        # Calculate the end of the last full week (Sunday of the previous week)
        end_of_last_week = today - timedelta(days=today.weekday() + 1)
        
        # Calculate the start date for the last N full weeks
        start_date = end_of_last_week - timedelta(weeks=num_weeks - 1, days=end_of_last_week.weekday())
        
        # Calculate total days in these weeks
        total_days = num_weeks * 7  # Each week has exactly 7 days
        
        return start_date, end_of_last_week, num_weeks

    return None
def extract_last_n_days(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+days?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_days_str = match.group(2).lower()
        if not num_days_str:
            num_days_str = 1
        num_days = int(num_days_str) if num_days_str.isdigit() else word_to_num(num_days_str)
        
        # Get today's date
        today = datetime.now()
        
        # Exclude today by subtracting one day
        end_date = today - timedelta(days=1)
        
        # Calculate the start date for the last N days
        start_date = end_date - timedelta(days=num_days - 1)
        
        return start_date, end_date, num_days

    return None
def extract_sales_for_specific_date(df, specific_date):
    try:
        # Convert the specific_date to a datetime object
        specific_date = pd.to_datetime(specific_date, errors='coerce').date()
        
        if pd.isna(specific_date):
            return None, "Error: The provided date is invalid."
        
        # Ensure 'PurchaseDate' is converted to datetime and handle any NaT values
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        
        # Check if the conversion was successful
        if df['PurchaseDate'].isnull().any():
            return None, "Error: Some dates in 'PurchaseDate' could not be converted."
        
        # Filter and calculate sales
        daily_sales_count = df[df['PurchaseDate'].dt.date == specific_date]['SellingPrice'].count()
        daily_sales_price = df[df['PurchaseDate'].dt.date == specific_date]['SellingPrice'].sum()
        
        return daily_sales_count, daily_sales_price
    
    except Exception as e:
        return None, f"Error occurred: {str(e)}"


def convert_to_datetime(self, date_str: str) -> datetime:
    """Converts a date string to a datetime object."""
        
    # Normalize the string for parsing
    date_str = date_str.replace("st", "").replace("nd", "").replace("rd", "").replace("th", "")
        
    try:
        return datetime.strptime(date_str.strip(), "%d %B %Y")
    except ValueError:
        # Handle month-only strings (e.g., "August 2022")
        return datetime.strptime(date_str.strip(), "%B %Y")

def calculate_country_sales(df, country):
    # Filter the data for the given country
    country_data = df[df['countryname'].str.lower()== country.lower()]
    
    # Calculate the sales count and total revenue
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    
    return sales_count, total_revenue
    
def calculate_country_sales_by_year(df, country, year):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_country_sales_by_month_year(df, country, month, year):
    country_data = df[(df['countryname'].str.lower()== country.lower()) & 
                      (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
    sales_count = country_data['SellingPrice'].count()
    total_revenue = country_data['SellingPrice'].sum()
    return sales_count, total_revenue
###########################################




def extract_country_from_text(text):
    if not text:
        logging.error("Received empty or None text for country extraction.")
        return []
    text_lower = text.lower()
    text_cleaned = ''.join(e for e in text_lower if e.isalnum() or e.isspace())
    all_countries = df['countryname'].dropna().unique().tolist() 
    all_countries_lower = [country.lower().strip() for country in all_countries] 
    matched_country = [country for country in all_countries_lower if country in text_cleaned]
    logging.info(f"Matched countries: {matched_country}")
    
    return matched_country


def extract_country(text: str) -> List[str]:
    # Compile a regex pattern to match valid country names
    text_cleaned = re.sub(r'[^\w\s]', '', text)
    valid_countries_sorted = sorted(valid_countries, key=lambda x: len(x.split()), reverse=True)
    matched_countries = set()

    for country in valid_countries_sorted:
        country_pattern = r'\b' + re.escape(country.lower()) + r'\b'
        if re.search(country_pattern, text_cleaned, re.IGNORECASE):
            matched_countries.add(country)

    # Convert set to list and limit to 2 results
    result_countries = list(matched_countries)
    return result_countries[:2]

valid_countries =  df['countryname'].dropna().unique().tolist()
#valid_countries = df['countryname'].tolist()

    

def calculate_country_sales_by_quarter(df, country, start_month, end_month, year):
    
    filtered_sales = df[
        (df['countryname'].str.lower() == country.lower()) &  # Case-insensitive match for country
        (df['PurchaseDate'].dt.month >= start_month) &
        (df['PurchaseDate'].dt.month <= end_month) &
        (df['PurchaseDate'].dt.year == year) 
    ]

    # Calculate total sales count and price
    total_sales_count = filtered_sales['SellingPrice'].count()
    total_sales_price = filtered_sales['SellingPrice'].sum()

    return total_sales_count, total_sales_price

def calculate_sales_for_last_n_hours(df, country, start_time, end_time):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'] >= start_time) & (df['PurchaseDate'] <= end_time)]
    
    # Calculate total sales and count
    total_sales = country_data['SellingPrice'].sum()  # Total sales revenue
    sales_count = country_data['SellingPrice'].count()  # Number of sales transactions
    
    # Return results
    return {'TotalSales': total_sales, 'SalesCount': sales_count}

def calculate_sales_for_last_n_weeks(df, country, start_date, end_date):
    country_data = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
    
    # Calculate total sales and count
    total_sales = country_data['SellingPrice'].sum()  # Total sales revenue
    sales_count = country_data.shape[0]  # Number of sales transactions
    
    # Return results
    return {'TotalSales': total_sales, 'SalesCount': sales_count}
def calculate_sales_for_last_n_days(df, country, start_date, end_date):
    country_data = df[(df['countryname'].str.lower()== country.lower()) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
    
    # Calculate total sales and count
    total_sales = country_data['SellingPrice'].sum()  # Total sales revenue
    sales_count = country_data.shape[0]  # Number of sales transactions
    
    # Return results
    return {'TotalSales': total_sales, 'SalesCount': sales_count}
def calculate_country_sales_by_fortnight(df, country, start_date, end_date):
    try:
        # Ensure start_date and end_date are datetime objects
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')
        
        if pd.isna(start_date) or pd.isna(end_date):
            logging.error(f"Invalid date range provided: start_date={start_date}, end_date={end_date}")
            return 0, 0.0
        
        # Ensure 'PurchaseDate' is converted to datetime for comparison
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        
        # Filter data for the specified country and date range
        fortnight_sales = df[(df['countryname'].str.lower() == country.lower()) &
                             (df['PurchaseDate'] >= start_date) &
                             (df['PurchaseDate'] <= end_date)]
        
        # Calculate sales count and total price
        total_sales_count = fortnight_sales['SellingPrice'].count()
        total_sales_price = fortnight_sales['SellingPrice'].sum()
        
        return total_sales_count, total_sales_price
    except Exception as e:
        logging.error(f"Error calculating sales for fortnight {start_date} to {end_date} in {country}: {e}")
        return 0,0.0
    
    
def calculate_country_sales_for_today(df, country, text):
    try:
        # Ensure 'text' is a string before processing
        if not isinstance(text, str):
            logging.info(f"Invalid input for text: Expected string, got {type(text)}")
            return 0, 0.0
        # Validate if the input text refers to "today"
        if not extract_today(text):
            logging.info("The text does not refer to 'today'.")
            return 0, 0.0

        # Ensure 'PurchaseDate' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

        today = datetime.now().date()
        today_sales = df[
            (df['countryname'].str.lower() == country.lower()) & 
            (df['PurchaseDate'].dt.date == today)
        ]
        
        # Calculate sales count and revenue
        total_sales_count = today_sales['SellingPrice'].count()
        total_sales_price = pd.to_numeric(today_sales['SellingPrice'], errors='coerce').sum()

        return total_sales_count, total_sales_price

    except Exception as e:
        logging.error(f"Error in calculate_country_sales_for_today: {e}")
        return 0, 0.0

def calculate_country_sales_for_last_day(df, country, text):
    try:
        # Ensure 'text' is a string before processing
        if not isinstance(text, str):
            logging.info(f"Invalid input for text: Expected string, got {type(text)}")
            return 0, 0.0

        # Validate if the input text refers to "yesterday"
        if not extract_last_day(text):
            logging.info("The text does not refer to 'last day'.")
            return 0, 0.0
        # Ensure 'PurchaseDate' is in datetime format
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

        # Calculate the date for the last day
        last_day = (datetime.now() - timedelta(days=1)).date()
        last_day_sales = df[
            (df['countryname'].str.lower() == country.lower()) & 
            (df['PurchaseDate'].dt.date == last_day)
        ]
        
        # Calculate sales count and revenue
        total_sales_count = last_day_sales['SellingPrice'].count()
        total_sales_price = pd.to_numeric(last_day_sales['SellingPrice'], errors='coerce').sum()

        return total_sales_count, total_sales_price

    except Exception as e:
        logging.error(f"Error in calculate_country_sales_for_last_day: {e}")
        return 0, 0.0

def calculate_country_sales_for_specific_date(df, country, date):
    try:
        # Ensure 'PurchaseDate' is in datetime format
        # cleaned_date = clean_date_string(specific_date)
        date = pd.to_datetime(date, errors='coerce').date()
        
        if pd.isna(date):
            return None, "Error: The provided date is invalid."
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

        # Check if any dates couldn't be converted
        if df['PurchaseDate'].isnull().any():
            logging.error("Some 'PurchaseDate' values are invalid.")
            return None, "Error: Some dates in 'PurchaseDate' could not be converted."

        # Filter the dataframe by country (case-insensitive)
        country_sales_df = df[df['countryname'].str.lower() == country.lower()]

        if country_sales_df.empty:
            logging.info(f"No sales data found for country {country}.")
            return 0, 0.0  # No sales for the country
        

        # Filter sales data for the specific date
        daily_sales_df = country_sales_df[country_sales_df['PurchaseDate'].dt.date == date]
        
        # Calculate the count and total sales price
        daily_sales_count = daily_sales_df['SellingPrice'].count()
        daily_sales_price = daily_sales_df['SellingPrice'].sum()

        if daily_sales_count == 0:
            logging.info(f"No sales found for {country} on {date}")
            return 0, 0.0  # No sales found for the specific country and date
        daily_sales_price = float(daily_sales_price)


        return daily_sales_count, daily_sales_price
    
    except Exception as e:
        logging.error(f"Error extracting sales for {country} on {date}: {e}")
        return None, f"Error occurred: {str(e)}"
def calculate_country_sales_by_monthwise(df, country):
    df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
    # Group by Country and Month
    df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')  # Extract Month-Year
    monthly_sales = df[df['countryname'].str.lower() == country.lower()].groupby('MonthYear').agg(
        TotalSales=('SellingPrice', 'sum'),
        SalesCount=('SellingPrice', 'count'),
        AvgSalesPrice=('SellingPrice', 'mean')
    ).reset_index()
    monthly_sales['SalesPercentageIncrease'] = monthly_sales['SalesCount'].pct_change() * 100
    
    return monthly_sales

# Function to calculate country sales by year
def calculate_country_sales_by_yearwise(df, country):
    # Group by Country and Year
    df['Year'] = df['PurchaseDate'].dt.year  # Extract Year
    yearly_sales = df[df['countryname'].str.lower() == country.lower()].groupby('Year').agg(
        TotalSales=('SellingPrice', 'sum'),
        SalesCount=('SellingPrice', 'count'),
        AvgSalesPrice=('SellingPrice', 'mean')
    ).reset_index()
    yearly_sales['SalesPercentageIncrease'] = yearly_sales['SalesCount'].pct_change() * 100
    
    
    return yearly_sales

 ##################################################region sales#############################       
def calculate_region_sales(df, region):
    """Calculates total sales and revenue for the given region."""
    region_sales = df[df['regionname'].str.lower() == region.lower()]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue
    

def calculate_region_sales_by_month_year(df, region, month, year):
    """Calculates total sales and revenue for a region for a specific month and year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.month == month) &
                      (df['PurchaseDate'].dt.year == year)]
                      
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_region_sales_by_year(df, region, year):
    """Calculates total sales and revenue for a region for a specific year."""
    region_sales = df[(df['regionname'].str.lower() == region.lower()) & (df['PurchaseDate'].dt.year == year)]
    total_sales = region_sales['SellingPrice'].count()
    total_revenue = region_sales['SellingPrice'].sum()
    return total_sales, total_revenue

def calculate_total_region_sales(df, region):
    """Calculates total sales and revenue for the given region (all years and months)."""
    return calculate_region_sales(df, region)

#######################################helping function for planname sales##################################
def calculate_plannane_sales_for_specific_date(df, planname, date):
    try:
        # Ensure 'PurchaseDate' is in datetime format
        # cleaned_date = clean_date_string(specific_date)
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

        # Check if any dates couldn't be converted
        if df['PurchaseDate'].isnull().any():
            logging.error("Some 'PurchaseDate' values are invalid.")
            return None, "Error: Some dates in 'PurchaseDate' could not be converted."

        # Filter the dataframe by country (case-insensitive)
        plan_sales_df = df[df['PlanName'] == planname]

        if plan_sales_df.empty:
            logging.info("No sales data found  ")
            return 0, 0.0  # No sales for the country
    
        # Filter sales data for the specific date
        daily_sales_df = plan_sales_df[plan_sales_df['PurchaseDate'].dt.date == date]
        
        # Calculate the count and total sales price
        daily_sales_count = daily_sales_df['SellingPrice'].count()
        daily_sales_price = daily_sales_df['SellingPrice'].sum()

        if daily_sales_count == 0:
            logging.info(f"No sales found  on {date}")
            return 0, 0.0  # No sales found for the specific country and date

        return daily_sales_count, daily_sales_price
    
    except Exception as e:
        logging.error(f"Error extracting sales on {date}: {e}")
        return None, f"Error occurred: {str(e)}"


def calculate_planname_sales_by_month_year(df, planname, month, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['month'] == month) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_planname_sales_by_year(df, planname, year):
    if 'month' not in df.columns or 'year' not in df.columns:
        # Convert 'PurchaseDate' (or equivalent) to datetime and extract month and year
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
        df['month'] = df['PurchaseDate'].dt.month
        df['year'] = df['PurchaseDate'].dt.year
    filtered_df = df[(df['PlanName'] == planname) & (df['year'] == year)]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue = filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_planname_sales_by_quarter(df, planname, start_month, end_month, year):
    filtered = df[
        (df['PlanName'] == planname) &
        (df['PurchaseDate'].dt.month >= start_month) &
        (df['PurchaseDate'].dt.month <= end_month) &
        (df['PurchaseDate'].dt.year == year)
    ]
    sales_count = filtered['SellingPrice'].count()
    total_revenue = filtered['SellingPrice'].sum()
    return sales_count, total_revenue

def calculate_plansales_for_last_n_days(df, plan, start_date, end_date):

    filtered = df[
        (df['PlanName'] == plan) &
        (df['PurchaseDate'] >= start_date) &
        (df['PurchaseDate'] <= end_date)
    ]
    
    # Calculate sales count and total revenue
    sales_count = filtered['SellingPrice'].count()  # Number of sales
    total_revenue = filtered['SellingPrice'].sum()  # Total revenue
    
    return sales_count, total_revenue

def calculate_planname_for_last_n_hours(df, plan, start_time, end_time):
    # Filter the dataframe for the given plan and time range
    filtered = df[
        (df['PlanName'] == plan) &
        (df['PurchaseDate'] >= start_time) &
        (df['PurchaseDate'] <= end_time)
    ]
    
    # Calculate sales count and total revenue
    sales_count = filtered['SellingPrice'].count()  # Number of sales
    total_revenue = filtered['SellingPrice'].sum()  # Total revenue
    
    return sales_count, total_revenue



def calculate_plansales_for_last_n_weeks(df, plan, start_date, end_date):

    # Filter the dataframe for the given plan and the date range
    filtered = df[
        (df['PlanName'] == plan) &
        (df['PurchaseDate'] >= start_date) &
        (df['PurchaseDate'] <= end_date)
    ]
    
    # Calculate sales count and total revenue
    sales_count = filtered['SellingPrice'].count()  # Number of sales
    total_revenue = filtered['SellingPrice'].sum()  # Total revenue
    
    return sales_count, total_revenue


def calculate_planname_sales_by_last_day(df, planname, text):
    try:
        if not isinstance(text, str):
            logging.info(f"Invalid input for text: Expected string, got {type(text)}")
            return 0, 0.0
        if not extract_last_day(text):
            logging.info("The text does not refer to 'last day'.")
            return 0, 0.0
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        last_day = (datetime.now() - timedelta(days=1)).date()

        
        last_day_sales = df[
            (df['PlanName'] == planname) & 
            (df['PurchaseDate'].dt.date == last_day)
        ]

        # Calculate total sales count and revenue
        total_sales = last_day_sales['SellingPrice'].count() 
        total_revenue = last_day_sales['SellingPrice'].sum() 

        return total_sales, total_revenue

    except Exception as e:
        logging.error(f"Error in calculate_planname_sales_by_last_day: {e}")
        return 0, 0.0

def calculate_planname_sales_by_today(df, plan, text):
    try:
        # Ensure 'today_date' is a valid date
        if not isinstance(text, str):
            logging.info(f"Invalid input for text: Expected string, got {type(text)}")
            return 0, 0.0

        # Ensure 'PlanName' exists in the DataFrame and validate the input
        if not extract_today(text):
            logging.info("The text does not refer to 'today'.")
            return 0, 0.0
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

        today = datetime.now().date()

        # Filter data for the given plan and today's date
        today_sales = df[
            (df['PlanName'] == plan) & 
            (df['PurchaseDate'].dt.date == today)
        ]

        # Calculate total sales count and revenue
        total_sales = today_sales['SellingPrice'].count()
        total_revenue = today_sales['SellingPrice'].sum()

        return total_sales, total_revenue

    except Exception as e:
        logging.error(f"Error in calculate_planname_sales_by_today: {e}")
        return 0, 0.0
        
def calculate_planname_sales_for_fortnight(df, plan, start_date, end_date):
    try:
        # Ensure start_date and end_date are datetime objects
        start_date = pd.to_datetime(start_date, errors='coerce')
        end_date = pd.to_datetime(end_date, errors='coerce')
        
        if pd.isna(start_date) or pd.isna(end_date):
            logging.error(f"Invalid date range provided: start_date={start_date}, end_date={end_date}")
            return 0, 0.0
        
        # Ensure 'PurchaseDate' is converted to datetime for comparison
        if not pd.api.types.is_datetime64_any_dtype(df['PurchaseDate']):
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        
        # Filter data for the specified country and date range
        fortnight_sales = df[(df['PlanName'] == planname) &
                             (df['PurchaseDate'] >= start_date) &
                             (df['PurchaseDate'] <= end_date)]
        
        # Calculate sales count and total price
        total_sales_count = fortnight_sales['SellingPrice'].count()
        total_sales_price = fortnight_sales['SellingPrice'].sum()
        
        return total_sales_count, total_sales_price
    except Exception as e:
        logging.error(f"Error calculating sales for fortnight {start_date} to {end_date} in {country}: {e}")
        return 0,0.0

def calculate_total_planname_sales(df, planname):
    filtered_df = df[df['PlanName'] == planname]
    sales_count = filtered_df['SellingPrice'].count()
    total_revenue =  filtered_df['SellingPrice'].sum()
    return sales_count, total_revenue
#####################################################helping function for top sales plans##################################

def calculate_top_sales_plans(df, year=None, month=None):
    # Implement logic to calculate top selling plans
    # Example:
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nlargest(10, 'SalesCount')

def calculate_least_sales_plans(df, year=None, month=None):
    # Implement logic to calculate least selling plans
    if year:
        df = df[df['PurchaseDate'].dt.year == year]
    if month:
        df = df[df['PurchaseDate'].dt.month == month]
    return df.groupby('PlanName').agg(SalesCount=('SellingPrice', 'count'), TotalRevenue=('SellingPrice', 'sum')).nsmallest(10, 'SalesCount')

def extract_sales_in_date_range(df, start_date, end_date):
    try:
        # Ensure dates are in datetime format
        start_date = pd.to_datetime(start_date)
        end_date = pd.to_datetime(end_date)
        # Filter the DataFrame for the given date range
        filtered_df = df[(df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]

        # Calculate count and sum
        sales_count = filtered_df['SellingPrice'].count()
        total_price = filtered_df['SellingPrice'].sum()
        
        return sales_count, total_price
    except Exception as e:
        logging.error(f"Error in extract_sales_in_date_range: {e}")
        return None, f"An error occurred while processing sales data: {e}"
    

def extract_profit_margin_sales_for_specific_date(df, date):
    try:
        # Convert the specific_date to a datetime object
        date = pd.to_datetime(date, errors='coerce').date()
        
        if pd.isna(date):
            return None, "Error: The provided date is invalid."
        
        # Ensure 'PurchaseDate' is converted to datetime and handle any NaT values
        df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
        
        # Check if the conversion was successful
        if df['PurchaseDate'].isnull().any():
            return None
        df = df.dropna(subset=['PurchaseDate', 'ProfitMargin'])
        # Filter and calculate sales
        daily_profit_margin = df[df['PurchaseDate'].dt.date == date]['ProfitMargin'].sum()
        
        
        return daily_profit_margin, None
    
    except Exception as e:
        logging.error(f"Error occurred while calculating profit margin: {str(e)}")
        return None
#######################TOTALSALES#############################################################################################

class ActionGetTotalSales(Action):
    def name(self) -> Text:
        return "action_get_total_sales"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            global df
            logging.info("Running ActionGetTotalSales...")
            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")

            if df.empty or df['SellingPrice'].isnull().all():
                response_message = {
                    "text": "The sales data is empty or invalid\n."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            last_n_hours = extract_last_n_hours(user_message)
            last_n_weeks = extract_last_n_weeks(user_message)
            last_n_days = extract_last_n_days(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            monthwise = extract_monthwise(user_message)
            yearwise = extract_yearwise(user_message)
            total_sales_count = 0
            total_sales_price = 0.0
            response_message = ""
            specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M').astype(str)
            df['Year'] = df['PurchaseDate'].dt.year
           

            if specific_date_text:
                logging.info(f"Processing sales data for specific date: {specific_date_text}")
                
                daily_sales_count, daily_sales_price = extract_sales_for_specific_date(df, specific_date_text)
                
                if daily_sales_count > 0:
                    response_message = {
                        "text": f"The total sales for {specific_date_text} is {daily_sales_count} with a sales price of ${daily_sales_price:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded on {specific_date_text}."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            
            # Today's sales
            if today:
                today_date = datetime.now().date()
                logging.info(f"Processing today's sales...{today_date}")
                today_sales_count = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].count()
                today_sales_price = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].sum()
                if today_sales_count> 0:
                    response_message = {
                        "text": f"The total sales for today ({today_date}) is {today_sales_count} with a sales price of ${today_sales_price:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded today ({today_date})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Yesterday's sales
            if last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Processing yesterday's sales...{lastday}")
                yesterday_sales_count = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].count()
                yesterday_sales_price = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].sum()
                if yesterday_sales_count > 0:
                    response_message = {
                        "text": f"The total sales for yesterday ({lastday}) is {yesterday_sales_count} with a sales price of ${yesterday_sales_price:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded yesterday ({lastday})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
                
            
            if monthwise:
                logging.info("processing monthwise sales")
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce') 
                df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
                # Extract Month-Year
                monthly_data = df.groupby("MonthYear").agg(
                    TotalSales=('SellingPrice', 'count'),
                    SalesCount=('SellingPrice', 'sum')
                ).reset_index()
                response_message = {
                    "text": "",
                    "tables": []
                }
                if monthly_data.empty:
                    response_message["text"] += "No monthly sales data available."
                else:
                    response_message["text"] += "📅 Monthly Sales Summary:\n"
                    table = []
                    for _, row in monthly_data.iterrows():
                        table.append([
                            str(row['MonthYear']), 
                            int(row['SalesCount']),
                            f"${row['TotalSales']:,.2f}",
                            
                        ])
                    response_message["tables"].append({
                        "headers": ["Month-Year", "Total Sales count", "Total Sales revenue"],
                        "data": table
                    })
            
                dispatcher.utter_message(json_message=response_message)
                return []
                        
                           
            if yearwise:
                logging.info("processing yearwise sales")
                df['Year'] = df['PurchaseDate'].dt.year  # Extract Year
                yearly_data = df.groupby("Year").agg(
                    TotalSales=('SellingPrice', 'sum'),
                    SalesCount=('SellingPrice', 'count')
                ).reset_index()
        
                # response_message += "📅 Yearly Sales Summary:\n"
                response_message = {
                    "text": "",
                    "tables": []
                }
                if yearly_data.empty:
                    response_message["text"] += "No yearly sales data available."
                else:
                    response_message["text"] += "📅 Yearly Sales Summary:\n"
                    table = []
                    for _, row in yearly_data.iterrows():
                        table.append([
                            str(int(row['Year'])),
                            int(row['SalesCount']),
                            f"${row['TotalSales']:.2f}"
                        ])
                    
                    response_message["tables"].append({
                        "headers": ["Year", "Total Sales Count", "Total Sales Revenue"],
                        "data": table
                    })
                
                dispatcher.utter_message(json_message=response_message)
                return []
            # Handle half-year request
            if half_year:
                logging.info(f"Processing sales data for half-year... {half_year}")
                year,(start_month, end_month) = half_year
               
                half_year_sales_count = df[
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == year)
                ]['SellingPrice'].count()

                half_year_sales_price = df[
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == year)
                ]['SellingPrice'].sum()

                half_name = "First Half" if start_month == 1 else "Second Half"
                if half_year_sales_count == 0:
                    response_message = {
                        "text": f"There were no sales recorded for {half_name} of {year}."
                    }
                else:
                    response_message = {
                        "text": f"The total sales count for {half_name} of {year} is {half_year_sales_count} and the total sales revenue is ${half_year_sales_price:.2f}."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []

            #Handle fortnight request
            if fortnight:
                logging.info(f"Processing sales data for fortnight...{fortnight}")
                start_date, end_date = fortnight
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                logging.info(f"Start date: {start_date}, End date: {end_date}")
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                count_fortnight = df[
                    (df['PurchaseDate'] >= start_date) & 
                    (df['PurchaseDate'] <=  end_date)
                ]['SellingPrice'].count()
            
                price_fortnight = df[
                    (df['PurchaseDate'] >= start_date) & 
                    (df['PurchaseDate'] <= end_date)
                ]['SellingPrice'].sum()
                if count_fortnight > 0:
                    response_message = {
                        "text": f"The total sales for the last fortnight ({start_date_formatted} to {end_date_formatted}) is {count_fortnight} with a sales price of ${price_fortnight:.2f}."
                    }
                else:
                    response_message = {
                    "text": f"No sales were recorded for the fortnight ({start_date_formatted} to {end_date_formatted})."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            #Handle last N months request
            if last_n_months:
                logging.info(f"Processing sales data for the last N months...{last_n_months}")
                start_date, end_date, num_months= last_n_months
                count_last_n_months, price_last_n_months = extract_sales_in_date_range(df, start_date, end_date)
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                
                if count_last_n_months > 0:
                    response_message = {
                        "text": f"The total sales in the last {num_months} months ({start_date_formatted} to {end_date_formatted}) is {count_last_n_months} with a sales price of ${price_last_n_months:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded in the last {num_months} months ({start_date_formatted} to {end_date_formatted})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            if last_n_hours:
                logging.info(f"Processing sales data for the last N hours...{last_n_hours}")
                start_time, end_time, num_hours = last_n_hours
                count_last_n_hours = df[
                    (df['PurchaseDate'] >= start_time) &
                    (df['PurchaseDate'] <= end_time) 
                ]['SellingPrice'].count()
                
                price_last_n_hours = df[
                    (df['PurchaseDate'] >= start_time) &
                    (df['PurchaseDate'] <= end_time)
                ]['SellingPrice'].sum()
                if count_last_n_hours > 0:
                    response_message = {
                        "text": f"The total sales in the last {num_hours} hours ({start_time} to {end_time}) is {count_last_n_hours} with a sales price of ${price_last_n_hours:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded in the last {num_hours} hours ({start_time} to {end_time})."
                    }
                    
                dispatcher.utter_message(json_message=response_message)
                return []
            if last_n_days:
                logging.info(f"Processing sales data for the last N days...{last_n_days}")
                start_date, end_date, num_days = last_n_days
                count_last_n_days = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date) 
                    ]['SellingPrice'].count()
                    
                price_last_n_days = df[
                    (df['PurchaseDate'] >= start_date) &
                    (df['PurchaseDate'] <= end_date)
                ]['SellingPrice'].sum()
            
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                
                if count_last_n_days > 0:
                    response_message = {
                        "text": f"The total sales in the last {num_days} days ({start_date_formatted} to {end_date_formatted}) is {count_last_n_days} with a sales price of ${price_last_n_days:.2f}."
                    }
        
                else:
                    response_message = {
                        "text": f"No sales were recorded in the last {num_days} days ({start_date_formatted} to {end_date_formatted})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            if last_n_weeks:
                logging.info(f"Processing sales data for the last N weeks...{last_n_weeks}")
                start_date, end_date, num_weeks = last_n_weeks
                count_last_n_weeks = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date) 
                    ]['SellingPrice'].count()
                    
                price_last_n_weeks = df[
                    (df['PurchaseDate'] >= start_date) &
                    (df['PurchaseDate'] <= end_date)
                ]['SellingPrice'].sum()
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
    
                if count_last_n_weeks > 0:
                    response_message = {
                        "text": f"The total sales in the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}) is {count_last_n_weeks} with a sales price of ${price_last_n_weeks:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No sales were recorded in the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
                
                        
            if quarterly:
                logging.info(f"Processing quarterly sales... {quarterly}")
                try:
                    (start_month, end_month), year = quarterly

                    
                    # Filter data for the quarter
                    quarterly_sales_count = df[
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) &
                        (df['PurchaseDate'].dt.year == year)
                    ]['SellingPrice'].count()
                    
                    quarterly_sales_price = df[
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) &
                        (df['PurchaseDate'].dt.year == year)
                    ]['SellingPrice'].sum()

                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                    if quarterly_sales_count > 0:
                        response_message = {
                            "text": f"The total sales count for the {quarter_name} of {year} is {quarterly_sales_count} and sales price is ${quarterly_sales_price:.2f}."
                        }
                    else:
                        response_message = {
                            "text": f"There are no sales recorded for {quarter_name} of {year}."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing monthly sales: {str(e)}"
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
           
            if month_year_pairs:
                logging.info(f"sales with month - year... {month_year_pairs}")
                try:
                    total_sales_count = total_sales_price = 0.0
                    for month, year in month_year_pairs:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].sum()
                        if monthly_sales_count == 0:
                            response_message = {
                                "text": f"No sales were recorded on {months_reverse[month]} {year}."
                            }
                        else:
                            response_message = {
                                "text": f"The total sales count for {months_reverse[month]} {year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing monthly sales: {str(e)}"
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
                
        

            if years:
                logging.info(f"sales with year...: {years}")
                try:
                    for year in years:
                        total_sales_count = df[(df['PurchaseDate'].dt.year == year)]['SellingPrice'].count()
                        total_sales_price = df[(df['PurchaseDate'].dt.year == year)]['SellingPrice'].sum()
                        
                        if total_sales_count == 0:
                            response_message = {
                                "text": f"No sales were recorded on {year}."
                            }
                        else:
                            response_message = {
                                "text": f"The total sales count for {year} is {total_sales_count} and sales price is ${total_sales_price:.2f}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                except Exception as e:
                    response_message = {
                            "text": f"Error occurred while processing monthly sales: {str(e)}"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                    

            if months_extracted:
                logging.info(f"month with current year...:  {months_extracted}")
                try:
                    current_year = pd.to_datetime('today').year
                    monthly_sales_data = []
                    for month in months_extracted:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price
                        if total_sales_count == 0:
                            response_message = {
                                "text": f"No sales were recorded on {months_reverse[month]} {current_year}."
                            }
                        else:
                            response_message = {
                                "text": f"The total sales count for {months_reverse[month]} {current_year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing monthly sales: {str(e)}"
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            logging.info("processing total sales.....")
                
            total_sales_count, total_sales_price = self.calculate_total_sales(df)
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            response_message = {
                "text": f"The overall (from {start_date} to {end_date}) sales count is {total_sales_count} with a sales price of ${total_sales_price:.2f}."
            }
            dispatcher.utter_message(json_message=response_message)
            return []
        except Exception as e:
            response_message = {
                "text": f"Error occurred n processing your request: {str(e)}"
            }
            dispatcher.utter_message(json_message=response_message)
            return []
        
    def calculate_total_sales(self, df: pd.DataFrame) -> Tuple[int, float]:
        total_sales_count = int(df['SellingPrice'].count()) if 'SellingPrice' in df.columns else 0
        total_sales_price = float(df['SellingPrice'].sum()) if 'SellingPrice' in df.columns else 0.0
        return total_sales_count, total_sales_price

###############################################COMPARE SALES################################################################################



def remove_ordinal_suffix(date_str):
    """Remove ordinal suffixes (st, nd, rd, th) from the date string."""
    return re.sub(r'(\d+)(st|nd|rd|th)', r'\1', date_str)


def extract_quarters_and_years(text):
    quarter_pattern = r'\b(?:quarter\s*(1|2|3|4)|q(1|2|3|4)|first|second|third|fourth)\b'
    year_pattern = r'\b(20\d{2})\b'  # Matches years like 2023, 2024, etc.

    # Find all matches for quarters
    quarter_matches = re.findall(quarter_pattern, text, re.IGNORECASE)
    # Find all matches for years
    year_matches = re.findall(year_pattern, text)

    # Normalize quarter matches into quarter numbers
    quarter_map = {
        'first': 1,
        'second': 2,
        'third': 3,
        'fourth': 4
    }

    quarters = []
    for match in quarter_matches:
        if match[0]:  # Matches "quarter 1", "quarter 2", etc.
            quarters.append(int(match[0]))
        elif match[1]:  # Matches "q1", "q2", etc.
            quarters.append(int(match[1]))
        else:  # Matches "first", "second", "third", "fourth"
            quarter_name = match[0].lower()
            quarter_number = quarter_map.get(quarter_name)
            if quarter_number:
                quarters.append(quarter_number)

    # Ensure only two unique quarters are considered
    if len(quarters) < 2:
        return None

    quarters = quarters[:2]  # Only consider the first two quarters

    # Map quarter numbers to their month ranges
    quarter_month_map = {
        1: (1, 3),   # Q1: January to March
        2: (4, 6),   # Q2: April to June
        3: (7, 9),   # Q3: July to September
        4: (10, 12)  # Q4: October to December
    }

    quarter_1 = quarter_month_map.get(quarters[0])
    quarter_2 = quarter_month_map.get(quarters[1])
    current_year = datetime.now().year
    year1 = int(year_matches[0]) if len(year_matches) > 0 else current_year
    year2 = int(year_matches[1]) if len(year_matches) > 1 else current_year

    return (quarter_1, year1, quarter_2, year2)


class ActionCompareSales(Action):
    def name(self) -> str:
        return "action_compare_sales"

    def run(self, dispatcher, tracker, domain) -> list[Dict[Text, Any]]:
        global df
        try:
            logging.info("Running ActionCompareSales...")
            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")
           
            if not user_message:
                response_message = {
                    "text": "I didn't receive any message for comparison. Please specify a time range."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Ensure necessary columns are present in the dataset
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                response_message = {
                    "text": "Required columns ('PurchaseDate', 'SellingPrice') are missing in the dataset."
                }
                dispatcher.utter_message(json_message=response_message)
                return []


            month_pattern = r"(\w+ \d{4}) to (\w+ \d{4})"
            year_pattern = r"(\d{4}) to (\d{4})"
            compare_quarters =  extract_quarters_and_years(user_message)
            compare_dates = extract_date(user_message)
            date_range_pattern = r"(\d{1,2}(st|nd|rd|th)?\s?[A-Za-z]+\s\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4})\s*(?:to|-)\s*(\d{1,2}(st|nd|rd|th)?\s?[A-Za-z]+\s\d{4}|\d{1,2}[/-]\d{1,2}[/-]\d{4})"
            
            

            # Check for month comparison
            month_matches = re.findall(month_pattern, user_message)
            if month_matches:
                logging.info("month year comparison...")
                try:
                    month1, month2 = month_matches[0]
                    logging.info(f"Extracted months: {month1} and {month2}")

                    # Parse dates with fallback for abbreviated or full month names
                    def parse_month_year(date_str):
                        for fmt in ("%B %Y", "%b %Y"):  # Try full and short month names
                            try:
                                return datetime.strptime(date_str, fmt)
                            except ValueError:
                                continue
                        raise ValueError(f"Unable to parse date: {date_str}")

                    start_date_1 = parse_month_year(month1)
                    start_date_2 = parse_month_year(month2)

                    logging.info(f"Comparing sales for {month1} and {month2}...")
                    
                    sales_count_1 = int(df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].count())
                    sales_price_1 = float(df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].sum())
                    
                    sales_count_2 = int(df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].count())
                    sales_price_2 = float(df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].sum())

                    table = [
                        [month1, sales_count_1, f"${sales_price_1:.2f}"],
                        [month2, sales_count_2, f"${sales_price_2:.2f}"]
                    ]
                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2
                    response_message = {
                        "text": f"📊 Sales Comparison Between {month1} and {month2}:\n",
                        "tables": [
                            {
                                "headers": ["Month", "Total Sales", "Sales Revenue"],
                                "data": table
                            }
                        ]
                    }
                    if count_difference > 0:
                        response_message["text"] += f"\n\nDifference in sales: {month1} had {abs(count_difference)} more sales than {month2}.\n"
                    elif count_difference < 0:
                        response_message["text"] += f"\n\nDifference in sales: {month2} had {abs(count_difference)} more sales than {month1}.\n"
                    else:
                        response_message["text"] += "\n\nBoth months had the same number of sales.\n"
            
                    # Adding difference in revenue
                    if price_difference > 0:
                        response_message["text"] += f"\n\nDifference in revenue: {month1} generated ${abs(price_difference):.2f} more in sales revenue than {month2}."
                    elif price_difference < 0:
                        response_message["text"] += f"\n\nDifference in revenue: {month2} generated ${abs(price_difference):.2f} more in sales revenue than {month1}."
                    else:
                        response_message["text"] += "\n\nBoth months generated the same sales revenue."
            
                    # Send the response as a JSON message
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except ValueError as ve:
                    logging.error(f"Date parsing error for month comparison: {ve}")
                
                    response_message = {
                    "text": "Please provide a valid comparison in the format "
                }
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            # Check for year comparison
            year_matches = re.findall(year_pattern, user_message)
            if year_matches:
                logging.info("year comparison...")
                try:
                    year1, year2 = year_matches[0]
                    sales_count_1 = int(df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].count())
                    sales_price_1 = float(df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].sum())
                    
                    sales_count_2 = int(df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].count())
                    sales_price_2 = float(df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].sum())

                    table = [
                        [year1, sales_count_1, f"${sales_price_1:.2f}"],
                        [year2, sales_count_2, f"${sales_price_2:.2f}"]
                    ]

                    headers = ["Year", "Total Sales", "Total Revenue"]
                    comparison_table = tabulate(table, headers, tablefmt="grid")

                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2
                    response_message = {
                        "text": f"📊 Sales Comparison Between {year1} and {year2}:\n",
                        "tables": [
                            {
                                "headers": ["Year", "Total Sales", "Total Revenue"],
                                "data": table
                            }
                        ]
                    }

                    if count_difference > 0:
                        response_message["text"] += f"\n\nSales Count Difference: {year1} had {abs(count_difference)} more sales than {year2}.\n"
                    elif count_difference < 0:
                        response_message["text"] += f"\n\nSales Count Difference: {year2} had {abs(count_difference)} more sales than {year1}.\n"
                    else:
                        response_message["text"] += "\n\nBoth years had the same number of sales."

                    # Add revenue difference
                    if price_difference > 0:
                        response_message["text"] += f"\n\nRevenue Difference: {year1} generated ${abs(price_difference):.2f} more in sales revenue than {year2}."
                    elif price_difference < 0:
                        response_message["text"] += f"\n\nRevenue Difference: {year2} generated ${abs(price_difference):.2f} more in sales revenue than {year1}."
                    else:
                        response_message["text"] += "\n\nBoth years generated the same sales revenue."

                    dispatcher.utter_message(json_message=response_message)
                    return []
                except ValueError as ve:
                    logging.error(f"Date parsing error for year comparison: {ve}")
                    response_message = {
                        "text": "Please provide a valid comparison in the format 'Year to Year'."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
          
           
            date_matches = re.search(date_range_pattern, user_message)
            if date_matches:
                logging.info("Date-to-date comparison detected...")
                try:
                   
                    start_date_str, _, end_date_str = date_matches.groups()[:3]
                    start_date_str = remove_ordinal_suffix(start_date_str)
                    end_date_str = remove_ordinal_suffix(end_date_str)
        
                    start_date = dateparser.parse(start_date_str)
                    end_date = dateparser.parse(end_date_str)
                    if not start_date or not end_date:
                        return "Error parsing dates. Please ensure the dates are in a valid format."

                    start_date_str_formatted = start_date.strftime('%d-%m-%Y')
                    end_date_str_formatted = end_date.strftime('%d-%m-%Y')

                    start_sales_count = int(df[df['PurchaseDate'].dt.date == start_date.date() ]['SellingPrice'].count())
                    start_sales_revenue = float(df[df['PurchaseDate'].dt.date == start_date.date()]['SellingPrice'].sum())

                    end_sales_count = int(df[df['PurchaseDate'].dt.date == end_date.date()]['SellingPrice'].count())
                    end_sales_revenue = float(df[df['PurchaseDate'].dt.date == end_date.date()]['SellingPrice'].sum())

                    table = [
                        [start_date_str_formatted, start_sales_count, f"${start_sales_revenue:.2f}"],
                        [end_date_str_formatted, end_sales_count, f"${end_sales_revenue:.2f}"]
                    ]


                    headers = ["Date", "Total Sales", "Sales Revenue"]
                    count_difference = start_sales_count - end_sales_count
                    price_difference = start_sales_revenue - end_sales_revenue
                    response_message = {
                        "text": f"📊 Sales Comparison Between {start_date_str_formatted} and {end_date_str_formatted}:",
                        "tables": [
                            {
                                "headers": headers,
                                "data": table
                            }
                        ]
                    }
                    if count_difference > 0:
                        response_message["text"] += f"\n\nDifference in sales: {start_date_str_formatted} had {abs(count_difference)} more sales than {end_date_str_formatted}."
                    elif count_difference < 0:
                        response_message["text"] += f"\n\nDifference in sales: {end_date_str_formatted} had {abs(count_difference)} more sales than {start_date_str_formatted}."
                    else:
                        response_message["text"] += "\n\nBoth dates had the same number of sales."

                    # Add revenue difference
                    if price_difference > 0:
                        response_message["text"] += f"\n\nDifference in revenue: {start_date_str_formatted} generated ${abs(price_difference):.2f} more in sales revenue than {end_date_str_formatted}."
                    elif price_difference < 0:
                        response_message["text"] += f"\n\nDifference in revenue: {end_date_str_formatted} generated ${abs(price_difference):.2f} more in sales revenue than {start_date_str_formatted}."
                    else:
                        response_message["text"] += "\n\nBoth dates generated the same sales revenue."

                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error during date range comparison: {e}")
                    dispatcher.utter_message(json_message={
                        "text": "Error parsing date range. Please try again with a valid date range format."
                    })
                    return []
                    
            
            if compare_quarters:
                logging.info("quarter comparison...")
                (start_month_1, end_month_1), year1, (start_month_2, end_month_2), year2 = compare_quarters
                
                try:
                   
                    sales_count_1 = int(df[(df['PurchaseDate'].dt.year == year1) & 
                                        (df['PurchaseDate'].dt.month >= start_month_1) & 
                                        (df['PurchaseDate'].dt.month <= end_month_1)]['SellingPrice'].count())
                    sales_price_1 = float(df[(df['PurchaseDate'].dt.year == year1) & 
                                        (df['PurchaseDate'].dt.month >= start_month_1) & 
                                        (df['PurchaseDate'].dt.month <= end_month_1)]['SellingPrice'].sum())
                    sales_count_2 = int(df[(df['PurchaseDate'].dt.year == year2) & 
                               (df['PurchaseDate'].dt.month >= start_month_2) & 
                               (df['PurchaseDate'].dt.month <= end_month_2)]['SellingPrice'].count())
                    sales_price_2 = float(df[(df['PurchaseDate'].dt.year == year2) & 
                                       (df['PurchaseDate'].dt.month >= start_month_2) & 
                                       (df['PurchaseDate'].dt.month <= end_month_2)]['SellingPrice'].sum())
                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name_1 = quarter_name_map.get((start_month_1, end_month_1), "Quarter")
                    quarter_name_2 = quarter_name_map.get((start_month_2, end_month_2), "Quarter")


                    table = [
                        [f"{quarter_name_1} of {year1}", sales_count_1, f"${sales_price_1:.2f}"],
                        [f"{quarter_name_2} of {year2}", sales_count_2, f"${sales_price_2:.2f}"]
                    ]

                    headers = ["Quarter", "Total Sales", "Total Revenue"]
                    
                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2
                    response_message = {
                        "text": f"📊 Sales Comparison Between {quarter_name_1} of {year1} and {quarter_name_2} of {year2}:",
                        "tables": [
                            {
                                "headers": headers,
                                "data": table
                            }
                        ]
                    }

                    if count_difference > 0:
                        response_message["text"] += f"\n\nDifference in sales: {quarter_name_1} of {year1} had {abs(count_difference)} more sales than {quarter_name_2} of {year2}."
                    elif count_difference < 0:
                        response_message["text"] += f"\n\nDifference in sales: {quarter_name_2} of {year2} had {abs(count_difference)} more sales than {quarter_name_1} of {year1}."
                    else:
                        response_message["text"] += "\n\nBoth quarters had the same number of sales."

                    # Add revenue difference
                    if price_difference > 0:
                        response_message["text"] += f"\n\nDifference in revenue: {quarter_name_1} of {year1} generated ${abs(price_difference):.2f} more in sales revenue than {quarter_name_2} of {year2}."
                    elif price_difference < 0:
                        response_message["text"] += f"\n\nDifference in revenue: {quarter_name_2} of {year2} generated ${abs(price_difference):.2f} more in sales revenue than {quarter_name_1} of {year1}."
                    else:
                        response_message["text"] += "\n\nBoth quarters generated the same sales revenue."

                    # Send the response as a JSON message
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Unexpected error during quarter comparison: {e}")
                    dispatcher.utter_message(json_message={
                        "text": "An error occurred while processing your request. Please try again."
                    })
                    return []
        
            dispatcher.utter_message(json_message={
                "text": "Please provide a valid comparison in the format 'month year to month year' or 'quarter to quarter' or 'year to year'."
            })
            return []
        except Exception as e:
            logging.error(f"Unexpected error in ActionCompareQuarterSales: {e}")
            dispatcher.utter_message(json_message={
                "text": "An error occurred while processing your request. Please try again later."
            })
            return []




##############################################################salesforcountry###########################################################################

class ActionCountrySales(Action):
    def name(self) -> str:
        return "action_country_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        global df
        logging.info("Running ActionCountrySales...")

        try:
            # Check if the dataframe is empty
            if df.empty:
                response_message = {
                    "text": "The sales data is empty or invalid\n."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
             
            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")

            

            # Extract country from the user message
            country_extracted = extract_country_from_text(user_message)
            logging.info(f"Initial extracted country: {country_extracted}")
    
        
            if not country_extracted:
                logging.info(f"Country {country_extracted} not found in the dataset.")
                response_message = {
                    "text": f"Sorry, we do not have sales data for {country_extracted}. Please provide another country."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            country = country_extracted[0]

            # Extract years, months, and month-year pairs from the user message
            
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            last_n_hours = extract_last_n_hours(user_message)
            last_n_weeks = extract_last_n_weeks(user_message)
            last_n_days = extract_last_n_days(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            date = extract_date(user_message)
            monthwise = extract_monthwise(user_message)
            yearwise = extract_yearwise(user_message)
            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year
                # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            logging.info(f"Processing sales data for country: {country.upper()}")
        
            try:
                if date:
                    logging.info(f"country sales specific date...{date}...")
    
                    try:
                        daily_sales_count, daily_sales_price = calculate_country_sales_for_specific_date(df, country, date)
                        
                        
                        if daily_sales_count == 0:
                            response_message = {
                                "text": f"No sales were recorded for {country.upper()} on {date}"
                            }
                        else:
                            response_message = {
                                "text": f"Sales data for {country.upper()} on {date}:\n"
                                        f"\n{daily_sales_count} sales, generating a total revenue of ${daily_sales_price:,.2f}."
                                
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                            
                    except Exception as e:
                        logging.error(f"Error calculating sales for {country.upper()} on {date}: {e}")
                        response_message = {
                            "text": f"Sorry, there was an error retrieving the sales data for {country.upper()} on {date}. Please try again later."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                        
                elif month_year_pairs:
                    logging.info(f" country sales month year pairs....{month_year_pairs}")
                    for month, year in month_year_pairs:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, year)
                            if sales_count == 0:
                                response_message = {
                                    "text": f"In {months_reverse[month].capitalize()} {year}, {country.upper()} had no sales.\n"
                                }
                            else:
                                response_message = {
                                    "text": f"In {months_reverse[month].capitalize()} {year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"        
                                }
                            dispatcher.utter_message(json_message=response_message)
                            return []
                        
                        except Exception as e:
                            logging.error(f"Error calculating sales for {month}/{year} in {country.upper()}: {e}")
                            response_message = {
                                "text": f"Error calculating sales for {months_reverse[month].capitalize()} {year} in {country.upper()}. Please try again.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                            
                elif monthwise:
                    try:
                        logging.info(f"Fetching monthwise sales data for {country.upper()}")
                        monthly_data = calculate_country_sales_by_monthwise(df, country)
                        response_message = {
                            "text": "",
                            "tables": []
                        }
                        if monthly_data.empty:
                            response_message["text"] += f"\n No monthwise sales data available for {country.upper()}.\n"
                        else:
                            response_message["text"] += f"📅 Monthly Sales Summary for {country.upper()}:\n"
                            table = []
                            for _, row in monthly_data.iterrows():
                                
                

                                table.append([
                                    str(row['MonthYear']),  # Ensure MonthYear is in string format
                                    f"${row['TotalSales']:.2f}",
                                    int(row['SalesCount']),
                                    f"${row['AvgSalesPrice']:.2f}",
                                    f"{row['SalesPercentageIncrease']:.2f}%"
                                ])
                            response_message["tables"].append({
                                "headers": ["Month-Year", "Total Sales Revenue", "Total Sales Count", "Avg Sales Revenue", "Sales Count Percentage"],
                                "data": table
                            })
                        
                        dispatcher.utter_message(json_message=response_message)
                        return []
                            
                            
                    except Exception as e:
                        logging.error(f"Error calculating monthwise sales for {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating monthwise sales for {country.upper()}. Please try again."
                        }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                # Handle yearwise calculation
                elif yearwise:
                    try:
                        logging.info(f"Fetching yearwise sales data for {country.upper()}")
                        yearly_data = calculate_country_sales_by_yearwise(df, country)
                        response_message = {
                            "text": "",
                            "tables": []
                        }
                        if yearly_data.empty:
                            response_message["text"] += f"No yearwise sales data available for {country.upper()}.\n"
                        else:
                            # Tabulate yearly data
                            response_message["text"] += f" 📅 Yearly Sales Summary for {country.upper()}:\n"
                            table = []
                            for _, row in yearly_data.iterrows():
                                table.append([
                                    str(int(row['Year'])),  # Ensure the Year is in string format
                                    int(row['SalesCount']),
                                    f"${row['TotalSales']:.2f}",
                                    f"${row['AvgSalesPrice']:.2f}",
                                    f"{row['SalesPercentageIncrease']:.2f}%"
                                ])
                
                            response_message["tables"].append({
                                "headers": ["Year", "Total Sales Count", "Total Sales Revenue", "Avg Sales Revenue", "Sales Count Percentage"],
                                "data": table
                            })
                        
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    except Exception as e:
                        logging.error(f"Error calculating yearwise sales for {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating yearwise sales for {country.upper()}. Please try again."
                        }
                        dispatcher.utter_message(json_message=response_message)
                        return []

                elif years:
                    logging.info(f" country sales years....{years}")
                    for year in years:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_year(df, country, year)
                            if sales_count>0:
                                response_message = {
                                    "text": f"In {year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                                }
                            else:
                                response_message = {
                                    "text":  f"In {year}, {country.upper()} had no sales.\n"
                                }
                            dispatcher.utter_message(json_message=response_message)
                            return []
            
                        except Exception as e:
                            logging.error(f"Error calculating sales for year {year} in {country.upper()}: {e}")
                            response_message = {
                                "text": f"Error calculating sales for {year} in {country.upper()}. Please try again.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
            
                elif last_n_hours:
                    logging.info(f"Fetching sales data for the last {last_n_hours} hours for {country.upper()}")
                    start_time, end_time, num_hours = last_n_hours
                    sales_data = calculate_sales_for_last_n_hours(df, country, start_time, end_time)
                    if sales_data['TotalSales']>0:
                        response_message = {
                            "text": f"Sales data for the last {num_hours} hours ({start_time} to {end_time}) for {country.upper()}:\n"
                                    f"\nTotal Sales: ${sales_data['TotalSales']:,.2f}, Sales Count: {sales_data['SalesCount']}\n"
                        }
                        
                    else:
                        response_message = {
                            "text": f"No sales data available for the last {num_hours} hours ({start_time} to {end_time}) for {country.upper()}.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            

                elif last_n_weeks:
                    logging.info(f"Fetching sales data for the last {last_n_weeks} weeks for {country.upper()}")
                    start_date, end_date, num_weeks = last_n_weeks
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    sales_data = calculate_sales_for_last_n_weeks(df, country, start_date, end_date)
                    if sales_data['TotalSales']>0:
                        response_message = {
                            "text": f"Sales data for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}) for {country.upper()}:\n"
                                    f"\nTotal Sales: ${sales_data['TotalSales']:,.2f}, Sales Count: {sales_data['SalesCount']}\n"         
                        }
                        
                    else:
                        response_message = {
                            "text": f"No sales data available for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}) for {country.upper()}.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            
                    
    
                elif last_n_days:
                    logging.info(f"Fetching sales data for the last {last_n_days} days for {country.upper()}")
                    start_date, end_date, num_days = last_n_days
                    sales_data = calculate_sales_for_last_n_days(df, country, start_date, end_date)
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    if sales_data['TotalSales']>0:
                        response_message = {
                            "text": f"Sales data for the last {num_days} days ({start_date_formatted} to {end_date_formatted}) for {country.upper()}:\n"
                                    f"\nTotal Sales: ${sales_data['TotalSales']:,.2f}, Sales Count: {sales_data['SalesCount']}\n"
                        }
                    else:
                        response_message = {
                            "text": f"No sales data available for the last {num_days} days ({start_date_formatted} to {end_date_formatted}) for {country.upper()}.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            

                elif months_extracted:
                    current_year = datetime.now().year
                    logging.info(f" country sales month with current year....{months_extracted}")
                    for month in months_extracted:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, current_year)
                            if sales_count == 0:  # No sales in this month
                                response_message = {
                                    "text": f"In {months_reverse[month].capitalize()} {current_year}, {country.upper()} had no sales.\n"
                                }
                            else:
                                response_message = {
                                    "text": f"In {months_reverse[month].capitalize()} {current_year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                                }
                            dispatcher.utter_message(json_message=response_message)
                            return []
                        
                        except Exception as e:
                            logging.error(f"Error calculating sales for {months_reverse[month].capitalize()} {current_year} in {country.upper}: {e}")
                            response_message = {
                                "text": f"Error calculating sales for {months_reverse[month].capitalize()} {current_year} in {country.upper}. Please try again.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
            

                elif quarterly:
                    logging.info(f" country sales quarterly....{quarterly}")
                    (start_month, end_month),year = quarterly
                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                    try:
                        # Calculate sales for the given quarter
                        sales_count, total_revenue = calculate_country_sales_by_quarter(
                            df, country, start_month, end_month, year
                        )
                
                        # Build the response message
                        if sales_count > 0:
                            response_message = {
                                "text": f"In {quarter_name} of {year}, {country} recorded "
                                        f"{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                            }
                        else:
                            response_message = {
                                "text": f"No sales data found for {quarter_name} of {year} in {country.upper()}.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
            

                            
                    except Exception as e:
                        logging.error(f"Error calculating sales for quarter {quarter_name} in {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating sales for quarter{quarter_name} in {country.upper()}. Please try again.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            

                elif half_year:
                    logging.info(f" country sales half year....{half_year}")
                    year,(start_month, end_month)= half_year
                    
                    try:
                        half_year_sales = df[
                            (df['PurchaseDate'].dt.month >= start_month) &
                            (df['PurchaseDate'].dt.month <= end_month) &
                            (df['PurchaseDate'].dt.year == year) &
                            (df['countryname'].str.lower() == country.lower())
                        ]
            
                        # Calculate sales count and total sales price
                        half_year_sales_count = half_year_sales['SellingPrice'].count()
                        half_year_sales_price = half_year_sales['SellingPrice'].sum()
            
                        # Determine whether it's the first or second half of the year
                        half_name = "First Half" if start_month == 1 else "Second Half"
                        if half_year_sales_count>0:
                            response_message = {
                                "text": f"In the {half_name} of {year}, {country.upper()} recorded {half_year_sales_count} sales, generating a total revenue of ${half_year_sales_price:,.2f}.\n"
                            }
                        else:
                            response_message = {
                                "text": f"No sales data found for {half_name} of {year} in {country.upper()}.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    except Exception as e:
                        logging.error(f"Error calculating sales for {half_name} in {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating sales for {half_name} of {year} in {country.upper()}. Please try again.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []

                elif fortnight:
                    start_date, end_date = fortnight
                    logging.info(f"Processing fortnight data for {country.upper()}: {fortnight}")
                    try:
                        # Log the current fortnight being processed
                        logging.info(f"Processing fortnight from {start_date} to {end_date} for {country.upper()}.")
                        
                        # Calculate sales for the country in the given fortnight
                        sales_count, total_revenue = calculate_country_sales_by_fortnight(df, country, start_date, end_date)
                        start_date_formatted = start_date.date()
                        end_date_formatted = end_date.date()
                        
                        if sales_count > 0 or total_revenue > 0:
                            response_message = {
                                "text":f"In the fortnight from ({start_date_formatted} to {end_date_formatted}) of {datetime.now().year}, "
                                       f"\n{country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                            }
                        else:
                            response_message = {
                                "text": f"In the fortnight from ({start_date_formatted} to {end_date_formatted}) of {datetime.now().year}, "
                                        f"\n no sales were recorded for {country.upper()}.\n"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    except Exception as e:
                        logging.error(f"Error calculating sales for fortnight from {start_date_formatted} to {end_date_formatted} in {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating sales for fortnight from {start_date_formatted} to {end_date_formatted} in {country.upper()}. Please try again.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
                elif last_n_months:
                    logging.info(f" country sales last n months....{last_n_months}")
                    start_date, end_date, num_months = last_n_months
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()

                    
                    try:
                        n_month_sales = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
        
                        sales_count = n_month_sales['SellingPrice'].count()
                        sales_price = n_month_sales['SellingPrice'].sum()
                        if sales_count:
                            response_message = {
                                "text": f"In the last {num_months} months ({start_date_formatted} to {end_date_formatted}), {country.upper()} recorded {sales_count} sales, generating a total revenue of ${sales_price:,.2f}.\n"
                            }
                        else:
                            response_message = {
                                "text": f"No sales data found for last {num_months} months ({start_date_formatted} to {end_date_formatted})"
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                                    
                    except Exception as e:
                        logging.error(f"Error calculating sales for last {num_months} months ({start_date_formatted} to {end_date_formatted}) in {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating sales for last {num_months} months ({start_date_formatted} to {end_date_formatted}) in {country.upper()}. Please try again.\n"
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []

                elif today:
                    today_date = datetime.now().date()
                    logging.info(f" country sales today....{today_date}")
                    try:
                        sales_count, total_revenue = calculate_country_sales_for_today(df, country, today)
                        if sales_count != 0 and total_revenue != 0.0:
                            response_message = {
                                "text": f"Today's {today_date} sales data for {country.upper()}: \n"
                                        f"\n{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                            }
                            
                        else:
                            response_message = {
                                "text": f"No sales data found for {country.upper()} on {today_date}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []

                    except Exception as e:
                        logging.error(f"Error calculating today's sales for {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating today's sales data for {country.upper()}. Please try again later."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []

                elif last_day:
                    lastday = (datetime.now() - timedelta(days=1)).date()
                    logging.info(f" country sales last day....{lastday}")
                    try:
                        sales_count, total_revenue = calculate_country_sales_for_last_day(df, country, last_day)
                        if sales_count != 0 and total_revenue != 0.0:
                            response_message = {
                                "text": f"last day's {lastday} sales data for {country.upper()}:\n "
                                        f"\n{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                            }
                        else:
                            response_message = {
                                "text": f"No sales data found for {country.upper()} on {lastday}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    except Exception as e:
                        logging.error(f"Error calculating last day's sales for {country.upper()}: {e}")
                        response_message = {
                            "text": f"Error calculating last day's sales data for {country.upper()}. Please try again later."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []

                else:
                    # If no specific month or year, return total sales for the country
                    logging.info("total country sales....")
                    try:
                        # Attempt to calculate total sales for the country
                        sales_count, total_revenue = calculate_country_sales(df, country)
                        start_date = df['PurchaseDate'].min().date()
                        end_date = df['PurchaseDate'].max().date()
                        response_message = {
                            "text": f" (from {start_date} to {end_date}) In {country.upper()}, there have been a total of {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                        }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    except Exception as e:
                        logging.error(f"Error calculating total sales for {country.upper()}: {e}")
                        response_message = {
                            "text": f"Sorry, there was an error retrieving the total sales data for {country.upper()}. Please try again later."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            except Exception as e:
                response_message = {
                    "text": "An error occurred while calculating sales data. Please try again."
                }
                logging.error(f"Sales calculation error for country {country.upper()}: {e}")
            dispatcher.utter_message(json_message=response_message)
            return []

        except Exception as e:
            response_message = {
                "text": "An error occurred while processing your request. Please try again later."
            }

            logging.error(f"Error fetching or processing sales data: {e}")
        dispatcher.utter_message(json_message=response_message)
        return []
            

##########################################################################################################################PLANRELATEDSALES#################

class ActionPlanNameByCountry(Action):
    def name(self) -> str:
        return "action_planname_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionPlanNameByCountry...")
        global df

        try:
        #     

            # Get user message and extract entities
            user_message = tracker.latest_message.get('text')
            if not user_message:
                response_message = {
                    "text": "Sorry, I couldn't understand your message. Please try again."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
                
            logging.info(f"Received user message: {user_message}")
            country_extracted = extract_country_from_text(user_message)
            planname = next(tracker.get_latest_entity_values('planname'), None)
            
            if not country_extracted:
                response_message = {
                    "text": "❌ No valid country found in your query. Please specify a country."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
                
            
            country = country_extracted[0]
            
            
            

            # Extract plans for the specified country
            country_plans = df[df['countryname'].str.lower() == country.lower()]['PlanName'].unique()
            if len(country_plans) == 0:
                logging.info(f"No plans found for country: {country.upper()}")
                response_message = {
                    "text": f"No plans available for {country.upper()}."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
                
            # Extract years, months, and month-year pairs from user message
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            last_n_hours = extract_last_n_hours(user_message)
            last_n_weeks = extract_last_n_weeks(user_message)
            last_n_days = extract_last_n_days(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            date = extract_date(user_message)
            #specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            
            logging.info(f"Processing sales data for country: {country.upper()} and plans: {planname}")
            
            response_message = {
                "text": "",
                "tables": []
            }
            response_data = []
            country = country.upper()

            # Generate response based on provided filters
            if month_year_pairs:
                logging.info(f"Processing month year {month_year_pairs} data for {country} .")
                for month, year in month_year_pairs:
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, year)
                        if sales_count > 0 and total_revenue > 0:
                           response_data.append((plan, sales_count, total_revenue))
                    if response_data:
                        # Format the sales data into a table and append to the response message
                        response_data.sort(key=lambda x: x[1], reverse=True)

                        response_message["text"] += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {year} ({country} Plans):\n\n"
                        table_data = []
                        for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                            table_data.append([ plan, str(sales), f"${revenue:,.2f}"])

                        response_message["tables"].append({
                            "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })
                        response_data.clear()
                    else:
                        response_message["text"] += f"No sales data found for {months_reverse.get(month, month).capitalize()} {year} ({country} Plans).\n\n"
                    dispatcher.utter_message(json_message=response_message)
                    return []

                        


            elif years:
                logging.info(f"Processing yearly {years} data for {country}.")
                for year in years:
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_year(df, plan, year)
                        if sales_count > 0 or total_revenue > 0:
                            response_data.append((plan, sales_count, total_revenue))
                    if response_data:
                        # Format the sales data into a table and append to the response message
                        response_data.sort(key=lambda x: x[1], reverse=True)

                        response_message["text"] += f"📅 Sales Overview for {year} ({country} Plans):\n\n"
                        table_data = []
                        for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                            # Ensure that numeric values are converted to strings
                            table_data.append([ plan, str(sales), f"${revenue:,.2f}"])

                        response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })

                        # Clear response_data for the next iteration
                        response_data.clear()
                    else:
                        response_message["text"] += f"No sales data found for {year} ({country} Plans).\n\n"
                    dispatcher.utter_message(json_message=response_message)
                    return []

                        

            elif months_extracted:
                logging.info(f"Processing month with current year{months_extracted} data for {country}.")
                current_year = datetime.now().year
                for month in months_extracted:
                    for plan in country_plans:
                        sales_count, total_revenue = calculate_planname_sales_by_month_year(df, plan, month, current_year)
                        if sales_count > 0 or total_revenue > 0:
                            response_data.append((plan, sales_count, total_revenue))
                    if response_data:
                        # Format the sales data into a table and append to the response message
                        response_data.sort(key=lambda x: x[1], reverse=True)

                        response_message["text"] += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"

                        # Format the data into a table and add it to the tables section
                        table_data = []
                        for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                            # Convert sales and revenue to appropriate formats for JSON
                            table_data.append([ plan, str(sales), f"${revenue:,.2f}"])

                        response_message["tables"].append({
                            "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })

                        # Clear response_data for the next iteration
                        response_data.clear()
                    else:
                        response_message["text"] += f"No sales data found for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"
                    dispatcher.utter_message(json_message=response_message)
                    return []

            elif last_n_hours:
                logging.info(f"Fetching sales data for the last {last_n_hours} hours for {country.upper()}")
                start_time, end_time, num_hours = last_n_hours
                for plan in country_plans:
                    # Get sales count and total revenue for the plan using the function
                    sales_count, total_revenue = calculate_planname_for_last_n_hours(df, plan, start_time, end_time)
            
                    # Check if there is any sales data to add
                    if sales_count > 0 :
                        response_data.append((plan, sales_count, total_revenue))
            
                if response_data:
                    # Sort the sales data by sales count in descending order
                    response_data.sort(key=lambda x: x[1], reverse=True)
                    response_message["text"] += f"📊 Sales Data for the last {num_hours} hours ({start_time} to {end_time}) for ({country} Plans) :\n"
                    table_data = []
                    for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                        table_data.append([plan, str(sales), f"${revenue:,.2f}"])
            
                    # Add the table to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Sales ($)"],
                        "data": table_data
                    })
            
                    response_data.clear()

                else:
                    response_message["text"] += f"No sales data available for the last {num_hours} hours ({start_time} to {end_time}) for {country} plans.\n"

            # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []

            elif last_n_weeks:
                logging.info(f"Fetching sales data for the last {last_n_weeks} weeks for {country.upper()}")
                start_date, end_date, num_weeks = last_n_weeks
                for plan in country_plans:
                    sales_count, total_revenue = calculate_plansales_for_last_n_weeks(df, plan, start_date, end_date)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))

                if response_data:
            # Sort sales data by sales count in descending order
                    response_data.sort(key=lambda x: x[1], reverse=True)
        
                    # Add introductory text
                    response_message["text"] += f"📅 Sales Overview for the Last {num_weeks} Weeks ({start_date.date()} to {end_date.date()}) for {country} Plans"
                       
        
                    # Format the data into a table
                    table_data = []
                    for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                        table_data.append([ plan, str(sales), f"${revenue:,.2f}"])
        
                    # Add table to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                    response_data.clear()

                else:
                    response_message["text"] += f"No sales data available for the last {num_weeks} weeks for {country} plans.\n"
                dispatcher.utter_message(json_message=response_message)
                return []

            elif last_n_days:
                logging.info(f"Fetching sales data for the last {last_n_days} days for {country.upper()}")
                start_date, end_date, num_days = last_n_days
                for plan in country_plans:
                    sales_count, total_revenue = calculate_plansales_for_last_n_days(df, plan, start_date, end_date)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
        
                if response_data:
                    # Sort sales data by sales count in descending order
                    response_data.sort(key=lambda x: x[1], reverse=True)
        
                    # Add introductory text
                    response_message["text"] +=  f"📅 Sales Overview for the Last {num_days} Days ({start_date.date()} to {end_date.date()}) for {country} Plans."
                          # Format the data into a table
                    table_data = []
                    for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                        table_data.append([ plan, str(sales), f"${revenue:,.2f}"])
        
                    # Add table to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                    response_data.clear()
                else:
                    response_message["text"] += f"No sales data available for the last {num_days} days for {country} plans.\n"
                dispatcher.utter_message(json_message=response_message)
                return []


            elif quarterly:
                logging.info(f"Processing quarterly {quarterly} data for {country}.")
                (start_month, end_month),year = quarterly
                quarter_name_map = {
                    (1, 3): "First Quarter",
                    (4, 6): "Second Quarter",
                    (7, 9): "Third Quarter",
                    (10, 12): "Fourth Quarter"
                }
                quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_by_quarter(df, plan, start_month, end_month, year)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📅 Sales Overview for {quarter_name} {year} ({country}):\n\n"

        # Format the data into a table and add it to the tables section
                    table_data = []
                    for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                        # Add plan data into table with index, sales, and formatted revenue
                        table_data.append([ plan, str(sales), f"${revenue:,.2f}"])
            
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
            
                    response_data.clear()  # Clear data for the next iteration
            
                else:
                    response_message["text"] += f"No sales data found for {quarter_name} of {year} in {country} plans.\n"
                dispatcher.utter_message(json_message=response_message)
                return []



            elif date:
                logging.info(f"Processing data for {country} on {date}.")
                for plan in country_plans:
                    daily_sales_count ,daily_sales_price =  calculate_plannane_sales_for_specific_date(df, planname, date)
                    if daily_sales_count > 0 and daily_sales_price > 0:
                        response_data.append((plan, daily_sales_count, daily_sales_price))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📅 Sales Overview for {date} ({country} Plans):\n\n"

        # Format the data into a table and add it to the tables section
                    table_data = []
                    for idx, (plan, sales, revenue) in enumerate(response_data, 1):
                        # Add plan data into table with index, sales, and formatted revenue
                        table_data.append([ plan, str(sales), f"${revenue:,.2f}"])
            
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
            
                    response_data.clear()  # Clear data for the next iteration
            
                else:
                    response_message["text"] += f"No sales data found on {date} ({country} Plans):\n\n"
                dispatcher.utter_message(json_message=response_message)
                return []


            elif half_year:
                logging.info(f"Processing half-year {half_year} data for {country}.")
                year,(start_month, end_month) = half_year
               
                try:
                    # Filter the DataFrame for the half-year and country
                    half_year_sales = df[
                        (df['PlanName'] == plan)&
                        (df['PurchaseDate'].dt.year == year) &
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) 
                    ]
                    
                    # Calculate sales count and total sales revenue
                    for plan in country_plans:
                        plan_sales = half_year_sales[half_year_sales['PlanName'] == plan]
            
                        # Calculate sales count and total revenue for the plan
                        sales_count = plan_sales['SellingPrice'].count()
                        total_revenue = plan_sales['SellingPrice'].sum()
            
                        if sales_count > 0 or total_revenue > 0:
                            response_data.append((plan, sales_count, total_revenue))
                    half_name = "First Half" if start_month == 1 else "Second Half"
            
                    if response_data:
                        # Sort by sales count in descending order
                        response_data.sort(key=lambda x: x[1], reverse=True)
            
                        # Format the response message
                        response_message["text"] = (
                            f"📅 Sales Overview for {half_name} of {year} ({country} Plans):\n\n"
                        )
            
                        # Format data into table format
                        table_data = []
                        for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                            table_data.append([plan, str(sales_count), f"${total_revenue:,.2f}"])
            
                        # Add the table to the response message
                        response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })
            
                        response_data.clear()  # Clear data after processing
            
                    else:
                        response_message["text"] += f"No sales data found for the {half_name} of {year} ({country} Plans).\n\n"
                    dispatcher.utter_message(json_message=response_message)
                    return []
            
                except Exception as e:
                    logging.error(f"Error calculating sales for half-year {half_name} in {country.upper()}: {e}")
                    response_message["text"] += (
                        f"Error calculating sales for the period in {country.upper()}. "
                        "Please try again later.\n"
                    )
                    dispatcher.utter_message(json_message=response_message)
                    return []
                    
            elif fortnight:
                logging.info(f"Processing fortnight {fortnight} data for {country}.")
                start_date, end_date = fortnight
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                if country_plans is None:
                    country_plans = [] 
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_for_fortnight(df, plan, start_date, end_date)
                    
                    if sales_count > 0 :
                        response_data.append((plan, sales_count, total_revenue))
                    
    
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📅 Sales Overview for Fortnight ({start_date_formatted} to {end_date_formatted}) in {country} plans:\n\n"
                    table_data = []
                    for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                        table_data.append([
                            
                            plan,
                            str(sales_count),
                            f"${total_revenue:,.2f}"
                        ])
        
                    # Add the table to the response message
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
        
                    response_data.clear()
                else:
                    response_message["text"] += f"No sales data found for the period {start_date_formatted} to {end_date_formatted} in {country} plans."
                dispatcher.utter_message(json_message=response_message)
                return []



            elif last_n_months:
                logging.info(f"Processing last N months {last_n_months} data for {country}.")
                start_date, end_date, num_months = last_n_months
        
               
                for plan in country_plans:
                    filtered_df = df[
                        (df['PlanName'] == plan) &
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]
                    sales_count = filtered_df['SellingPrice'].count()
                    total_revenue = filtered_df['SellingPrice'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)
                    response_message["text"] += f"📅 Sales Overview for Last {num_months} Months ({start_date_formatted} to {end_date_formatted}){country} Plans."
                    table_data = []
                    for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                        table_data.append([
                            
                            plan, 
                            str(sales_count), 
                            f"${total_revenue:,.2f}"
                        ])
        
                    # Add the table to the response_message
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
        
                    response_data.clear()
                else:
                    response_message["text"] += f"No sales data found for Last {num_months} Months ({start_date_formatted} to {end_date_formatted}){country} Plans :\n\n"
                dispatcher.utter_message(json_message=response_message)
                return []
                        


            elif today:
                today_date = datetime.now().date()
                logging.info(f"Processing today's {today_date} data for {country}.")
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_by_today(df, plan, today)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📅 Sales Overview for Today ({today_date}) ({country} Plans):\n\n"
                    table_data = []
                    for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                        table_data.append([
                            plan, 
                            str(sales_count), 
                            f"${total_revenue:,.2f}"
                        ])
        
                    # Add the table to the response_message
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
        
                    response_data.clear()
                else:
                    response_message["text"] += f"No sales data found for Today ({today_date}) ({country} Plans):\n\n"
                dispatcher.utter_message(json_message=response_message)
                return []
                            


            elif last_day:
                last_date = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Processing last day's {last_date} data for {country}.")
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_by_last_day(df, plan, last_day)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📅 Sales Overview for Last Day ({last_date}) ({country} Plans):\n\n"
                    table_data = []
                    for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                        table_data.append([
                            plan, 
                            str(sales_count), 
                            f"${total_revenue:,.2f}"
                        ])
        
                    # Add the table to the response_message
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
        
                    response_data.clear() # Clear data for the next iteration
                else:
                    response_message["text"] += f"No sales data found for Last Day ({last_date}) ({country} Plans):\n\n"
                dispatcher.utter_message(json_message=response_message)
                return []
            else:
                logging.info("total sales plans")
                for plan in country_plans:
                    sales_count, total_revenue = calculate_total_planname_sales(df, plan)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message["text"] += f"📊 Total Sales Overview for {country} Plans (from {start_date} to {end_date}):\n\n"
                    table_data = []
                    for idx, (plan, sales_count, total_revenue) in enumerate(response_data, 1):
                        table_data.append([
                            plan,
                            str(sales_count),
                            f"${total_revenue:,.2f}"
                        ])
        
                    # Add the table to the response message
                    response_message["tables"].append({
                        "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
        
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message["text"] += f"No sales data found for ({country} Plans):\n\n"
                dispatcher.utter_message(json_message=response_message)
                return []

            # Check if no data was found
            if not response_message.strip():
                dispatcher.utter_message(text=f"No sales data found for the specified criteria in {country}.")
                logging.info("No sales data found for specified criteria.")
                dispatcher.utter_message(json_message=response_message)
                return []

        except KeyError as e:
            logging.error(f"KeyError: {e}")
            response_message["text"] = (
                "An error occurred while processing the sales data. "
                "Some required fields might be missing in the dataset. Please try again."
            )
            dispatcher.utter_message(json_message=response_message)
            return []
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            response_message["text"] = (
                "There was an issue with the input data format. "
                "Ensure that all dates are correctly specified and valid."
            )
            dispatcher.utter_message(json_message=response_message)
            return []
    
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            response_message["text"] = (
                "An unexpected error occurred while processing the sales data. "
                "Please try again later."
            )
            dispatcher.utter_message(json_message=response_message)
            return []


#################################################################################################################active plans and country name##############


class ActionGetActiveAndInactivePlans(Action):
    def name(self) -> str:
        return "action_get_active_inactive_plans"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActivePlans...")
        global df

        try:
        

            user_message = tracker.latest_message.get('text')
            
            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                logging.error(f"Date parsing error: {e}")
                response_message = {
                    "text": "Error parsing purchase dates. Please check the date format."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Get the current year dynamically
            try:
                current_date = datetime.now()
                six_months_ago = current_date - timedelta(days=6 * 30) 
            except Exception as e:
                logging.error(f"Error fetching current year: {e}")
                response_message = {
                    "text": "Could not retrieve the current year."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
               

            # Identify active and inactive plans
            active_plans = df[
                (df['PurchaseDate'] >= six_months_ago) & (df['PurchaseDate'] <= current_date)
            ]['PlanName'].unique()
            past_plans = df[
                (df['PurchaseDate'] < six_months_ago)
            ]['PlanName'].unique()
            active_plans = list(active_plans)
            inactive_plans = list(set(past_plans) - set(active_plans))
            plans_count = len(active_plans)
            inactive_plans_count = len(inactive_plans)
            active_plans.sort()
            inactive_plans.sort()

            # Generate response
            if plans_count == 0 and inactive_plans_count == 0:
                logging.info("No plans found for the specified criteria.")
                response_message = {
                    "text": "No active or inactive plans found for the specified criteria."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
                
            response_message = {
                "text": ""
            }

            response_message["text"] += f"Total Active Plans (current 6 months): {plans_count}\n\n"
            response_message["text"] += "HERE ARE THE ACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in active_plans) + "\n\n"
            response_message["text"] += f"Total Inactive Plans (current 6 months): {inactive_plans_count}\n\n"
            response_message["text"] += "HERE ARE THE INACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in inactive_plans)

            dispatcher.utter_message(json_message=response_message)
            return []
        
        except Exception as e:
            logging.error(f"Unexpected error in ActionGetActiveAndInactivePlans: {e}")
            response_message = {
                    "text": "An unexpected error occurred. Please try again later."
                }
            dispatcher.utter_message(json_message=response_message)
            return []
        


class ActionGetActiveAndInactiveCountries(Action):
    def name(self) -> str:
        return "action_get_active_inactive_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: dict) -> list:
        logging.info("Running ActionActiveCountry...")
        global df

        try:
        #     
            user_message = tracker.latest_message.get('text')
            

            # Ensure 'PurchaseDate' can be converted to datetime
            try:
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'])
            except Exception as e:
                logging.error(f"Date parsing error: {e}")
                response_message = {
                    "text": "Error parsing purchase dates. Please check the date format."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                logging.error(f"Error fetching current year: {e}")
                response_message = {
                    "text": "Could not retrieve the current year."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            end_date = datetime.now()
            start_date = end_date - timedelta(days=6 * 30)

            # Identify active and inactive countries
            active_countries_last_6_months = df[(df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]['countryname'].unique()
            past_countries = df[df['PurchaseDate'] < start_date]['countryname'].unique()
            active_countries = list(active_countries_last_6_months)
            inactive_countries = list(set(past_countries) - set(active_countries))
            active_countries.sort()
            inactive_countries.sort()

            countries_count = len(active_countries)
            inactive_countries_count = len(inactive_countries)
            # Generate response
            if countries_count == 0 and inactive_countries_count == 0:
                logging.info("No countries found for the specified criteria.")
                response_message = {
                    "text": "No active or inactive countries found for the specified criteria."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            response_message = {
                "text": ""
            }

            response_message["text"] += f"Total Active Countries (current 6 Months): {countries_count}\n\n"
            response_message["text"] += "HERE ARE THE ACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in active_countries) + "\n\n"
            response_message["text"] += f"Total Inactive Countries (current 6 Months): {inactive_countries_count}\n\n"
            response_message["text"] += "HERE ARE THE INACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in inactive_countries)

            dispatcher.utter_message(json_message=response_message)
            return []
        
        except Exception as e:
            logging.error(f"Unexpected error in ActionGetActiveAndInactiveCountries: {e}")
            response_message = {
                "text": "An unexpected error occurred. Please try again later."
            }
            dispatcher.utter_message(json_message=response_message)
            return []



#############################################################################################################top and lowest sales plan############


class ActionTopPlansSales(Action):

    def name(self) -> str:
        return "action_top_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionTopPlansSales...")
        
        global df
        user_message = tracker.latest_message.get('text')

        # Extract year and month from user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        quarterly = extract_quarters_from_text(user_message)
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_hours = extract_last_n_hours(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_days = extract_last_n_days(user_message)
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)

        logging.info(f"Processing top plans sales data based on user request: {user_message}")
        n = self.extract_n_highest_from_message(user_message)
        if n is None:
            n = 1

        response_message = {
            "text": "",
            "tables": []
        }
        try:
            # Determine the response based on user input
            if month_year_pairs:
                logging.info(f"top plans for month year :{month_year_pairs}")

                for month, year in month_year_pairs:
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(n, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message["text"] += f"No sales data available for the top {n}  plans in the {months_reverse[month]} {year} .\n"
                    else:
                        response_data = []
                        response_message["text"] += f"📈 Top {n} Plans for {months_reverse[month]} {year} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            response_data.append([ row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                        # Add table data to the response message
                        response_message["tables"].append({
                            "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": response_data
                        })

                    dispatcher.utter_message(json_message=response_message)
                    return []
                        

            elif years:
                logging.info(f"top plans for year {years}")
                for year in years:
                    top_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(n, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message["text"] += f"No sales data available for the top {n} plans in the {years}.\n"
                    
                    else:
                        table_data = []
                        response_message["text"] += f"📈 Top {n} Plans for {years} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                # Append only the serial number, Plan Name, Sales Count, and Total Revenue
                            table_data.append([ row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                        
                        # Now the table has only the serial number and the other columns
                        response_message["tables"].append({
                            "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })
                    dispatcher.utter_message(json_message=response_message)
                    return []



            elif months_extracted:
                logging.info(f"top plans for month with current year :{months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(n, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message["text"] += f"No sales data available for the top {n} plans in the {months_reverse[month]} {current_year}.\n"
                    else:
                        response_data = []
                        response_message["text"] += f"📈 Top {n} Plans for {months_reverse[month]} {current_year} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                        response_message["tables"].append({
                            "headers": [ "Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": response_data
                        })
                    dispatcher.utter_message(json_message=response_message)
                    return []
            elif last_n_hours:
                logging.info(f"Top plans for the last {last_n_hours} hours")
                start_time, end_time, num_hours = last_n_hours
                top_plans = df[(df['PurchaseDate'] >= start_time) &
                        (df['PurchaseDate'] <= end_time)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the Top {n} plans in last {num_hours} hours.\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the last {num_hours} hours ({start_time} to {end_time}):\n"
                    
                    # Add plan details to response data
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    
                   
            elif last_n_days:
                logging.info(f"Top plans for the last {last_n_days} days")
                start_date, end_date, num_days = last_n_days
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                top_plans = df[(df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the Top {n} plans in last {num_days} days ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the last {num_days} days ({start_date_formatted} to {end_date_formatted})\n"
                    
                    # Add plan details to response data
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []

            elif last_n_weeks:
                logging.info(f"Top plans for the last {last_n_weeks} weeks")
                start_date, end_date, num_weeks = last_n_weeks
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                top_plans = df[(df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the Top {n} Plans in last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}):\n"
                    
                    # Add plan details to response data
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
            elif quarterly:
                logging.info(f"top plans for quarterly: {quarterly}")
                (start_month, end_month),year = quarterly
                
                
                # Map quarters to names
                quarter_name_map = {
                    (1, 3): "First Quarter",
                    (4, 6): "Second Quarter",
                    (7, 9): "Third Quarter",
                    (10, 12): "Fourth Quarter"
                }
                
                quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                top_plans = df[(df['PurchaseDate'].dt.month >= start_month) & (df['PurchaseDate'].dt.month <= end_month)&(df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans in the {quarter_name} {year}.\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for {quarter_name} {year}:\n"
                    
                    # Add plan details to response data
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    

            elif half_year:
                logging.info(f"top plans for half year: {half_year}")
                year,(start_month, end_month) = half_year
                current_year = pd.to_datetime('today').year
                top_plans = df[
                    (df['PurchaseDate'].dt.month == start_month) &                                                       (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == year)
                    ].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(n, 'total_sales').reset_index()
                half_name = "First Half" if start_month == 1 else "Second Half"
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans in the {half_name} of {year} .\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for {half_name} of {year}:\n"
                    
                    # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    

            elif fortnight:
                logging.info(f"top plans for fortnight :{fortnight}")
                start_date, end_date = fortnight
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                top_plans = df[(df['PurchaseDate']>=start_date) & (df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for last fortnight ({start_date_formatted} to {end_date_formatted}):\n"
                    
                    # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"top plans for last_day :{lastday}")
                top_plans = df[(df['PurchaseDate']==lastday)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans in the last day {lastday}.\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the last day ({lastday}):\n"
                    
                    # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            elif today:
                today_date = datetime.now().date()

                logging.info(f"top plans for last_day :{today_date}")
                top_plans = df[(df['PurchaseDate']==today_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"]+= f"No sales data available for the top {n} plans on {today_date}.\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for today ({today_date}):\n"
        
        # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                                
                    
            elif date:
                logging.info(f"Processing top plans for the specific date: {date}")
                date = pd.to_datetime(date, errors='coerce').date()
                if pd.isna(date):
                    return None, "Error: The provided date is invalid."
                df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
         
                
                # Filter data for the specific date and group by 'PlanName'
                top_plans = (
                    df[df['PurchaseDate'].dt.date == date]
                    .groupby('PlanName')
                    .agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    )
                    .nlargest(n, 'total_sales')
                    .reset_index()
                )
            
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans on {date}.\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the specific date {date}:\n"
                    
                    # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    

            elif last_n_months:
                logging.info(f"top plans for last n months :{last_n_months}")
                start_date, end_date, num_months = last_n_months
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                top_plans = df[(df['PurchaseDate']>=start_date)&(df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans in last {last_n_months} months ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_data = []
                    response_message["text"] += f"📈 Top {n} Plans for the last {num_months} months ({start_date_formatted} to {end_date_formatted}):\n"
                    
                    # Collect data for the response
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    

            
            else:
                logging.info("top plans overall")
                top_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(n, 'total_sales').reset_index()
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if top_plans.empty:
                    response_message["text"] += f"No sales data available for the top {n} plans.\n"
                else:
                    response_message["text"] += f"📈 Top {n} Sales Plans Overall (from {start_date} to {end_date}):\n"
        
                    # Prepare the response data
                    response_data = []
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    
                    # Add table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
                
                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
                    


        except Exception as e:
            logging.error(f"Error while processing top sales plans: {e}")
            response_message = {
                "text": "An error occurred while processing your request. Please try again."
            }
            dispatcher.utter_message(json_message = response_message)
            return []
    
    def extract_n_highest_from_message(self, message: str) -> int:
        match = re.search(r'(\d+)\s*(?:highest|top|most|greatest|best|maximum|strongest|leading|high|record|peak|elite|upper)?\s*(?:plans|plan)?', message.lower())
        if match:
            return int(match.group(1))  # Extracted number
        return None



class ActionLowestPlansSales(Action):

    def name(self) -> str:
        return "action_lowest_plans_sales"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionLowestPlansSales...")

        global df
        user_message = tracker.latest_message.get('text')
        

        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        quarterly = extract_quarters_from_text(user_message)
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_hours = extract_last_n_hours(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_days = extract_last_n_days(user_message)
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
        n = self.extract_n_from_message(user_message)
        if n is None:
            n = 1

        response_message = {
            "text": "",
            "tables": []
        }
        try:
            if month_year_pairs:
                logging.info(f"top plans for month year :{month_year_pairs}")

                for month, year in month_year_pairs:
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(n, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message["text"] += f"No sales data available for the lowest {n} plans in {months_reverse[month]} {year} .\n"
                    else:
                        response_message["text"] += f"📉 lowest {n} Plans for {months_reverse[month]} {year} :\n"
                        response_data = []
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                            # Add table data to the response message
                        response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": response_data
                        })

                    # Send the response message in JSON format
                    dispatcher.utter_message(json_message=response_message)
                    return []

            elif years:
                logging.info(f"top plans for year :{years}")
                for year in years:
                    lowest_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(n, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message["text"] += f"No sales data available for the lowest {n} plans in {years} .\n"
                    else:
                        response_message["text"] += f"📉 lowest {n} Plans for {years} :\n"
                        response_data = []
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                        # Add table data to the response message
                        response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": response_data
                        })
            
                    # Send the response message in JSON format
                    dispatcher.utter_message(json_message=response_message)
                    return []
            elif months_extracted:
                logging.info(f"top plans for month with current year :{months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(n, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message["text"] += f"No sales data available for the lowest {n} plans in {months_reverse[month]}.\n"
                    else:
                        response_message["text"] += f"📉 lowest {n} Plans for {months_reverse[month]} :\n"
                        response_data = []
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                        # Add table data to the response message
                        response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": response_data
                        })
                    dispatcher.utter_message(json_message=response_message)
                    return []
                        
            elif fortnight:
                logging.info(f"top lowest plans for fortnight :{fortnight}")
                start_date, end_date = fortnight
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                lowest_plans = df[(df['PurchaseDate']>=start_date) & (df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_message["text"] += f"📉 Lowest {n} Plans for last fortnight ({start_date_formatted} to {end_date_formatted}) :\n"
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            elif quarterly:
                logging.info(f"top lowest plans for quarterly: {quarterly}")
                (start_month, end_month),year = quarterly
                
                # Map quarters to names
                quarter_name_map = {
                    (1, 3): "First Quarter",
                    (4, 6): "Second Quarter",
                    (7, 9): "Third Quarter",
                    (10, 12): "Fourth Quarter"
                }
                
                quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                lowest_plans = df[(df['PurchaseDate'].dt.month>=start_month) & (df['PurchaseDate'].dt.month<=end_month)&(df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans in the {quarter_name} {year}.\n"
                else:
                    response_message["text"] += f"📉 Lowest {n} Plans for {quarter_name} {year}:\n"
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"top lowest plans for last_day :{lastday}")
                lowest_plans = df[(df['PurchaseDate']==lastday)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans in the last day {lastday}.\n"
                else:
                    response_message["text"] += f"📉 Lowest {n} Plans for last day ({lastday}).\n"
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
    
            elif today:
                today_date = datetime.now().date()

                logging.info(f"top lowest plans for last_day :{today_date}")
                lowest_plans = df[(df['PurchaseDate']==today_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans on {today_date}.\n"
                else:
                    response_message["text"] += f"📉 Lowest {n} Plans for today ({today_date}).\n"
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
            elif date:
                logging.info(f"Processing top lowest plans for the specific date: {date}")
        
                
                # Filter data for the specific date and group by 'PlanName'
                lowest_plans = (
                    df[df['PurchaseDate'].dt.date == date]
                    .groupby('PlanName')
                    .agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    )
                    .nsmallest(n, 'total_sales')
                    .reset_index()
                )
            
                if lowest_plans.empty:
                    response_message["text"] = f"No sales data available for the lowest {n} plans on {date}."
                else:
                    response_message["text"] = f"📉 Lowest {n} Plans for the specific date {date}:\n"
                    
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []

            elif last_n_months:
                logging.info(f"top lowest plans for last n months :{last_n_months}")
                start_date, end_date,num_months = last_n_months
                
                lowest_plans = df[(df['PurchaseDate']>=start_date)&(df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans in last {num_months} months ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_message["text"] += f"📉 Top Lowest {n} Plans for the last {num_months} months ({start_date_formatted} to {end_date_formatted}):\n"
        
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []

            # For last_n_hours
            elif last_n_hours:
                logging.info(f"Top lowest plans for the last {last_n_hours} hours")
                start_time, end_time, num_hours = last_n_hours
                lowest_plans = df[(df['PurchaseDate'] >= start_time) &
                                  (df['PurchaseDate'] <= end_time)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] = f"No sales data available for the lowest {n} plans in the last {num_hours} hours ({start_time} to {end_time})."
                else:
                    response_message["text"] = f"📉 Top {n} Lowest Plans for the last {num_hours} hours ({start_time} to {end_time}):\n"
                    
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
            
            # For last_n_days
            elif last_n_days:
                logging.info(f"Top lowest plans for the last {last_n_days} days")
                start_date, end_date, num_days = last_n_days
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                lowest_plans = df[(df['PurchaseDate'] >= start_date) &
                                  (df['PurchaseDate'] <= end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] = f"No sales data available for the lowest {n} plans in the last {num_days} days ({start_date} to {end_date})."
                else:
                    response_message["text"] = f"📉 Top {n} Lowest Plans for the last {num_days} days ({start_date_formatted} to {end_date_formatted}):\n"
                    
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            # For last_n_weeks
            elif last_n_weeks:
                logging.info(f"Top lowest plans for the last {last_n_weeks} weeks")
                start_date, end_date, num_weeks = last_n_weeks
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                lowest_plans = df[(df['PurchaseDate'] >= start_date) &
                                  (df['PurchaseDate'] <= end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message["text"] += f"No sales data available for the lowest {n} plans in last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_message["text"] = f"📉 Top {n} Lowest Plans for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}):\n"
        
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            elif half_year:
                logging.info(f"top lowest plans for half year: {half_year}")
                year,(start_month, end_month) = half_year
                current_year = pd.to_datetime('today').year
                lowest_plans = df[
                    (df['PurchaseDate'].dt.month == start_month) &                                                      (df['PurchaseDate'].dt.month <= end_month) &
                    (df['PurchaseDate'].dt.year == year)
                    ].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(n, 'total_sales').reset_index()
                half_name = "First Half" if start_month == 1 else "Second Half"
                if lowest_plans.empty:
                    response_message["text"] = f"No sales data available for the lowest {n} plans in the {half_name} of {year}."
                else:
                    response_message["text"] = f"📉 Lowest {n} Plans for the {half_name} of {year}:\n"
                    
                    # Prepare the table data
                    response_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
            
                    # Add the table data to the response message
                    response_message["tables"].append({
                        "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
            
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
            else:
                lowest_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(n, 'total_sales').reset_index()
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if lowest_plans.empty:
                    response_message["text"] += f"No data available for the lowest {n} plans overall."
                else:
                    response_message["text"] += f"📉 lowest {n} Plans for overall (from {start_date} to {end_date}).\n"
                    table_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([row['PlanName'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                    response_message["tables"].append({
                            "headers": ["Plan Name", "Sales Count", "Total Revenue ($)"],
                            "data": table_data
                        })
                    dispatcher.utter_message(json_message=response_message)
                    return []
                    
                    

        except Exception as e:
            logging.error(f"Error while processing lowest sales plans: {e}")
            response_message = {
                "text": "An error occurred while processing your request. Please try again."
            }
            dispatcher.utter_message(json_message = response_message)
            return []

        

    def extract_n_from_message(self, message: str) -> int:
        match = re.search(r'(\d+)\s*(?:lowest|low|fewest|bottom|least|minimal|smallest|weakest|poorest)?\s*(?:plans|plan)?', message.lower())
        if match:
            return int(match.group(1))  # Extracted number
        return None


        
########################################################################################################### top highest and lowest country sales #########################
class ActionTopHighestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_highest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("top highest sales by country")
        global df

        
        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        quarterly = extract_quarters_from_text(user_message)
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_hours = extract_last_n_hours(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_days = extract_last_n_days(user_message)
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        
        n = self.extract_n_highest_from_message(user_message)
        if n is None:
            n = 1

        response_message = {
            "text": "",
            "tables": []
        }

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            logging.info(f"top highest sales by country for month year pairs:{month_year_pairs}")
            for month, year in month_year_pairs:
                top_sales  = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
                if top_sales.empty:
                    response_message["text"] = f"No sales data found for top {n} highest sales by country for {months_reverse[month]} {year}."
                else:
                    # Add introductory message for the current month-year
                    response_message["text"] += f"📈 Top {n} Highest Sales by Country for {months_reverse[month]} {year}:\n"
        
                    # Prepare the table data for the response
                    response_data = []
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
        
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []

        # If only year is provided, show results for the entire year
        elif years:
            logging.info(f"top highest sales by country for years:{years}")
            for year in years:
                top_sales  = df[df['PurchaseDate'].dt.year == year].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
                if top_sales.empty:
                    response_message["text"] = f"No sales data found for top {n} highest sales by country for {year}."
                else:
                    response_message["text"] += f"📈 Top {n} Highest Sales by Country for {year}:\n"
        
                    # Prepare the table data for the response
                    response_data = []
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
        
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
                    
        elif quarterly:
            logging.info(f"top highest sales by country for quarterly: {quarterly}")
            (start_month, end_month),year = quarterly
            current_year = datetime.now().year
            
            # Map quarters to names
            quarter_name_map = {
                (1, 3): "First Quarter",
                (4, 6): "Second Quarter",
                (7, 9): "Third Quarter",
                (10, 12): "Fourth Quarter"
            }
            
            quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
            top_sales  = df[
                (df['PurchaseDate'].dt.month >= start_month) &
                (df['PurchaseDate'].dt.month <= end_month) &
                (df['PurchaseDate'].dt.year == year)
            ].groupby('countryname').agg(
            total_sales=('SellingPrice', 'count'),
            total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for {quarter_name} {year}."
            else:
                response_message["text"] += f"📈 Top {n} Highest Sales by Country for {quarter_name} {year}:\n"
        
                # Prepare the table data for the response
                response_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    response_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": response_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []
            
        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            logging.info(f"top highest sales by country for month extracted:{months_extracted}")
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales  = df[(df['PurchaseDate'].dt.month==month) &(df['PurchaseDate'].dt.year == current_year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
                if top_sales.empty:
                    response_message["text"] += f"No sales data found for top {n} highest sales by country for {months_reverse[month]} {current_year}."
                else:
                    response_message["text"] += f"📈 Top {n} Highest Sales by Country for {months_reverse[month]} {current_year}:\n"
        
                    # Prepare the table data for the response
                    response_data = []
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        response_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": ["Country", "Sales Count", "Total Revenue ($)"],
                        "data": response_data
                    })
        
                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []
        elif last_n_months:
            logging.info(f"top highest sales by country for last {last_n_months} months.")
            start_date, end_date,num_months = last_n_months
            top_sales  = df[
                (df['PurchaseDate']>= start_date) &
                (df['PurchaseDate'] <= end_date)
            ].groupby('countryname').agg(
            total_sales=('SellingPrice', 'count'),
            total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last {num_months} months ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last {num_months} Months ({start_date_formatted} to {end_date_formatted}):\n"
        
                # Prepare the table data for the response
                response_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    response_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": response_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []

            
                
        elif half_year:
            logging.info(f"top highest sales by country for half year: {half_year}")
            year,(start_month, end_month) = half_year
            top_sales  = df[
                (df['PurchaseDate'].dt.month >= start_month) &
                (df['PurchaseDate'].dt.month <= end_month) &
                (df['PurchaseDate'].dt.year == year)
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            half_name = "First Half" if start_month == 1 else "Second Half"
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for half-year {half_name} of {year}."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for {half_name} of {year}:\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
                
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []

                
           

        elif last_n_days:
            logging.info(f"Top highest sales by country for last {last_n_days} days")
            start_date, end_date, num_days = last_n_days
            top_sales = df[(df['PurchaseDate'] >= start_date)&(df['PurchaseDate']<=end_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last {num_days} days ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last {num_days} Days ({start_date_formatted} to {end_date_formatted}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []
            
        
        elif last_n_weeks:
            logging.info(f"Top highest sales by country for last {last_n_weeks} weeks")
            start_date, end_date, num_weeks = last_n_weeks
            top_sales = df[df['PurchaseDate'] >= start_date].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last {num_weeks} Weeks ({start_date_formatted} to {end_date_formatted}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []
                    
        
        elif last_n_hours:
            logging.info(f"Top highest sales by country for last {last_n_hours} hours")
            start_time, end_time, num_hours = last_n_hours
            top_sales = df[(df['PurchaseDate'] >= start_time)& (df['PurchaseDate'] <= end_time)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last {num_hours} hours ({start_time} to {end_time})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last {num_hours} Hours ({start_time} to {end_time}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []
                    

        elif fortnight:
            logging.info(f"top highest sales by country for fortnight: {fortnight}")
            start_date, end_date = fortnight
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            logging.info(f"Start date: {start_date}, End date: {end_date}")
            start_date_formatted = start_date.date()  # Get only the date part
            end_date_formatted = end_date.date()
            top_sales  = df[
                (df['PurchaseDate']>= start_date) & 
                (df['PurchaseDate'] <=  end_date)
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last fortnight ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last Fortnight ({start_date_formatted} to {end_date_formatted}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []
            
        elif today:
            today_date = datetime.now().date()
            logging.info(f"top highest sales by country for today...{today_date}")
            top_sales  = df[
                df[df['PurchaseDate'].dt.date == today_date]
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for today {today_date}."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for Today ({today_date}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []

        elif last_day:
            lastday = (datetime.now() - timedelta(days=1)).date()
            logging.info(f"top highest sales by country for the last day.{lastday}")
            top_sales  = df[
                df[df['PurchaseDate'].dt.date == lastday]
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for top {n} highest sales by country for the last day ({lastday})."
            else:
                response_message["text"] = f"📈 Top {n} Highest Sales by Country for the Last Day ({lastday}):\n"
        
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1, row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": ["Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the response message in JSON format
            dispatcher.utter_message(json_message=response_message)
            return []

        
        elif date:
            logging.info(f"top highest sales by country for for specific date: {date}")
            try:
                top_sales = (
                    df[df['PurchaseDate'].dt.date == date]
                    .groupby('PlanName')
                    .agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).sort_values('total_sales', ascending=False).head(n).reset_index()
                )
         
                if top_sales.empty:
                    response_message["text"] = f"No sales data found for top {n} highest sales by country for {date}."
                else:
                    response_message["text"] = f"📈 Top {n} Highest Sales by Country for {date}:\n"
        
                    # Prepare the table data for the response
                    table_data = []
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1, row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": ["Country", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                dispatcher.utter_message(json_message=response_message)
                return []
            except Exception as e:
                response_message["text"] = f"Error processing the specific date: {e}"

                # Send the response message in JSON format
                dispatcher.utter_message(json_message=response_message)
                return []

        # If no filters, show overall top 10 highest sales by country
        else:
            logging.info("top highest sales by country for total")
            top_sales  = df.groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(n).reset_index()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            if top_sales.empty:
                response_message["text"] = f"No sales data found for the top {n} highest sales by country."
            else:
                # Add a message for the top sales by country
                response_message["text"] = f"📈 Top {n} Highest Sales by Country Overall (from {start_date} to {end_date}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            dispatcher.utter_message(json_message=response_message)
            return []


        
    def extract_n_highest_from_message(self, message: str) -> int:
        match = re.search(r'(\d+)\s*(?:highest|top|most|greatest|best|maximum|strongest|leading|high|record|peak|elite|upper)?\s*(?:country|countries)?', message.lower())
        if match:
            return int(match.group(1))  # Extracted number
        return None



class ActionTopLowestSalesByCountry(Action):
    def name(self) -> str:
        return "action_top_lowest_sales_by_country"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info(f"top lowest sales by country")
        global df

        user_message = tracker.latest_message.get('text')
        
        # Extract filters from the user's message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        quarterly = extract_quarters_from_text(user_message)
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_hours = extract_last_n_hours(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_days = extract_last_n_days(user_message)
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        
        response_message = {
            "text": "",
            "tables": []
        }

        n = self.extract_n_from_message(user_message)
        if n is None:
            n = 1

        if month_year_pairs:
            logging.info(f"top country for month year :{month_year_pairs}")
            for month, year in month_year_pairs:
                lowest_sales = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(n).reset_index()
                if lowest_sales.empty:
                    response_message["text"] += f"No sales data found for Top {n} Lowest Sales Countries for {months_reverse[month]} {year}.\n"
                else:
                    response_message["text"] += f"📉 Top {n} Lowest Sales Countries for {months_reverse[month]} {year}:\n"
                    
                    # Prepare the table data for the response
                    table_data = []
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers":  ["Country", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                dispatcher.utter_message(json_message=response_message)
                return []

        elif years:
            logging.info(f"top country for year :{years}")
            for year in years:
                lowest_sales = df[(df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(n).reset_index()
                if lowest_sales.empty:
                    response_message["text"] += f"No sales data available for the top lowest countries in {year}.\n"
                else:
                    response_message["text"] += f"📉 Top lowest {n} Countries for {year}:\n"
                    
                    # Prepare the table data for the response
                    table_data = []
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                dispatcher.utter_message(json_message=response_message)
                return []


        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                lowest_sales = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(n).reset_index()
                if lowest_sales.empty:
                    response_message["text"] += f"No sales data available for the top lowest countries in {months_reverse[month]} {current_year}.\n"
                else:
                    response_message["text"] += f"📉 Top lowest {n} Countries for {months_reverse[month]} {current_year}:\n"
                    
                    # Prepare the table data for the response
                    table_data = []
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                    # Add the table data to the JSON response
                    response_message["tables"].append({
                        "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                        "data": table_data
                    })
                dispatcher.utter_message(json_message=response_message)
                return []
        elif fortnight:
            logging.info(f"top lowest country for fortnight :{fortnight}")
            start_date, end_date = fortnight
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            start_date_formatted = start_date.date()  # Get only the date part
            end_date_formatted = end_date.date()
            lowest_sales = df[(df['PurchaseDate']>=start_date) & (df['PurchaseDate']<=end_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
            else:
                response_message["text"] += f"📉 Top Lowest {n} Country for last fortnight ({start_date_formatted} to {end_date_formatted}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []

        elif quarterly:
            logging.info(f"top lowest country for quarterly: {quarterly}")
            (start_month, end_month),year = quarterly
            current_year = datetime.now().year
            
            # Map quarters to names
            quarter_name_map = {
                (1, 3): "First Quarter",
                (4, 6): "Second Quarter",
                (7, 9): "Third Quarter",
                (10, 12): "Fourth Quarter"
            }
            
            quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
            lowest_sales = df[(df['PurchaseDate'].dt.month>=start_month) & (df['PurchaseDate'].dt.month<=end_month)&(df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country in the {quarter_name} {year}.\n"
            else:
                response_message["text"] += f"📉 Top {n} Lowest Country for {quarter_name} {year}:\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
                
        elif last_day:
            lastday = (datetime.now() - timedelta(days=1)).date()
            logging.info(f"top lowest plans for last_day :{lastday}")
            lowest_sales = df[(df['PurchaseDate']==lastday)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country in the last day {lastday}.\n"
            else:
                response_message["text"] += f"📉 Top Lowest {n} Country for last day ({lastday}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": ["Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
        elif today:
            today_date = datetime.now().date()

            logging.info(f"top lowest country for today :{today_date}")
            lowest_sales = df[(df['PurchaseDate']==today_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country on today {today_date}.\n"
            else:
                response_message["text"] += f"📉 Top Lowest {n} Country for today ({today_date}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the JSON response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
                
        elif date:
            logging.info(f"Processing top lowest country for the specific date: {date}")
    
            # Filter data for the specific date and group by 'PlanName'
            lowest_sales = (
                df[df['PurchaseDate'].dt.date == date]
                .groupby('countryname')
                .agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                )
                .sort_values('total_sales').head(n).reset_index()
            )
        
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country on {date}.\n"
            else:
                response_message["text"] += f"📉 Top lowest {n} country for the specific date {date}:\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
                

        elif last_n_months:
            logging.info(f"top lowest country for last n months :{last_n_months}")
            start_date, end_date, num_months = last_n_months
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            lowest_sales = df[(df['PurchaseDate']>=start_date)&(df['PurchaseDate']<=end_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country in last {num_months} months .\n"
            else:
                response_message["text"] += f"📉 Top Lowest {n} Country for last {num_months} months ({start_date_formatted} to {end_date_formatted}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
        elif last_n_days:
            logging.info(f"Top lowest sales by country for last {last_n_days} days")
            start_date, end_date, num_days = last_n_days
            lowest_sales = df[(df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
        
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
        
            if lowest_sales.empty:
                response_message["text"] += f"No sales data found for top {n} lowest sales by country for the last {num_days} days ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] += f"📉 Top {n} lowest sales by country for the Last {num_days} days ({start_date_formatted} to {end_date_formatted}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
        # For last_n_hours
        elif last_n_hours:
            logging.info(f"Top lowest sales by country for last {last_n_hours} hours")
            start_time, end_time, num_hours = last_n_hours
            lowest_sales = df[(df['PurchaseDate'] >= start_time) & (df['PurchaseDate'] <= end_time)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
        
            if lowest_sales.empty:
                response_message["text"] += f"No sales data found for top {n} lowest sales by country for the last {num_hours} hours ({start_time} to {end_time})."
            else:
                response_message["text"] += f"📉 Top {n} lowest sales by country for the Last {num_hours} hours ({start_time} to {end_time}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
        
                
        # For last_n_weeks
        elif last_n_weeks:
            logging.info(f"Top lowest sales by country for last {last_n_weeks} weeks")
            start_date, end_date, num_weeks = last_n_weeks
            lowest_sales = df[df['PurchaseDate'] >= start_date].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
        
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
        
            if lowest_sales.empty:
                response_message["text"] += f"No sales data found for top {n} lowest sales by country for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted})."
            else:
                response_message["text"] += f"📉 Top {n} lowest sales by country for the Last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
        elif half_year:
            logging.info(f"top lowest country for half year: {half_year}")
            year,(start_month, end_month) = half_year
            current_year = pd.to_datetime('today').year
            lowest_sales = df[
                (df['PurchaseDate'].dt.month == start_month) &                                                      (df['PurchaseDate'].dt.month <= end_month) &
                (df['PurchaseDate'].dt.year == year)
                ].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(n).reset_index()
            half_name = "First Half" if start_month == 1 else "Second Half"
            if lowest_sales.empty:
                response_message["text"] += f"No sales data available for the top lowest country in the {half_name} of {year}."
            else:
                response_message["text"] += f"📉 Top lowest {n} Country for {half_name} of {year}:\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
        
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []     
    
                    
        else:
            logging.info("top country overall")
            lowest_sales = df.groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(n).reset_index()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            if lowest_sales.empty:
                response_message["text"] += "No data available for the top lowest plans overall."
            else:
                # Building the response text
                response_message["text"] += f"📉 Top {n} lowest Country Overall (from {start_date} to {end_date}):\n"
                
                # Prepare the table data for the response
                table_data = []
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([ row['countryname'], row['total_sales'], f"${row['total_revenue']:,.2f}"])
        
                # Add the table data to the response
                response_message["tables"].append({
                    "headers": [ "Country", "Sales Count", "Total Revenue ($)"],
                    "data": table_data
                })
            
            # Send the formatted JSON message
            dispatcher.utter_message(json_message=response_message)
            return []
            



    def extract_n_from_message(self, message: str) -> int:
        match = re.search(r'(\d+)\s*(?:lowest|low|fewest|bottom|least|minimal|smallest|weakest|poorest)?\s*(?:country|countries)?', message.lower())
        # match = re.search(r'(\d+)\s*(?:lowest|low|fewest|bottom)?\s*(?:sales)?\s*(?:country|countries)', message.lower())
        if match:
            return int(match.group(1))  # Extracted number
        return None

#################################################compare countries sales#######################################################################


class ActionCompareCountries(Action):
    def name(self) -> Text:
        return "action_compare_countries"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info(f"Comparing country sales")
        global df
        user_message = tracker.latest_message.get('text')
        logging.info(f"User message: {user_message}")

        # Extract countries, year, and month
        countries = extract_country(user_message)
        quarterly = extract_quarters_from_text(user_message)
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)  
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_hours =  extract_last_n_hours(user_message)
        last_n_days = extract_last_n_days(user_message)
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        logging.info(f"Extracted countries: {countries}")
        


        # Validate that two countries are provided for comparison
        if len(countries) != 2:
            detected_message = f"Detected countries: {', '.join(countries)}" if countries else "No countries detected"
            response_message = {
                "text": "Please provide two countries for comparison."
            }
            dispatcher.utter_message(json_message = response_message)
            return []

        country1, country2 = countries[0], countries[1]
        response_message = {
            "text": "",
            "tables": []
        }
        


        # Filter data by countries and time period
        try:
            if month_year_pairs:
                logging.info(f"compare country sales {month_year_pairs}")
                for month, year in month_year_pairs:
                    df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.year == year) & (df['PurchaseDate'].dt.month == month)]
                    df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.year == year) & (df['PurchaseDate'].dt.month == month)]
                    comparison_type = f"{months_reverse[month].capitalize()} {year}"
            elif months_extracted:
                logging.info(f"compare country sales {months_extracted}")
                for month in months_extracted:
                    current_year = pd.to_datetime('today').year
                    df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.year == current_year) & (df['PurchaseDate'].dt.month == month)]
                    df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.year == current_year) & (df['PurchaseDate'].dt.month == month)]
                    comparison_type = f"{months_reverse[month].capitalize()} {current_year}"
            elif years:
                logging.info(f"compare country sales {years}")
                # Compare whole year
                for year in years:
                    df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.year == year)]
                    df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.year == year)]
                    comparison_type = f"Year {year}"
            elif half_year:
                logging.info(f"compare country sales for {half_year}")
                year,(start_month, end_month)= half_year
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.month >= start_month) & (df['PurchaseDate'].dt.month <= end_month) & (df['PurchaseDate'].dt.year == year)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.month >= start_month) & (df['PurchaseDate'].dt.month <= end_month) & (df['PurchaseDate'].dt.year == year)]
                half_name = "First Half" if start_month == 1 else "Second Half"
                comparison_type = f"{half_name} of {year}"

            elif last_n_months:
                logging.info(f"Comparing country sales for last N months: {last_n_months}")
                start_date, end_date, num_months = last_n_months
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()

                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                comparison_type = f"Last {num_months} months ({start_date_formatted} to {end_date_formatted})"

            # Compare based on last N weeks
            elif last_n_weeks:
                logging.info(f"Comparing country sales for last N weeks: {last_n_weeks}")
                start_date, end_date, num_weeks = last_n_weeks
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                comparison_type = f"Last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted})"
            elif last_n_hours:
                logging.info(f"Comparing country sales for last N hours: {last_n_hours}")
                # Calculate the start and end dates based on the last N hours
                start_time, end_time, num_hours = last_n_hours
                # Filter data for each country for the given time period
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'] >= start_time) & (df['PurchaseDate'] <= end_time)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'] >= start_time) & (df['PurchaseDate'] <= end_time)]
                
            
                # Prepare the comparison type description
                comparison_type = f"Last {num_hours} hours ({start_time} to {end_time})"
    

            # Compare based on last N days
            elif last_n_days:
                start_date, end_date, num_days = last_n_days
                logging.info(f"Comparing country sales for last N days: {last_n_days}")
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                comparison_type = f"Last {num_days} days ({start_date_formatted} to {end_date_formatted})"

            # Compare based on today
            elif date:
                logging.info(f"Comparing country sales for {date}")
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.date == date)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.date == date)]
                comparison_type = f"{date}"
            elif today:
                today_date = datetime.now().date()
                logging.info(f"Comparing country sales for today {today_date}")
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.date == today_date)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.date == today_date)]
                comparison_type = f"Today {today_date}"

            # Compare based on the last day
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Comparing country sales for last day  {lastday}")
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.date == lastday)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.date == lastday)]
                comparison_type = f"Last Day  {lastday}"
            # Handle fortnight comparisons
            elif fortnight:
                logging.info(f"compare country sales {fortnight}")
                start_date, end_date = fortnight
                logging.info(f"Start date: {start_date}, End date: {end_date}")
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date) & (df['PurchaseDate'].dt.year == fortnight[1])]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.day >= start_end) & (df['PurchaseDate'].dt.day <= end_date) ]
                comparison_type = f"Fortnight (from {start_date_formatted} to {end_date_formatted})"
            elif quarterly:
                logging.info(f"Comparing country sales for quarters: {quarterly}")
                (start_month, end_month),year = quarterly
                quarter_name_map = {
                    (1, 3): "First Quarter",
                    (4, 6): "Second Quarter",
                    (7, 9): "Third Quarter",
                    (10, 12): "Fourth Quarter"
                }
                quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
            
                df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.month >= start_month) & (df['PurchaseDate'].dt.month <= end_month) & (df['PurchaseDate'].dt.year == year)]
                df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.month >= start_month) & (df['PurchaseDate'].dt.month <= end_month) &(df['PurchaseDate'].dt.year == year)]
                comparison_type = f"{quarter_name} of {year}"

            else:
                response_message = {
                    "text": "Please specify a valid year or month and year for comparison."
                }
                dispatcher.utter_message(json_response = response_message)
                return []

            
            comparison_result = self.compare_sales(df_country1, df_country2, country1, country2, comparison_type)
            
            response_message["text"] = f"Comparison of sales between {country1} and {country2} for {comparison_type}\n\n"
            response_message["text"] += comparison_result["text"]
            response_message["tables"].extend(comparison_result["tables"])

            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
        
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            response_message = {
                "text": "An error occurred while comparing sales. Please try again later."
            }
            dispatcher.utter_message(json_message = response_message)
            return []
        return []

    def compare_sales(self, df_country1, df_country2, country1, country2, comparison_type):
        result = {
            "text": "",
            "tables": []
        }
        try:
            # Total sales comparison
            total_sales_amount_country1 = df_country1['SellingPrice'].sum()
            total_sales_amount_country2 = df_country2['SellingPrice'].sum()
            total_sales_count_country1 = df_country1['SellingPrice'].count()
            total_sales_count_country2 = df_country2['SellingPrice'].count()
            if int(total_sales_count_country1) == 0 and int(total_sales_count_country2 == 0):
                result["text"]  +=  f"There were no sales in {comparison_type} for both {country1} and {country2}."
                
            else:
                total_sales_json = {
                    "section": f"Sales Comparison for {comparison_type}",
                    "headers": ["Country", "Sales Count", "Sales Price ($)"],
                    "data": [
                        [country1, int(total_sales_count_country1), f"${total_sales_amount_country1:,.2f}"],
                        [country2, int(total_sales_count_country2), f"${total_sales_amount_country2:,.2f}"]
                    ]
                }
    
                result["tables"].append(total_sales_json)
        
                # Generate result summary for country comparison
                sales_count_difference = abs(total_sales_count_country1 - total_sales_count_country2)
                sales_amount_difference = abs(total_sales_amount_country1 - total_sales_amount_country2)
        
                if total_sales_amount_country1 > total_sales_amount_country2:
                    result["text"] += f"\n\n{country1} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country2}.\n\n"
                elif total_sales_amount_country1 < total_sales_amount_country2:
                    result["text"]  += f"\n\n{country2} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country1}.\n\n"
                else:
                    result["text"]  += f"\n\nSales revenue or counts were equal between {country1} and {country2}.\n\n"
                top_plans_country1 = df_country1.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'sum'),
                    sales_count=('SellingPrice', 'count')
                ).reset_index().sort_values(by='sales_count', ascending=False).head(5)
        
                top_plans_country2 = df_country2.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'sum'),
                    sales_count=('SellingPrice', 'count')
                ).reset_index().sort_values(by='sales_count', ascending=False).head(5)
                top_plans_json_country1 = {
                    "section": f"📈 Top 5 Sold Plans in {country1.upper()}",
                    "headers": ["Plan Name", f"{country1} - Sales Count", f"{country1} - Sales Price ($)"],
                    "data": []
                }
        
                for i in range(len(top_plans_country1)):
                    plan = top_plans_country1.iloc[i]
                    top_plans_json_country1["data"].append([
                        plan['PlanName'],
                        int(plan['sales_count']),
                        f"${plan['total_sales']:,.2f}"
                    ])
                top_plans_json_country2 = {
                    "section": f"📈 Top 5 Sold Plans in {country2.upper()}",
                    "headers": ["Plan Name", f"{country2} - Sales Count", f"{country2} - Sales Price ($)"],
                    "data": []
                }
        
                for i in range(len(top_plans_country2)):
                    plan = top_plans_country2.iloc[i]
                    top_plans_json_country2["data"].append([
                        plan['PlanName'],
                        int(plan['sales_count']),
                        f"${plan['total_sales']:,.2f}"
                    ])
                result["tables"].append(top_plans_json_country1)
                result["tables"].append(top_plans_json_country2)
                least_plans_country1 = df_country1.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'sum'),
                    sales_count=('SellingPrice', 'count')
                ).reset_index().sort_values(by='sales_count', ascending=True).head(5)
                
                least_plans_country2 = df_country2.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'sum'),
                    sales_count=('SellingPrice', 'count')
                ).reset_index().sort_values(by='sales_count', ascending=True).head(5)
                least_plans_json_country1 = {
                    "section": f"📉 Least 5 Sold Plans in {country1.upper()}",
                    "headers": ["Plan Name", f"{country1} - Sales Count", f"{country1} - Sales Price ($)"],
                    "data": []
                }
                
                for i in range(len(least_plans_country1)):
                    plan = least_plans_country1.iloc[i]
                    least_plans_json_country1["data"].append([
                        plan['PlanName'],
                        int(plan['sales_count']),
                        f"${plan['total_sales']:,.2f}"
                    ])
                
                least_plans_json_country2 = {
                    "section": f"📉 Least 5 Sold Plans in {country2.upper()}",
                    "headers": ["Plan Name", f"{country2} - Sales Count", f"{country2} - Sales Price ($)"],
                    "data": []
                }
                for i in range(len(least_plans_country2)):
                    plan = least_plans_country2.iloc[i]
                    least_plans_json_country2["data"].append([
                        plan['PlanName'],
                        int(plan['sales_count']),
                        f"${plan['total_sales']:,.2f}"
                    ])
                
                # Add the least plans tables to the result
                result["tables"].append(least_plans_json_country1)
                result["tables"].append(least_plans_json_country2)
    
                
    
        except Exception as e:
            logging.error(f"Error in sales comparison: {e}")
            response_message = {
                "text": "An error occurred while generating the sales comparison."
            }
            dispatcher.utter_message(json_response = message_response)
            return []

        return result
    

    

##############################################################################################################most and least buying sales plan for each country#########################
    
class ActionMostAndLeastSoldPlansForCountry(Action):
    def name(self) -> str:
        return "action_most_and_least_sold_plans_for_country"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info(f"most and leat sold plans for country")
        global df
        
        
        user_message = tracker.latest_message.get('text')

        # Extract filters from the user's message
        country_extracted = extract_country_from_text(user_message)
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        quarterly = extract_quarters_from_text(user_message)
        half_year = extract_half_year_from_text(user_message)
        fortnight = extract_fortnight(user_message)
        today = extract_today(user_message)
        last_n_months = extract_last_n_months(user_message)
        last_n_hours = extract_last_n_hours(user_message)
        last_n_weeks = extract_last_n_weeks(user_message)
        last_n_days = extract_last_n_days(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        


        if not country_extracted:
            logging.info(f"Country {country_extracted} not found in the dataset.")
            response_message = {
                "text" : f"Sorry, we do not have sales data for {country_extracted}. Please provide another country."
            }
            dispatcher.utter_message(json_message=response_message)
            return []
        country = country_extracted[0]
    

        response_message = {
            "text": "",
            "tables": []
        }

        
        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            logging.info(f"most and least sold plans for month year pairs:{month_year_pairs}")
            for month, year in month_year_pairs:
                top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.month == month) &
                    (df['PurchaseDate'].dt.year == year)
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
                if top_sales.empty:
                    response_message["text"] += f"No sales data found for {months_reverse[month]} {year} in {country.upper()}.\n"
                    dispatcher.utter_message(json_message=response_message)
                    return []
                response_message["text"] += f"🔍 Sales Overview for {months_reverse[month]} {year} in {country.upper()}:\n"
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                most_sold_table = []
                for _, row in most_sold.iterrows():
                    most_sold_table.append([
                        row['PlanName'],
                        int(row['SalesCount']),
                        f"${row['TotalRevenue']:,.2f}"
                    ])
            
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": most_sold_table,
                    "section": f"📈 Most Sold Plans in {country.upper()}"
                })
            
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                least_sold_table = []
                for _, row in least_sold.iterrows():
                    least_sold_table.append([
                        row['PlanName'],
                        int(row['SalesCount']),
                        f"${row['TotalRevenue']:,.2f}"
                    ])
            
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": least_sold_table,
                    "section": f"📉 Least Sold Plans in {country.upper()}"
                })
            
                # Send the formatted message
                dispatcher.utter_message(json_message=response_message)
                return []
 
                
                
        elif quarterly:
            logging.info(f"most and least sold plans for quarterly:{quarterly}")
            (start_month, end_month),year = quarterly
            
            
            # Map quarters to names
            quarter_name_map = {
                (1, 3): "First Quarter",
                (4, 6): "Second Quarter",
                (7, 9): "Third Quarter",
                (10, 12): "Fourth Quarter"
            }
            quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
            
            # Fetch sales data for the quarter
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) & 
                    (df['PurchaseDate'].dt.year == year)
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            if top_sales.empty:
                response_message ={
                    "text": f"No sales data found for {quarter_name} {year} in {country.upper()}."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Sales Overview for {quarter_name} {year} in {country.upper()}:\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans in {country.upper()}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans in {country.upper()}"
            })
        
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
 
        elif half_year:
            logging.info(f"most and least sold plans for half year:{half_year}")
            year, (start_month, end_month) = half_year
            current_year = datetime.now().year
            
            # Determine half-year name
            
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.month >= start_month) &
                    (df['PurchaseDate'].dt.month <= end_month) & 
                    (df['PurchaseDate'].dt.year == year)
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            half_name = "First Half" if start_month == 1 else "Second Half"
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for {half_name} {year} in {country.upper()}."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Sales Overview for {half_name} {year} in {country.upper()}:\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans in {country.upper()}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans in {country.upper()}"
            })
        
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
        elif last_n_hours:
            logging.info(f"Filtering data for last {last_n_hours} hours.")
            # Filter the data based on the last N hours
            start_time, end_time, num_hours = last_n_hours
            top_sales = df[
                (df['countryname'].str.lower() == country.lower()) & 
                (df['PurchaseDate'] >= start_time) &
                (df['PurchaseDate'] <= end_time)
            ].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).reset_index()

            if top_sales.empty:
                response_message = {
                    "text":f"No sales data found for the last {num_hours} hours ({start_time} to {end_time}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []

            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Most Sold Plans in {country.upper()} for the Last {num_hours} Hours ({start_time} to {end_time}):\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans in {country.upper()}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans in {country.upper()}"
            })
        
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []


        elif last_n_weeks:
            logging.info(f"Filtering data for last {last_n_weeks} weeks.")
            # Filter the data based on the last N weeks
            start_date, end_date, num_weeks = last_n_weeks
            top_sales = df[
                (df['countryname'].str.lower() == country.lower()) & 
                (df['PurchaseDate'] >= start_date) &
                (df['PurchaseDate'] <= end_date)
            ].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()

            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for the last {num_weeks} weeks ({start_date_formatted} to {end_date_formatted}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []

            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Most Sold Plans in {country.upper()} for the Last {num_weeks} Weeks ({start_date_formatted} to {end_date_formatted}):\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans in {country.upper()}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans in {country.upper()}"
            })
        
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []

        elif last_n_days:
            logging.info(f"Filtering data for last {last_n_days} days.")
            # Filter the data based on the last N days
            start_date, end_date, num_days = last_n_days
            top_sales = df[
                (df['countryname'].str.lower() == country.lower()) & 
                (df['PurchaseDate'] >= start_date) &
                (df['PurchaseDate'] <= end_date)
            ].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()

            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for the last {num_days} days ({start_date_formatted} to {end_date_formatted}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []

            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Most Sold Plans in {country.upper()} for the Last {num_days} Days ({start_date_formatted} to {end_date_formatted}):\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans in {country.upper()}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
        
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans in {country.upper()}"
            })
        
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []

        

        elif fortnight:
            logging.info(f"most and least sold plans for fortnight:{fortnight}")
            start_date, end_date = fortnight
            start_date = pd.to_datetime(start_date)
            end_date = pd.to_datetime(end_date)
            # Fetch sales data for the fortnight
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'] >= start_date) &
                    (df['PurchaseDate'] <= end_date) 
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for the fortnight ({start_date_formatted} to {end_date_formatted}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []
            response_message["text"] += f"🔍 Sales Overview for Last Fortnight ({start_date_formatted} to {end_date_formatted}) in {country.upper()}:\n"
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans on {today_date}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans on {today_date}"
            })
            
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
                

        elif today:
            today_date = datetime.now().date()
            logging.info(f"most and least sold plans for today: {today_date}")
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.date == today_date) 
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for today ({today_date}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []
            response_message["text"] += f"🔍 Sales Overview for Today ({today_date}) in {country.upper()}:\n"
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans on {today_date}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans on {today_date}"
            })
            
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
                
        
        elif last_day:
            # Get the previous day's date
            lastday = (datetime.now() - timedelta(days=1)).date()
            top_sales = df[
                    (df['countryname'].str.lower()== country.lower())&
                    (df['PurchaseDate'].dt.date == lastday) 
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for yesterday ({lastday}) in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return
            
        
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Sales Overview for last day ({lastday}) in {country.upper()}:\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans on {lastday}"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans on {lastday}"
            })
            
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
            
        elif last_n_months:
            logging.info(f"most and least sold plans for last n months...{last_n_months}")
            start_date, end_date, num_months = last_n_months
            # Calculate sales for the range
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate']>= start_date) &
                    (df['PurchaseDate'] <= end_date) 
                ].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for the last {num_months} months  ({start_date_formatted} to {end_date_formatted})  in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []
    
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Sales Overview for the Last {num_months} Months  ({start_date_formatted} to {end_date_formatted}) in {country.upper()}:\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans for {num_months} Months ({start_date_formatted} to {end_date_formatted})"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans for {num_months} Months ({start_date_formatted} to {end_date_formatted})"
            })
            
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
                        
        elif date:
            logging.info(f"most and least sold plans for specific date: {date}")
            # Convert the extracted specific date to a pandas datetime object
            
            # Check if the date is valid
            if pd.isna(date):
                response_message = {
                    "text": f"Sorry, I couldn't understand the date format. Please provide a valid date."
                }
                dispatcher.utter_message(json_message = response_message)
                return []
        
            logging.info(f"Processing sales data for specific date: {date}")
        
            # Calculate sales for the specific date
            top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.date == date) 
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for {date} in {country.upper()}."
                }
                dispatcher.utter_message(json_message = response_message)
                return []
            
        
            # Most sold plans for specific date
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Sales Overview for {date} in {country.upper()}:\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": f"📈 Most Sold Plans on {date}"
            })
        
            # Least sold plans for the specific date
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": f"📉 Least Sold Plans on {date}"
            })
            
            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
                

        # If only year is provided, show results for the entire year
        elif years:
            logging.info(f"most and least sold plans for years:{years}")
            for year in years:
                top_sales = df[
                    (df['countryname'].str.lower() == country.lower())&
                    (df['PurchaseDate'].dt.year == year)
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
                if top_sales.empty:
                    response_message = {
                        "text": f"No sales data found for {year} in {country.upper()}."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message["text"] += f"🔍 Sales Overview for {year} in {country.upper()}:\n"
                # Most Sold Plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                most_sold_table = [ ]
                
                for _, row in most_sold.iterrows():
                    most_sold_table.append([
                        row['PlanName'], 
                        int(row['SalesCount']), 
                        f"${row['TotalRevenue']:,.2f}"
                    ])
                
                # response_message["text"] += f"\n📈 Most Sold Plans\n"
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": most_sold_table,
                    "section": "📈 Most Sold Plans"
                })
        
                # Least Sold Plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                least_sold_table = []
                for _, row in least_sold.iterrows():
                    least_sold_table.append([
                        row['PlanName'], 
                        int(row['SalesCount']), 
                        f"${row['TotalRevenue']:,.2f}"
                    ])
                
                # response_message["text"] += f"\n📉 Least Sold Plans \n"
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": least_sold_table,
                    "section": "📉 Least Sold Plans"
                })
            
            # Send the response message after processing all years
            dispatcher.utter_message(json_message=response_message)
            return []

        
        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            logging.info(f"most and least sold plans for month with current year:{months_extracted}")
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales = df[
                    (df['countryname'].str.lower()== country.lower())&
                    (df['PurchaseDate'].dt.month == month)&
                    (df['PurchaseDate'].dt.year == current_year)
                ].groupby('PlanName').agg(
                    SalesCount=('SellingPrice', 'count'),
                    TotalRevenue=('SellingPrice', 'sum')
                ).reset_index()
                if top_sales.empty:
                    response_message = {
                        "text": f"No sales data found for {months_reverse[month]} {current_year} in {country.upper()}."
                    }
                    dispatcher.utter_message(json_message = response_message)
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message["text"] += f"🔍 Sales Overview for {months_reverse[month]} {current_year} in {country.upper()}:\n"
                most_sold_table = []
                for _, row in most_sold.iterrows():
                    most_sold_table.append([
                        row['PlanName'],
                        int(row['SalesCount']),
                        f"${row['TotalRevenue']:,.2f}"
                    ])
                
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": most_sold_table,
                    "section": "📈 Most Sold Plans"
                })
                
                
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                least_sold_table = []
                for _, row in least_sold.iterrows():
                    least_sold_table.append([
                        row['PlanName'],
                        int(row['SalesCount']),
                        f"${row['TotalRevenue']:,.2f}"
                    ])
        
                response_message["tables"].append({
                    "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                    "data": least_sold_table,
                    "section": "📉 Least Sold Plans"
                })
        
                # Send the response message after processing each month
                dispatcher.utter_message(json_message=response_message)
                return []
                
        # If no filters, show overall top 10 highest sold plans by country
        else:
            logging.info("total most and least sold plans")
            top_sales = df[df['countryname'].str.lower() == country.lower()].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).fillna(0).reset_index()
            if top_sales.empty:
                response_message = {
                    "text": f"No sales data found for {country.upper()}."
                }
                dispatcher.utter_message(json_message=response_message)
                return []
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message["text"] += f"🔍 Overall Most Sold Plans in {country.upper()} (from {start_date} to {end_date}):\n"
            most_sold_table = []
            for _, row in most_sold.iterrows():
                most_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": most_sold_table,
                "section": "📈 Overall Most Sold Plans"
            })
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            least_sold_table = []
            for _, row in least_sold.iterrows():
                least_sold_table.append([
                    row['PlanName'],
                    int(row['SalesCount']),
                    f"${row['TotalRevenue']:,.2f}"
                ])
            
            response_message["tables"].append({
                "headers": ["Plan Name", "Sales Count", "Total Revenue"],
                "data": least_sold_table,
                "section": "📉 Overall Least Sold Plans"
            })

            # Send the formatted message
            dispatcher.utter_message(json_message=response_message)
            return []
                
        if not response_message.strip():
            response_message = {
                "text": "No sales data found for the specified criteria."
            }
            dispatcher.utter_message(json_message_response_message)
            return []

        






#########################################################################################################################calculate refsite,source and payment gateway sales########

class ActionSalesBySourcePaymentGatewayRefsite(Action):

    def name(self) -> str:
        return "action_sales_by_source_payment_gateway_refsite"

    def run(self, dispatcher: CollectingDispatcher, tracker, domain):
        logging.info("Running ActionSalesBySourcePaymentGatewayRefsite...")
        
        global df

        

        user_message = tracker.latest_message.get('text')

        
        # Check if required columns are present
        required_columns = ['PurchaseDate', 'SellingPrice', 'source', 'payment_gateway', 'Refsite']
        missing_columns = [col for col in required_columns if col not in df.columns]
        if missing_columns:
            response_message = {
                    "text":f"Data is missing the following required columns: {', '.join(missing_columns)}"
                }
            dispatcher.utter_message(json_message=response_message)
            return []
           

        source = next(tracker.get_latest_entity_values('source'), None)
        payment_gateway = next(tracker.get_latest_entity_values('payment_gateway'), None)
        refsite = next(tracker.get_latest_entity_values('refsite'), None)

        # Normalize source column by converting to lowercase
        df['source'] = df['source'].str.lower()

        # Extract time conditions from the user message
        years = extract_years(user_message)
        months_extracted = extract_months(user_message)
        month_year_pairs = extract_month_year(user_message)
        

        response_message = {
            "text": "",
            "tables": []
        }

        try: 
            if month_year_pairs:
                logging.info(f"source, refsite,payment_gateway sales in {month_year_pairs}")
                for month, year in month_year_pairs:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]
                    if  filtered_df.empty:
                        response_message["text"] += f"No sales data found for {months_reverse[month]} {year}.\n"
                    else:
                        response_message["text"] += f"📅 Sales Overview for {months_reverse[month].capitalize()} {year}:\n"
                        response_message = self.process_sales_data(filtered_df, response_message)

                        # response_message["tables"].append(self.process_sales_data(filtered_df))
                   

            elif years:
                logging.info(f"source, refsite,payment_gateway sales in {years}")
                
                for year in years:
                    filtered_df = df[df['PurchaseDate'].dt.year == year]
                    if filtered_df.empty:
                        response_message["text"] += f"No sales data found for {year}.\n"
                    else:
                        response_message["text"] += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {year}:\n\n"
                        response_message = self.process_sales_data(filtered_df, response_message)


            elif months_extracted:
                logging.info(f"source, refsite,payment_gateway sales in {months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]
                    if filtered_df.empty:
                        response_message["text"] += f"No sales data found for {months_reverse[month]} {current_year}.\n" 
                    else:
                        response_message["text"] += f"📅 Sales Overview for {months_reverse[month].capitalize()} {current_year}:\n"
                        response_message = self.process_sales_data(filtered_df, response_message)


            else:
                logging.info("source, refsite,payment_gateway sales ")
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                filtered_df = df
                response_message["text"] += f"📊 Overall Sales Overview (from {start_date} to {end_date}) by Source, Payment Gateway, and Refsite:\n\n"
                response_message = self.process_sales_data(filtered_df, response_message)


    
            dispatcher.utter_message(json_message=response_message)
            return []

            

        except Exception as e:
            logging.error(f"An error occurred while processing sales data: {str(e)}")
            response_message = {
                    "text": "An error occurred while processing the sales data. Please try again later."
                }
            dispatcher.utter_message(json_message=response_message)
            return []
            

    def process_sales_data(self, filtered_df, response_message):
        if 'source' in filtered_df.columns:
            response_message = self.get_top_n_sales(filtered_df, 'source', response_message)

        # Process payment gateways
        if 'payment_gateway' in filtered_df.columns:
            response_message = self.get_top_n_sales(filtered_df, 'payment_gateway', response_message)

        # Process refsites
        if 'Refsite' in filtered_df.columns:
            response_message = self.get_top_n_sales(filtered_df, 'Refsite', response_message)

        return response_message
        

    def get_top_n_sales(self, df, column,response_message, n=5):
        sales_summary = df.groupby(column)['SellingPrice'].agg(['sum', 'count']).reset_index()
        sales_summary.columns = [column, 'TotalRevenue', 'SalesCount']
        sales_summary['TotalRevenue'] = pd.to_numeric(sales_summary['TotalRevenue'], errors='coerce').fillna(0)
        sales_summary['SalesCount'] = pd.to_numeric(sales_summary['SalesCount'], errors='coerce').fillna(0)

        top_sales = sales_summary.nlargest(n, 'SalesCount')
        top_sales_data = [(row[column], f"${row['TotalRevenue']:,.2f}", int(row['SalesCount'])) for _, row in top_sales.iterrows()]
        response_message["tables"].append({
            "headers": [column.capitalize(), "Total Revenue ($)", "Sales Count"],
            "data": top_sales_data,
            "section": f"📈 Top {n} Highest Sales by {column.capitalize()}"
        })

        # Get top N lowest sales
        lowest_sales = sales_summary.nsmallest(n, 'SalesCount')
        lowest_sales_data = [(row[column], f"${row['TotalRevenue']:,.2f}", int(row['SalesCount'])) for _, row in lowest_sales.iterrows()]
        response_message["tables"].append({
            "headers": [column.capitalize(), "Total Revenue ($)", "Sales Count"],
            "data": lowest_sales_data,
            "section": f"📉 Top {n} Lowest Sales by {column.capitalize()}"
        })

        return response_message

    

################################################################################################################calculate total sales for each month and each year, sales growth ####################################


class ActionCalculateSalesMetricsAndGrowth(Action):

    def name(self) -> str:
        return "action_calculate_sales_metrics_and_growth"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculateSalesMetrics...")

        global df
        try:
            # Check if the necessary columns exist
            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                response_message = {
                    "text": "Required columns 'PurchaseDate' or 'SellingPrice' are missing from the dataset."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Convert 'PurchaseDate' to datetime
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate', 'SellingPrice'])  # Drop rows with invalid dates or sales data
            
            if df.empty:
                response_message = {
                    "text": "No valid sales data available for analysis."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year

            # Calculate monthly sales totals, counts, averages
            monthly_data = df.groupby("MonthYear").agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count'),
                AvgSalesPrice=('SellingPrice', 'mean')
            ).reset_index()
            monthly_data['SalesPercentageIncrease'] = monthly_data['SalesCount'].pct_change() * 100

            # Calculate monthly average sales count
            monthly_avg_sales_count = monthly_data['SalesCount'].mean()

            # Prepare the response message in JSON format
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()

            response_message = {
                "text": f"📊 Sales Metrics Overview (from {start_date} to {end_date}):\n",
                "tables":[]
            }

            # Monthly Sales Overview (Table format)
            response_message["text"] += "\n📅 Monthly and Yearly Sales Summary:\n"
            
            
            monthly_table = {
                "headers": ["Month-Year", "Total Sales ($)", "Sales Count", "Avg Sales Price ($)", "Sales Percentage Increase (%)"],
                "data": []
            }

            if not monthly_data.empty:
                for _, row in monthly_data.iterrows():
                    monthly_table["data"].append([
                        str(row['MonthYear']),
                        f"${row['TotalSales']:,.2f}",
                        str(row['SalesCount']),
                        f"${row['AvgSalesPrice']:,.2f}",
                        f"{row['SalesPercentageIncrease']:.2f}%" if pd.notnull(row['SalesPercentageIncrease']) else "N/A"
                    
                    ])
                
            else:
                response_message["text"] += "No monthly sales data available.\n\n"
                monthly_table["data"].append(["No data available", "N/A", "N/A", "N/A", "N/A"])

            response_message["tables"].append(monthly_table)

            
            dispatcher.utter_message(json_message=response_message)
        

            yearly_summary = df.groupby('Year').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count'),
                AvgSalesPrice=('SellingPrice', 'mean')
            ).reset_index()
            yearly_summary['SalesPercentageIncrease'] = yearly_summary['SalesCount'].pct_change() * 100  # Percentage change from previous year


            # Calculate yearly average sales count
           
            yearly_table = {
                "headers": ["Year", "Total Sales ($)", "Sales Count", "Avg Sales Price ($)", "Sales Percentage Increase (%)"],
                "data": []
            }

            if not yearly_summary.empty:
                for _, row in yearly_summary.iterrows():
                    yearly_table["data"].append([
                        str(int(row['Year'])),
                        f"${row['TotalSales']:,.2f}",
                        str(row['SalesCount']),
                        f"${row['AvgSalesPrice']:,.2f}", # Add yearly average sales count
                        f"{row['SalesPercentageIncrease']:.2f}%" if pd.notnull(row['SalesPercentageIncrease']) else "N/A"
                    ])
            else:
                response_message_yearly["text"] += "No yearly sales data available.\n\n"
                yearly_table["data"].append(["No data available", "N/A", "N/A", "N/A", "N/A"])

            response_message["tables"].append(yearly_table)

            # Send the yearly sales data as the second part of the response
            dispatcher.utter_message(json_message=response_message)
            

        except Exception as e:
            logging.error(f"Error while calculating sales metrics: {str(e)}")
            response_message = {
                "text": "An error occurred while calculating sales metrics. Please try again later."
            }
            dispatcher.utter_message(json_message=response_message)

            return []




###################################################################################################################repeated registered emails###################################################


class ActionCountRepeatedEmails(Action):
    def name(self) -> str:
        return "action_count_repeated_emails"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCountRepeatedEmails...")

        global df

        try:
            user_message = tracker.latest_message.get('text')
            
            # Extract year, month, and other criteria
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)

            # Check for necessary columns in the dataset
            required_columns = ['Email', 'PlanName', 'PurchaseDate',"IOrderId"]
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                response_message = {
                    "text": "The required columns for processing are missing in the dataset."
                }
                dispatcher.utter_message(json_response = response_message)
                return []
            response_message = {
                "text": "",
                "tables": []
            }

            # Condition: Month-Year specific
            if month_year_pairs:
                logging.info(f"repeated emails in month year pairs: {month_year_pairs}")
                for month, year in month_year_pairs:
                    filtered_data = df[
                        (df['PurchaseDate'].dt.year == year) &
                        (df['PurchaseDate'].dt.month == month)
                    ]
                    email_counts = filtered_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].head(3).index.tolist()
                    total_repeated_in_month_year = len(email_counts[email_counts > 1])

                    if repeated_emails:
                        filtered_data = filtered_data[filtered_data['Email'].isin(repeated_emails)]

                        grouped = (
                            filtered_data.groupby('Email', as_index=False)
                            .agg(
                                Count=('Email', 'size'),
                                Plans=('PlanName', list),
                                Order_ids= ('IOrderId',list),
                                PurchaseDates=('PurchaseDate', list)
                            )
                            .to_dict('records')
                        )
                        for entry in grouped:
                            repeated_email_table = []
                            
                            for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'], entry['Order_ids']):
                                repeated_email_table.append([
                                    plan,  
                                    date.strftime("%d-%m-%y"), 
                                    orders_id
                                ])
                            response_message["tables"].append({
                                "headers": ["Plan", "Purchase Date", "Order ID"],
                                "data": repeated_email_table,
                                "section": f"Email: {entry['Email']}\n\nRepeated {entry['Count']} times"
                                
                            })
                        response_message["text"] += f"📊 Total number of repeated emails in {months_reverse[month].capitalize()} {year}:  {total_repeated_in_month_year}\n\n"
                        response_message["text"] += "\n\nTop 3 most repeaed emails"
                    else:
                        response_message["text"] += f"No repeated email data found for {months_reverse[month].capitalize()} {year}.\n"
                dispatcher.utter_message(json_message=response_message)
                return []

            # Condition: Year specific
            elif years:
                logging.info(f"repeated emails in year : {years}")
                for year in years:
                    filtered_data = df[
                        (df['PurchaseDate'].dt.year == year)
                    ]
                    email_counts = filtered_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].head(3).index.tolist()
                    total_repeated_in_year = len(email_counts[email_counts > 1])
                    if repeated_emails:
                        filtered_data = filtered_data[filtered_data['Email'].isin(repeated_emails)]

                        grouped = (
                            filtered_data.groupby('Email', as_index=False)
                            .agg(
                                Count=('Email', 'size'),
                                Plans=('PlanName', list),
                                Order_ids=('IOrderId', list),
                                PurchaseDates=('PurchaseDate', list)
                            )
                            .to_dict('records')
                        )
                        if grouped:
                            for entry in grouped:
                                repeated_email_table = []
                                
                                # Prepare table data for repeated emails
                                for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'], entry['Order_ids']):
                                    repeated_email_table.append([
                                        plan,  
                                        date.strftime("%d-%m-%y"), 
                                        orders_id
                                    ])
            
                                # Add the data table for each email group
                                response_message["tables"].append({
                                    "headers": ["Plan", "Purchase Date", "Order ID"],
                                    "data": repeated_email_table,
                                    "section": f"Email: {entry['Email']}\n\nRepeated {entry['Count']} times"
                                })
                            response_message["text"] += f"📊 Total number of repeated emails in {year}:  {total_repeated_in_year}"
                            response_message["text"] += "\n\nTop 3 most repeaed emails"

                        # If no repeated emails, add a message indicating so
                        else:
                            response_message["text"] += f"No repeated email data found for {year}."
            
                        # Send the response message
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    
            # Condition: Month specific
            elif months_extracted:
                logging.info(f"repeated emails in month with current year: {months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    filtered_data = df[
                        (df['PurchaseDate'].dt.month == month) &
                        (df['PurchaseDate'].dt.year == current_year)
                    ]
                    email_counts = filtered_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].head(3).index.tolist()
                    total_repeated_in_month = len(email_counts[email_counts > 1])
                    if repeated_emails:
                        filtered_data = filtered_data[filtered_data['Email'].isin(repeated_emails)]

                        grouped = (
                            filtered_data.groupby('Email', as_index=False)
                            .agg(
                                Count=('Email', 'size'),
                                Plans=('PlanName', list),
                                Order_ids=('IOrderId', list),
                                PurchaseDates=('PurchaseDate', list)
                            )
                            .to_dict('records')
                        )
                        if grouped:
                            for entry in grouped:
                                repeated_email_table = []
                                
                                # Prepare table data for repeated emails
                                for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'], entry['Order_ids']):
                                    repeated_email_table.append([
                                        plan,  
                                        date.strftime("%d-%m-%y"), 
                                        orders_id
                                    ])
            
                                # Add the data table for each email group
                                response_message["tables"].append({
                                    "headers": ["Plan", "Purchase Date", "Order ID"],
                                    "data": repeated_email_table,
                                    "section": f"Email: {entry['Email']}\n\nRepeated {entry['Count']} times"
                                })
            
                        # Add the total count of repeated emails in this month
                        response_message["text"] += f"📊 Total number of repeated emails in {months_reverse[month].capitalize()} {current_year}:  {total_repeated_in_month}"
                        response_message["text"] += "\n\nTop 3 most repeaed emails"
            
                    else:
                        response_message["text"] += f"No repeated email data found for {months_reverse[month].capitalize()} {current_year}."
            
                    # Send the response message
                    dispatcher.utter_message(json_message=response_message)
                    return []
                        
                        
            
            elif quarterly:
                logging.info(f"Processing repeated email data for quarterly: {quarterly}")
                try:
                    (start_month, end_month),year = quarterly
                
                    logging.info(f"Start month: {start_month}, End month: {end_month}")

                    # Filter data for the quarter
                    quarter_data = df[
                        (df['PurchaseDate'].dt.month >= start_month) &
                        (df['PurchaseDate'].dt.month <= end_month) &
                        (df['PurchaseDate'].dt.year == year)
                    ]
                    email_counts = quarter_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].head(3).index.tolist()
                    total_repeated_in_quarter = len(email_counts[email_counts > 1])
                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                    if repeated_emails:
                        quarter_data = quarter_data[quarter_data['Email'].isin(repeated_emails)]
                        grouped = (
                            quarter_data.groupby('Email', as_index=False)
                            .agg(
                                Count=('Email', 'size'),
                                Plans=('PlanName', list),
                                Order_ids=('IOrderId', list),
                                PurchaseDates=('PurchaseDate', list)
                            )
                            .to_dict('records')
                        )
                        if grouped:
                            for entry in grouped:
                                repeated_email_table = []
                
                                # Prepare table data for repeated emails
                                for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'], entry['Order_ids']):
                                    repeated_email_table.append([
                                        plan,  
                                        date.strftime("%d-%m-%y"), 
                                        orders_id
                                    ])
                
                                # Add the table to the response
                                response_message["tables"].append({
                                    "headers": ["Plan", "Purchase Date", "Order ID"],
                                    "data": repeated_email_table,
                                    "section": f"Email: {entry['Email']}\n\nRepeated {entry['Count']} times"
                                })
                
                            # Add the total count of repeated emails for this quarter
                        response_message["text"] += f"📊 Total number of repeated emails in {quarter_name} {year}:  {total_repeated_in_quarter}"
                        response_message["text"] += "\n\nTop 3 most repeaed emails"
                
                    else:
                        response_message["text"] += f"No repeated email data found for {quarter_name} {year}."
                
                        # Send the response message
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
                except Exception as e:
                    logging.error(f"Error processing quarterly repeated emails: {e}")
                    response_message = {
                        "text": "An error occurred while processing quarterly repeated email details. Please try again later."
                    }
                    dispatcher.utter_message(json_message = response_message )
                    return []
                        
                        
                        
            # Default: Overall condition
            else:
                logging.info("repeated emails in overall")
                email_counts = df['Email'].value_counts()
                repeated_emails = email_counts[email_counts > 1].head(3).index.tolist()
                total_repeated_emails = len(email_counts[email_counts > 1])
                
    
                if not repeated_emails:
                    dispatcher.utter_message(text="There are no repeated emails in the data.")
                    return []
                repeated_data = df[df['Email'].isin(repeated_emails)]
                grouped = (
                    repeated_data.groupby('Email', as_index=False)
                    .agg(
                        Count=('Email', 'size'),
                        Plans=('PlanName', list),
                        Order_ids=('IOrderId', list),
                        PurchaseDates=('PurchaseDate', list)
                    )
                    .to_dict('records')
                )
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if grouped:
                    for entry in grouped:
                        repeated_email_table = []
            
                        # Prepare table data for repeated emails
                        for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'], entry['Order_ids']):
                            repeated_email_table.append([
                                plan,  
                                date.strftime("%d-%m-%y"), 
                                orders_id
                            ])
            
                        # Add the table to the response
                        response_message["tables"].append({
                            "headers": ["Plan", "Purchase Date", "Order ID"],
                            "data": repeated_email_table,
                            "section":f"\nEmail: {entry['Email']}\n\nRepeated {entry['Count']} times"
                        })
            
                    # Add the total count of repeated emails to the response
                    response_message["text"] += f"📊 Total number of repeated emails (From {start_date} to {end_date}): {total_repeated_emails}"
                    response_message["text"] += "\n\nTop 3 most repeaed emails"
            
                else:
                    response_message["text"] += f"No repeated email data found."

                # Send the response message
                dispatcher.utter_message(json_message=response_message)
                return []
               

            
        except Exception as e:
            logging.error(f"Error processing repeated email details: {e}")
            response_message = {
                "text": "An error occurred while retrieving repeated email details. Please try again later."
            }
            dispatcher.utter_message(response_message)
            return []
        ############################################################################################################profit margin#######################################
class ActionGetProfitMargin(Action):
    def name(self) -> Text:
        return "action_get_profit_margin"

    def run(self, dispatcher: CollectingDispatcher,
            tracker: Tracker,
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        
        try:
            global df
            logging.info("Running ActionGetProfitMargin...")
            if df.empty or df['SellingPrice'].isnull().all() or df['CompanyBuyingPrice'].isnull().all():
                response_message = {
                    "text": "Error: The sales data is empty or invalid."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Add a ProfitMargin column to the DataFrame
            df['ProfitMargin'] = df['SellingPrice'] - df['CompanyBuyingPrice']

            user_message = tracker.latest_message.get('text')
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            last_n_hours = extract_last_n_hours(user_message)
            last_n_weeks = extract_last_n_weeks(user_message)
            last_n_days = extract_last_n_days(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            date = extract_date(user_message)
            monthwise = extract_monthwise(user_message)
            yearwise = extract_yearwise(user_message)
            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year

            total_profit_margin = 0.0
            
            #Handle specific date
            
            if date:
                logging.info(f"Profit margin for specific date: {date}")
                daily_profit_margin, error_message = extract_profit_margin_sales_for_specific_date(df, date)
                if error_message:  # Check if there's an error message
                    response_message = {
                        "text": error_message
                    }
                    
    
                elif daily_profit_margin > 0:
                    response_message = {
                        "text": f"The profit margin for {date} is ${daily_profit_margin:.2f}."
                    }
                    
                else:
                    response_message = {
                        "text": f"No sales were recorded on {date}."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
                

            # Handle today's profit margin
            elif today:
                today_date = datetime.now().date()
                logging.info(f"Profit margin of today {today_date}")
                today_profit_margin = df[df['PurchaseDate'].dt.date == today_date]['ProfitMargin'].sum()
                if today_profit_margin>0:
                    response_message = {
                        "text": f"The profit margin for today ({today_date}) is ${today_profit_margin:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No profit margin was recorded for today ({today_date})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Profit margin of last day {lastday}")
                yesterday_profit_margin = df[df['PurchaseDate'].dt.date == lastday]['ProfitMargin'].sum()
                if yesterday_profit_margin>0:
                    response_message = {
                        "text": f"The profit margin for yesterday ({lastday}) is ${yesterday_profit_margin:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No profit margin was recorded for last day ({lastday})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            elif fortnight:
                logging.info(f"Profit margin of fortnight {fortnight}")
                start_date, end_date = fortnight
                start_date = pd.to_datetime(start_date)
                end_date = pd.to_datetime(end_date)
                logging.info(f"Start date: {start_date}, End date: {end_date}")
                start_date_formatted = start_date.date()  # Get only the date part
                end_date_formatted = end_date.date()
                logging.info(f"Start date: {start_date}, End date: {end_date}")
                fortnight_profit_margin = df[
                    (df['PurchaseDate'] >= start_date) & 
                    (df['PurchaseDate'] <= end_date)
                ]['ProfitMargin'].sum()
                if fortnight_profit_margin:
                    response_message = {
                        "text": f"The profit margin for the last fortnight (from {start_date_formatted} to {end_date_formatted}) is ${fortnight_profit_margin:.2f}."
                    }
                else:
                    response_message = {
                        "text": f"No profit margin was recorded for the last fortnight (from {start_date_formatted} to {end_date_formatted})."
                    }
                dispatcher.utter_message(json_message=response_message)
                return []
            elif last_n_months:
                logging.info(f"Profit margin of last {last_n_months} months")
                try:
                    start_date, end_date, num_months = last_n_months
                    
                    last_months_profit_margin = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]['ProfitMargin'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    if last_months_profit_margin > 0:
                        response_message = {
                            "text": f"The profit margin for the last {num_months} months (from {start_date_formatted} to {end_date_formatted}) is ${last_months_profit_margin:.2f}."
                        }
                    
                    else:
                        response_message = {
                            "text": f"No profit margin recorded for the last {num_months} months (from {start_date_formatted} to {end_date_formatted})."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin: {e}")
                    response_message = {
                        "text": "Could not process the profit margin for the last months. Please provide a valid date range."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            elif quarterly:
                logging.info(f"profit margin of quarterly {quarterly}")
                # Map quarters to start and end months
                try:
                    (start_month, end_month),year = quarterly
                    current_year = pd.to_datetime('today').year
                    quarterly_profit_margin = df[
                        (df['PurchaseDate'].dt.month >= start_month) & 
                        (df['PurchaseDate'].dt.month <= end_month) & 
                        (df['PurchaseDate'].dt.year == year)
                    ]['ProfitMargin'].sum()
                    
                    
                    quarter_name_map = {
                        (1, 3): "First Quarter",
                        (4, 6): "Second Quarter",
                        (7, 9): "Third Quarter",
                        (10, 12): "Fourth Quarter"
                    }
                    quarter_name = quarter_name_map.get((start_month, end_month), "Quarter")
                    if quarterly_profit_margin>0:
                        response_message = {
                            "text": f"The profit margin for {quarter_name} of {year} is ${quarterly_profit_margin:.2f}."
                        }
                    else:
                        response_message = {
                            "text": f"No profit margin was recorded for {quarter_name} of {year}."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    response_message = {
                        "text": f"An error occurred while processing quarterly sales: {str(e)}"
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                    
                
            elif monthwise:
                logging.info("Calculating monthwise profit margin")
                month_profit_margin = df.groupby('MonthYear').agg(
                    TotalProfitMargin=('ProfitMargin', 'sum'),
                    AvgProfitMargin=('ProfitMargin', 'mean')
                ).reset_index()
            
                # Calculate the percentage increase in ProfitMargin compared to the previous month
                month_profit_margin['ProfitMarginPercentageIncrease'] = month_profit_margin['TotalProfitMargin'].pct_change() * 100
            
                # Convert 'MonthYear' to string for display purposes
                
                # month_profit_margin = df.groupby('MonthYear')['ProfitMargin'].sum().reset_index()
                month_profit_margin['MonthYear'] = month_profit_margin['MonthYear'].astype(str)
                response_message = {
                    "text": "",
                    "tables": []
                }
                if month_profit_margin.empty:
                    response_message["text"] += "No monthwise profit margin data available."
                else:
                    response_message["text"] += "📊 Monthwise Profit Margin Summary:\n"
                    table = []
                    for _, row in month_profit_margin.iterrows():
                        table.append([
                            str(row['MonthYear']),
                            f"${row['TotalProfitMargin']:,.2f}",
                            f"${row['AvgProfitMargin']:,.2f}",
                            f"{row['ProfitMarginPercentageIncrease']:.2f}%" if pd.notna(row['ProfitMarginPercentageIncrease']) else "N/A"
                        ])
                        
                    
                    response_message["tables"].append({
                        "headers": ["Month-Year", "Profit Margin ($)", "Average Profit Margin ($)", "Percentage Increase (%)"],
                        "data": table
                    })
                dispatcher.utter_message(json_message=response_message)
                return []
                
                
            
            # Handle yearwise profit margin
            elif yearwise:
                logging.info("Calculating yearwise profit margin")
    
                year_profit_margin = df.groupby('Year').agg(
                    TotalProfitMargin=('ProfitMargin', 'sum'),
                    AvgProfitMargin=('ProfitMargin', 'mean')
                ).reset_index()
                year_profit_margin['ProfitMarginPercentageIncrease'] = year_profit_margin['TotalProfitMargin'].pct_change() * 100
                if 'Year' not in df.columns or 'ProfitMargin' not in df.columns:
                    raise ValueError("Missing 'Year' or 'ProfitMargin' columns in the dataframe.")
                year_profit_margin['Year'] = year_profit_margin['Year'].astype(str)
                response_message = {
                    "text": "",
                    "tables": []
                }
                
                if year_profit_margin.empty:
                    response_message["text"] += "No yearwise profit margin data available."
                else:
                    response_message["text"] += "📊 Yearwise Profit Margin Summary:\n"
                    table = []
                    for _, row in year_profit_margin.iterrows():
                        table.append([
                            str(row['Year']),
                            f"${row['TotalProfitMargin']:,.2f}",
                            f"${row['AvgProfitMargin']:,.2f}",
                            f"{row['ProfitMarginPercentageIncrease']:.2f}%" if pd.notna(row['ProfitMarginPercentageIncrease']) else "N/A"
                        ])
                    
                    response_message["tables"].append({
                        "headers": ["Year", "Profit Margin ($)","Average Profit Margin ($)", "Percentage Increase (%)"],
                        "data": table
                    })
                
                dispatcher.utter_message(json_message=response_message)
                return []
                

            elif last_n_hours:
                logging.info(f"Profit margin for the last {last_n_hours} hours")
                try:
                    start_time, end_time, num_hours = last_n_hours
                    last_hours_profit_margin = df[
                        (df['PurchaseDate'] >= start_time) &
                        (df['PurchaseDate'] <= end_time)
                    ]['ProfitMargin'].sum()
                    if last_hours_profit_margin > 0:
                        response_message = {
                            "text": f"The profit margin for the last {num_hours} hours (from {start_time} to {end_time}) is ${last_hours_profit_margin:.2f}."
                        }
                        
                    else:
                        response_message = {
                            "text": f"No profit margin recorded for the last {num_hours} hours (from {start_time} to {end_time})."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin for last {num_hours} hours: {e}")
                    response_message = {
                        "text": "Could not process the profit margin for the last hours. Please provide a valid time range."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []

            # Handle last N weeks profit margin
            elif last_n_weeks:
                logging.info(f"Profit margin for the last {last_n_weeks} weeks")
                try:
                    start_date, end_date, num_weeks = last_n_weeks
                    last_weeks_profit_margin = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]['ProfitMargin'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    if last_weeks_profit_margin > 0:
                        response_message = {
                            "text": f"The profit margin for the last {num_weeks} weeks (from {start_date_formatted} to {end_date_formatted}) is ${last_weeks_profit_margin:.2f}."
                        }
                        
                    else:
                        response_message = {
                            "text": f"No profit margin recorded for the last {num_weeks} weeks (from {start_date_formatted} to {end_date_formatted})."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin for last {num_weeks} weeks: {e}")
                    response_message = {
                        "text": "Could not process the profit margin for the last weeks. Please provide a valid week range."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
            elif last_n_days:
                logging.info(f"Profit margin for the last {last_n_days} days")
                try:
                    start_date, end_date, num_days = last_n_days
                    last_days_profit_margin = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]['ProfitMargin'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    if last_days_profit_margin > 0:
                        response_message = {
                            "text": f"The profit margin for the last {num_days} days (from {start_date_formatted} to {end_date_formatted}) is ${last_days_profit_margin:.2f}."
                        }
                    else:
                        response_message = {
                            "text": f"No profit margin recorded for the last {num_days} days (from {start_date_formatted} to {end_date_formatted})."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin for last {num_days} days: {e}")
                    response_message = {
                        "text": "Could not process the profit margin for the last days. Please provide a valid day range."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
            elif half_year:
                logging.info(f"Profit margin of half yearly {half_year}")
                year,(start_month, end_month) = half_year
              
                try:
                    half_yearly_profit_margin_data = df[
                        (df['PurchaseDate'].dt.month >= start_month) & 
                        (df['PurchaseDate'].dt.month <= end_month) & 
                        (df['PurchaseDate'].dt.year == year)
                    ]['ProfitMargin'].sum()
            
                
                    # Determine whether it's the first or second half of the year
                    half_name = "First Half" if start_month == 1 else "Second Half"
                    if half_yearly_profit_margin_data>0:
                        response_message = {
                            "text": f"The profit margin for the {half_name} of {year} is ${half_yearly_profit_margin_data:.2f}."
                        }
                    else:
                        response_message = {
                            "text": f"No profit margin was recorded for the {half_name} of {year}."
                        }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin for {half_name} of {year}: {e}")
                    response_message = {
                        "text": f"Error calculating profit margin for the {half_name} of {year}. Please try again."
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                    
                   

            # Handle month-year pairs
            elif month_year_pairs:
                logging.info(f"profit margin of month year {month_year_pairs}")
                try:
                    for month, year in month_year_pairs:
                        month_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == year)
                        ]['ProfitMargin'].sum()
                        if month_profit_margin>0:
                            response_message = {
                                "text": f"The profit margin for {months_reverse[month]} {year} is ${month_profit_margin:.2f}."
                            }
                        else:
                            response_message = {
                                "text":f"No profit margin was recorded for {months_reverse[month]} {year}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                        
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing monthly profit margins: {str(e)}"
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []

            # Handle years
            elif years:
                logging.info(f"profit margin of year {years}")
                try:
                    for year in years:
                        yearly_profit_margin = df[
                            (df['PurchaseDate'].dt.year == year)
                        ]['ProfitMargin'].sum()
                        if yearly_profit_margin>0:
                            response_message = {
                                "text": f"The total profit margin for {year} is ${yearly_profit_margin:.2f}."
                            }
                        else:
                            response_message = {
                                "text": f"No profit margin was recorded for {year}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                    
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing yearly profit margins: {str(e)}"
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
            

            # Handle months in the current year
            elif months_extracted:
                logging.info(f"profit margin of month with current year {months_extracted}")
                current_year = pd.to_datetime('today').year
                try:
                    for month in months_extracted:
                        monthly_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == current_year)
                        ]['ProfitMargin'].sum()
                        if monthly_profit_margin >0:
                            response_message= {
                                "text": f"The profit margin for {months_reverse[month]} {current_year} is ${monthly_profit_margin:.2f}."
                            }
                        else:
                            response_message = {
                                "text": f"No profit margin was recorded for {months_reverse[month]} {current_year}."
                            }
                        dispatcher.utter_message(json_message=response_message)
                        return []
                        
                except Exception as e:
                    response_message = {
                        "text": f"Error occurred while processing monthly profit margins: {str(e)}"
                    }
                    dispatcher.utter_message(json_message=response_message)
                    return []
                
            else:

                # Handle total profit margin
                logging.info("total profit margin")
                total_profit_margin = df['ProfitMargin'].sum()
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                response_message= {
                    "text": f"The overall total profit margin (from {start_date} to {end_date}) is ${total_profit_margin:.2f}."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

        except Exception as e:
            response_message:{
                "text": f"An error occurred while processing your request: {str(e)}"
            }
            dispatcher.utter_message(json_message=response_message)
            return []
        
       

################################################################country sales metric########################

class ActionCalculateCountrySalesMetrics(Action):
    
    def name(self) -> str:
        return "action_calculate_country_sales_metrics"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculateCountrySalesMetrics...")

        global df  # Assuming 'df' is the DataFrame that contains the sales data
        
        try:
            # Check if the necessary columns exist
            required_columns = ['countryname', 'SellingPrice', 'PurchaseDate']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                response_message = {
                    "text": f"Required columns missing from the dataset: {', '.join(missing_columns)}"
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Check if the dataset is empty
            if df.empty:
                response_message = {
                    "text": "No valid sales data available for analysis"
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Convert PurchaseDate to datetime and handle any invalid dates
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate'])  # Drop rows with invalid PurchaseDate

            # Calculate country-wise sales totals and counts
            country_data = df.groupby('countryname').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()
            country_data.columns = ['countryname', 'TotalSales', 'SalesCount']
            country_data = country_data.sort_values(by='SalesCount', ascending=False)
            country_data = country_data.reset_index(drop=True)

            # Get the date range for the dataset
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            total_country = len(country_data)

            # Prepare the response message
            response_message = {
                "text": "",
                "tables": []
            }

            if country_data.empty:
                response_message["text"] += "No sales data available by country.\n\n"
            else:
                response_message["text"] += f"📊 Country Sales Metrics Overview (from {start_date} to {end_date}):\n\n"
                response_message["text"] += f"Total Number of Unique country: {total_country}\n\n"
                response_message["text"] += "🌍 Sales by Country:\n"
                table_data = {
                    "headers": ["Country", "Total Sales", "Sales Count"],
                    "data": []
                }
                for _, row in country_data.iterrows():
                    table_data["data"].append([row['countryname'], f"${row['TotalSales']:,.2f}", row['SalesCount']])
                response_message["tables"].append(table_data)


            # Send the response message
            dispatcher.utter_message(json_message=response_message)
            return []

        except Exception as e:
            logging.error(f"Error while calculating country sales metrics: {str(e)}")
            response_message = {
                "text": "An error occurred while calculating country sales metrics. Please try again later."
            }
            dispatcher.utter_message(json_message=response_message)
            return []

#####################################################planname sales metrics#####################

class ActionCalculatePlanSalesMetrics(Action):

    def name(self) -> str:
        return "action_calculate_plan_sales_metrics"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculatePlanSalesMetrics...")

        global df
        try:
            # Check if the necessary columns exist
            if 'PlanName' not in df.columns or 'SellingPrice' not in df.columns:
                response_message = {
                    "text": "Required columns 'PlanName' or 'SellingPrice' are missing from the dataset."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            df = df.dropna(subset=['PlanName', 'SellingPrice','PurchaseDate'])  # Drop rows with invalid plan or sales data
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

            # Filter data to exclude sales before January 1, 2024
            two_months_ago = datetime.now() - timedelta(days=2 * 30)
            today = datetime.now()
            df = df[(df['PurchaseDate'] >= two_months_ago) & (df['PurchaseDate'] <= today)]
            if df.empty:
                response_message = {
                    "text": "No valid sales data available for analysis."
                }
                dispatcher.utter_message(json_message=response_message)
                return []

            # Calculate sales totals, counts, averages by PlanName
            plan_data = df.groupby('PlanName').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count'),
                AvgSalesPrice=('SellingPrice', 'mean')
            ).reset_index()

            # Ensure the column names are properly aligned and no tuples are passed
            plan_data.columns = ['PlanName', 'TotalSales', 'SalesCount', 'AvgSalesPrice']
            
            plan_data = plan_data.sort_values(by='SalesCount', ascending=False)
            plan_data = plan_data.reset_index(drop=True)
            start_date = two_months_ago.date()
            end_date = today.date()
            total_plans = len(plan_data)

            # Prepare the JSON response
            response_message = {
                "text": "",
                "tables": []
            }

            # Sales by Plan Summary
            
            if plan_data.empty:
                response_message["text"] += "No sales data available by plan.\n\n"
            else:
                response_message["text"] += f"📊 Plan Sales Metrics Overview (From {start_date} to {end_date})\n\n"
                response_message["text"] += f"Total Number of Unique Plans: {total_plans}\n\n"
                response_message["text"] += " 📅 Sales by Plan for last 2 months:\n"
                # Create the table format
                plan_table = {
                    "headers": ["Plan Name", "Total Sales ($)", "Sales Count", "Avg Sales Price ($)"],
                    "data": []
                }
                for _, row in plan_data.iterrows():
                    plan_table["data"].append([row['PlanName'], f"${row['TotalSales']:,.2f}", row['SalesCount'], f"${row['AvgSalesPrice']:,.2f}"])

                response_message["tables"].append(plan_table)

            # Send the JSON formatted response
            dispatcher.utter_message(json_message=response_message)

        except Exception as e:
            logging.error(f"Error while calculating plan sales metrics: {str(e)}")
            response_message = {
                "text": "An error occurred while calculating plan sales metrics. Please try again later."
            }
            dispatcher.utter_message(json_message=response_message)
        return []

###############################################################################predicted sales questions#####################################
class ActionSalesPrediction(Action):
    def name(self) -> Text:
        return "action_sales_prediction"

    def run(self, 
            dispatcher: CollectingDispatcher, 
            tracker: Tracker, 
            domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        

        # Generate forecasts
        df1, df2, df3 = generate_sales_forecasts(df)
        user_message = tracker.latest_message.get('text', '').lower()
        
        # Get the condition from the user's query (slot or intent)
        condition = tracker.get_slot("sales_condition")
        if not condition:
            user_message = tracker.latest_message.get('text', '').lower()
            if "daily" in user_message:
                condition = "daily"
            elif "monthly" in user_message:
                condition = "monthly"
            elif "yearly" in user_message:
                condition = "yearly"
            else:
                condition = None
        response_message = {
            "text": "📊 Sales Forecast Overview:\n",
            "tables": []
        }
        def serialize_dates(df):
                for col in df.select_dtypes(include=["datetime", "datetime64[ns]", "object"]).columns:
                    if df[col].dtype == "datetime64[ns]":
                        df[col] = df[col].dt.strftime('%Y-%m-%d')  # Format datetime as string
                    elif df[col].dtype == "object":  # Check for potential string representation of dates
                        df[col] = pd.to_datetime(df[col], errors='coerce').dt.strftime('%Y-%m-%d')
                return df
        # Respond based on the condition
        if condition == "daily":
            # Create table for daily sales forecast
            df1 = serialize_dates(df1)
            daily_table = {
                "headers": df1.columns.tolist(),
                "data": df1.values.tolist()
            }
            response_message["text"] += "\n📅 Daily Sales Forecast:"
            response_message["tables"].append(daily_table)

        elif condition == "monthly":
            # Create table for monthly sales forecast
            df2 = serialize_dates(df2)
            monthly_table = {
                "headers": df2.columns.tolist(),
                "data": df2.values.tolist()
            }
            response_message["text"] += "\n📅 Monthly Sales Forecast:\n"
            response_message["tables"].append(monthly_table)

        elif condition == "yearly":
            # Create table for yearly sales forecast
            df3 = serialize_dates(df3)
            yearly_table = {
                "headers": df3.columns.tolist(),
                "data": df3.values.tolist()
            }
            response_message["text"] += "\n📅 Yearly Sales Forecast:\n"
            response_message["tables"].append(yearly_table)

        else:
            response_message["text"] = "I couldn't understand the time frame. Please specify daily, monthly, or yearly."

        # Send the JSON formatted response
        dispatcher.utter_message(json_message=response_message)

        return []
        
##########################################################highest sales count in country##############################


        
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
