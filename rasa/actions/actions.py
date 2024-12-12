from fuzzywuzzy import process
from apscheduler.schedulers.background import BackgroundScheduler
from tabulate import tabulate
from typing import Any, Text, Dict, Tuple, List, Optional
from rasa_sdk import Action, Tracker
from rasa_sdk.events import SlotSet
from rasa_sdk.events import BotUttered
from rasa_sdk.executor import CollectingDispatcher
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor
from datetime import datetime,timedelta
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import re
import os
from .utils import df,generate_sales_forecasts
from dateutil.relativedelta import relativedelta
def background_forecast_job():
    generate_sales_forecasts(df)

# Schedule the job to run automatically (e.g., every 10 days)
scheduler = BackgroundScheduler()
scheduler.add_job(background_forecast_job, 'date', run_date=datetime.now() + timedelta(days=10))
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

# Dictionary for mapping month names to numbers

def extract_date(text):
    months = {
        'january': 1, 'jan': 1, 'february': 2, 'feb': 2, 'march': 3, 'mar': 3,
        'april': 4, 'apr': 4, 'may': 5, 'june': 6, 'jun': 6, 'july': 7, 'jul': 7,
        'august': 8, 'aug': 8, 'september': 9, 'sep': 9, 'sept': 9, 'october': 10,
        'oct': 10, 'november': 11, 'nov': 11, 'december': 12, 'dec': 12
    }
    # Remove ordinal suffixes
    cleaned_message = re.sub(r'(\d+)(st|nd|rd|th)', r'\1', text)

    # Refined regex patterns
    date_patterns = [
        r'\b\d{4}-\d{2}-\d{2}\b',  # YYYY-MM-DD
        r'\b\d{1,2}[-/]\d{1,2}[-/]\d{4}\b',  # DD-MM-YYYY or MM-DD-YYYY
        r'\b\d{1,2} [A-Za-z.]+ \d{4}\b',  # DD Month YYYY (with or without period)
        r'\b[A-Za-z.]+ \d{1,2},? \d{4}\b',  # Month DD, YYYY (with optional comma)
        r'\b\d{1,2}-[A-Za-z.]+-\d{4}\b'  # DD-Month-YYYY
    ]

    # Combine patterns
    combined_pattern = r'|'.join(date_patterns)
    matches = re.findall(combined_pattern, cleaned_message, re.IGNORECASE)

    # Convert matches
    dates = []
    for match in matches:
        match = match.replace(',', '').strip()
        try:
            # Direct parse for YYYY-MM-DD
            if re.match(r'^\d{4}-\d{2}-\d{2}$', match):
                dates.append(match)
            # Parse DD-MM-YYYY or MM-DD-YYYY
            elif re.match(r'^\d{1,2}[-/]\d{1,2}[-/]\d{4}$', match):
                day, month, year = re.split(r'[-/]', match)
                dates.append(pd.Timestamp(year=int(year), month=int(month), day=int(day)).strftime('%Y-%m-%d'))
            # Parse DD Month YYYY
            elif re.match(r'^\d{1,2} [A-Za-z.]+ \d{4}$', match):
                day, month_name, year = re.split(r'\s+', match)
                month = months.get(month_name.lower().rstrip('.'))
                dates.append(pd.Timestamp(year=int(year), month=month, day=int(day)).strftime('%Y-%m-%d'))
            # Parse Month DD YYYY
            elif re.match(r'^[A-Za-z.]+ \d{1,2} \d{4}$', match):
                month_name, day, year = re.split(r'\s+', match)
                month = months.get(month_name.lower().rstrip('.'))
                dates.append(pd.Timestamp(year=int(year), month=month, day=int(day)).strftime('%Y-%m-%d'))
        except Exception as e:
            print(f"Error processing date: {match}, Error: {e}")

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


def extract_last_n_months(text):
    pattern = r'\b(last\s+(\d+|one|two|three|four|five|six|seven|eight|nine|ten|eleven|twelve)\s+months?)\b'
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        num_months_str = match.group(2).lower()
        num_months = int(num_months_str) if num_months_str.isdigit() else word_to_num.get(num_months_str, 0)
        
        # Get the current date
        today = datetime.now()

        
        start_date = today - relativedelta(months=num_months)
        
        return start_date, today  , num_months
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
                dispatcher.utter_message(text="Error: The sales data is empty or invalid.")
                return []

            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            total_sales_count = 0
            total_sales_price = 0.0
            specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            #date_range = extract_date_range(user_message)
            
           

            if specific_date_text:
                logging.info(f"Processing sales data for specific date: {specific_date_text}")
                
                daily_sales_count, daily_sales_price = extract_sales_for_specific_date(df, specific_date_text)
                if daily_sales_count is not None:
                    if daily_sales_count > 0:
                        dispatcher.utter_message(text=f"The total sales for {specific_date_text} is {daily_sales_count} with a sales price of ${daily_sales_price:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded on {specific_date_text}.")
                else:
                    dispatcher.utter_message(text=daily_sales_price)  # Return the error message from the function
                return []
            # if date_range :
            #     logging.info(f"date range...{date_range}")
            #     start_date, end_date, error_message = extract_date_range(date_range)
    
            #     if error_message:
            #         dispatcher.utter_message(text=error_message)
            #         return []
            #     logging.info(f"Extracted start_date: {start_date}, end_date: {end_date}")
            #     count_range = df[
            #         (df['PurchaseDate'] >= pd.Timestamp(start_date)) & 
            #         (df['PurchaseDate'] <= pd.Timestamp(end_date))
            #     ]['SellingPrice'].count()
                
            #     price_range = df[
            #         (df['PurchaseDate'] >= pd.Timestamp(start_date)) & 
            #         (df['PurchaseDate'] <= pd.Timestamp(end_date))
            #     ]['SellingPrice'].sum()
            #     if count_range > 0:
            #         dispatcher.utter_message(text=f"The total sales from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')} is {count_range} with a sales price of ${price_range:,.2f}.")
            #     else:
            #         dispatcher.utter_message(text=f"No sales were recorded from {start_date.strftime('%Y-%m-%d')} to {end_date.strftime('%Y-%m-%d')}.")
            # else:
            #     dispatcher.utter_message(text="No valid date range was provided.")
            # return []
                
            # Today's sales
            if today:
                today_date = datetime.now().date()
                logging.info(f"Processing today's sales...{today_date}")
                today_sales_count = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].count()
                today_sales_price = df[df['PurchaseDate'].dt.date == today_date]['SellingPrice'].sum()
                if today_sales_count> 0:
                    dispatcher.utter_message(
                        text=f"The total sales for today ({today_date}) is {today_sales_count} with a sales price of ${yesterday_sales_price:.2f}."
                    )
                else:
                    dispatcher.utter_message(
                        text=f"No sales were recorded today ({today_date})."
                    )
                
                return []

            # Yesterday's sales
            if last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Processing yesterday's sales...{lastday}")
                yesterday_sales_count = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].count()
                yesterday_sales_price = df[df['PurchaseDate'].dt.date == lastday]['SellingPrice'].sum()
                if yesterday_sales_count > 0:
                    dispatcher.utter_message(
                        text=f"The total sales for yesterday ({lastday}) is {yesterday_sales_count} with a sales price of ${yesterday_sales_price:.2f}."
                    )
                else:
                    dispatcher.utter_message(
                        text=f"No sales were recorded yesterday ({lastday})."
                    )
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
                dispatcher.utter_message(
                    text=f"The total sales count for {half_name} of {year} is {half_year_sales_count} and sales price is ${half_year_sales_price:.2f}."
                )
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
                if count_fortnight is not None:
                    if count_fortnight > 0:
                        dispatcher.utter_message(text=f"The total sales for the last fortnight ({start_date_formatted} to {end_date_formatted}) is {count_fortnight} with a sales price of ${price_fortnight:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded for the fortnight ({start_date_formatted} to {end_date_formatted}).")
                else:
                    dispatcher.utter_message(text=price_fortnight)  # Return the error message from the function
                return []

            #Handle last N months request
            if last_n_months:
                logging.info(f"Processing sales data for the last N months...{last_n_months}")
                start_date, end_date, num_months= last_n_months
                # Extract the number of months for display
                # num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)

                count_last_n_months, price_last_n_months = extract_sales_in_date_range(df, start_date, end_date)
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                if count_last_n_months is not None:
                    if count_last_n_months > 0:
                        dispatcher.utter_message(text=f"The total sales in the last  {num_months} months ({start_date_formatted} to {end_date_formatted}) is {count_last_n_months} with a sales price of ${price_last_n_months:.2f}.")
                    else:
                        dispatcher.utter_message(text=f"No sales were recorded in the last N months ({start_date_formatted} to {end_date_formatted}).")
                else:
                    dispatcher.utter_message(text=price_last_n_months)  # Return the error message from the function
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
                    
                    dispatcher.utter_message(
                        text=f"The total sales count for the {quarter_name} of {year} is {quarterly_sales_count} and sales price is ${quarterly_sales_price:.2f}."
                    )
                except Exception as e:
                    dispatcher.utter_message(
                        text=f"An error occurred while processing quarterly sales: {str(e)}"
                    )
                return []

           
  
            if month_year_pairs:
                logging.info(f"sales with month - year... {month_year_pairs}")
                try:
                    total_sales_count = total_sales_price = 0.0
                    for month, year in month_year_pairs:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price

                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales by year: {str(e)}")
                return []

            if years:
                logging.info(f"sales with year...: {years}")
                try:
                    total_sales_count = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].count()
                    total_sales_price = df[df['PurchaseDate'].dt.year.isin(years)]['SellingPrice'].sum()
                    
                    if len(years) == 1:
                        dispatcher.utter_message(text=f"The total sales count for {years[0]} is {total_sales_count} and sales price is ${total_sales_price:.2f}.")
                    else:
                        years_str = ', '.join(map(str, years))
                        dispatcher.utter_message(text=f"Total sales count for {years_str} is {total_sales_count} and total sales price is ${total_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing annual sales: {str(e)}")
                return []

            if months_extracted:
                logging.info(f"month with current year...:  {months_extracted}")
                try:
                    current_year = pd.to_datetime('today').year
                    for month in months_extracted:
                        monthly_sales_count = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].count()
                        monthly_sales_price = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]['SellingPrice'].sum()
                        total_sales_count += monthly_sales_count
                        total_sales_price += monthly_sales_price
                        
                        dispatcher.utter_message(text=f"The total sales count for {months_reverse[month]} {current_year} is {monthly_sales_count} and sales price is ${monthly_sales_price:.2f}.")
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly sales: {str(e)}")
                return []
            logging.info("processing total sales.....")
    
            total_sales_count, total_sales_price = self.calculate_total_sales(df)
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            dispatcher.utter_message(text=f"The overall (from {start_date} to {end_date}) sales count is {total_sales_count} with a sales price of ${total_sales_price:.2f}.")

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred in processing your request: {str(e)}")
        
        return []

    def calculate_total_sales(self, df: pd.DataFrame) -> Tuple[int, float]:
        total_sales_count = df['SellingPrice'].count() if 'SellingPrice' in df.columns else 0
        total_sales_price = df['SellingPrice'].sum() if 'SellingPrice' in df.columns else 0.0
        return total_sales_count, total_sales_price

###############################################COMPARESALES################################################################################
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
    # quarter_pattern = r'\b(?:quarter\s*(1|2|3|4)|q(1|2|3|4)|first|second|third|fourth)\b'
    # # year_pattern = r'\b(20\d{2})\b'
    # quarter_pattern = r'(q(\d)|quarter\s*(\d)|first|second|third|fourth)'
    # year_pattern = r'(\d{4})'
    # comparison_pattern = fr"{quarter_pattern}(?:\s*{year_pattern})?\s*to\s*{quarter_pattern}(?:\s*{year_pattern})?"
    # # comparison_pattern = r"(q(\d)|quarter\s*(\d)|first|second|third|fourth)\s*(\d{4})?\s*to\s*(q(\d)|quarter\s*(\d)|first|second|third|fourth)\s*(\d{4})?"
    # match = re.search(comparison_pattern, text, re.IGNORECASE)

    # if match:
    #     # Extract first quarter, year and second quarter, year
    #     quarter1 = match.group(2) or match.group(3) or match.group(4)
    #     quarter2 = match.group(8) or match.group(9) or match.group(10)
    #     year1 = match.group(5)
    #     year2 = match.group(11)

    #     # Normalize quarter 1 and 2 (convert first/second/third/fourth to 1, 2, 3, 4)
    #     text_to_quarter = {'first': 1, 'second': 2, 'third': 3, 'fourth': 4}
    #     quarter1 = text_to_quarter.get(quarter1.lower(), int(quarter1)) if quarter1 and not quarter1.isdigit() else int(quarter1 or 0)
    #     quarter2 = text_to_quarter.get(quarter2.lower(), int(quarter2)) if quarter2 and not quarter2.isdigit() else int(quarter2 or 0)


    #     # Default year handling
    #     current_year = datetime.now().year
    #     year1 = int(year1) if year1 else current_year
    #     year2 = int(year2) if year2 else year1

    #     # Return the corresponding quarter range
    #     quarters = {
    #         1: (1, 3),   # Q1: January to March
    #         2: (4, 6),   # Q2: April to June
    #         3: (7, 9),   # Q3: July to September
    #         4: (10, 12)  # Q4: October to December
    #     }

    #     return (quarters.get(quarter1), year1, quarters.get(quarter2), year2)

    # return None
   

class ActionCompareSales(Action):
    def name(self) -> str:
        return "action_compare_sales"

    def run(self, dispatcher, tracker, domain) -> list[Dict[Text, Any]]:
        global df
        try:
            logging.info("Running ActionCompareSales...")
            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")
            
            response = {"status": "success", "message": ""}
            # if not user_message:
                # response["status"] = "failure"
                # response["message"] = "I didn't receive any message for comparison. Please specify a time range."
                # dispatcher.utter_message(text=response["message"])
                # return [BotUttered(text=response["message"])]
            if not user_message:
                dispatcher.utter_message(text="I didn't receive any message for comparison. Please specify a time range.")
                return []
            # if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
            #     response["status"] = "failure"
            #     response["message"] = "Required columns ('PurchaseDate', 'SellingPrice') are missing in the dataset."
            #     dispatcher.utter_message(text=response["message"])
            #     return [BotUttered(text=response["message"])]

            if 'PurchaseDate' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns ('PurchaseDate', 'SellingPrice') are missing in the dataset.")
                return []

            month_pattern = r"(\w+ \d{4}) to (\w+ \d{4})"
            year_pattern = r"(\d{4}) to (\d{4})"
            compare_quarters =  extract_quarters_and_years(user_message)
            

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
                    
                    sales_count_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].count()
                    sales_price_1 = df[(df['PurchaseDate'].dt.month == start_date_1.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_1.year)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].count()
                    sales_price_2 = df[(df['PurchaseDate'].dt.month == start_date_2.month) & 
                                        (df['PurchaseDate'].dt.year == start_date_2.year)]['SellingPrice'].sum()

                    table = [
                        [month1, sales_count_1, f"${sales_price_1:.2f}"],
                        [month2, sales_count_2, f"${sales_price_2:.2f}"]
                    ]
                    # table = [
                    #     {"Month": month1, "Total Sales": sales_count_1, "Sales Revenue": f"${sales_price_1:.2f}"},
                    #     {"Month": month2, "Total Sales": sales_count_2, "Sales Revenue": f"${sales_price_2:.2f}"}
                    # ]

                    headers = ["Month", "Total Sales", "Sales Revenue"]
                    comparison_table = tabulate(table, headers, tablefmt="grid")

                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2
                    # if count_difference > 0:
                    #     table.append({"Message": f"Difference in sales: {month1} had {abs(count_difference)} more sales than {month2}."})
                    # elif count_difference < 0:
                    #     table.append({"Message": f"Difference in sales: {month2} had {abs(count_difference)} more sales than {month1}."})
                    # else:
                    #     table.append({"Message": "Both months had the same number of sales."})

                    # if price_difference > 0:
                    #     table.append({"Message": f"Difference in revenue: {month1} generated ${abs(price_difference):.2f} more in sales revenue than {month2}."})
                    # elif price_difference < 0:
                    #     table.append({"Message": f"Difference in revenue: {month2} generated ${abs(price_difference):.2f} more in sales revenue than {month1}."})
                    # else:
                    #     table.append({"Message": "Both months generated the same sales revenue."})


                    if count_difference > 0:
                        comparison_table += f"\n\nDifference in sales: {month1} had {abs(count_difference)} more sales than {month2}.\n"
                    elif count_difference < 0:
                        comparison_table += f"\n\nDifference in sales: {month2} had {abs(count_difference)} more sales than {month1}.\n"
                    else:
                        comparison_table += "\n\nBoth months had the same number of sales.\n"

                    if price_difference > 0:
                        comparison_table += f"\n\nDifference in revenue: {month1} generated ${abs(price_difference):.2f} more in sales revenue than {month2}."
                    elif price_difference < 0:
                        comparison_table += f"\n\nDifference in revenue: {month2} generated ${abs(price_difference):.2f} more in sales revenue than {month1}."
                    else:
                        comparison_table += "\n\nBoth months generated the same sales revenue."

                    # response["message"] = comparison_table
                    dispatcher.utter_message(text=comparison_table)
                except ValueError as ve:
                    # response["status"] = "failure"
                    # response["message"] = "Please provide a valid comparison in the format 'month year to month year' or 'year to year' or 'quarter to quarter'."
                    # dispatcher.utter_message(text=response["message"])
                    # return [BotUttered(text=response["message"])]
                    dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year' or 'quarter to quarter'.")
                    logging.error(f"Date parsing error for month comparison: {ve}")
                

            # Check for year comparison
            year_matches = re.findall(year_pattern, user_message)
            if year_matches:
                logging.info("year comparison...")
                try:
                    year1, year2 = year_matches[0]
                    sales_count_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].count()
                    sales_price_1 = df[df['PurchaseDate'].dt.year == int(year1)]['SellingPrice'].sum()
                    
                    sales_count_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].count()
                    sales_price_2 = df[df['PurchaseDate'].dt.year == int(year2)]['SellingPrice'].sum()

                    table = [
                        [year1, sales_count_1, f"${sales_price_1:.2f}"],
                        [year2, sales_count_2, f"${sales_price_2:.2f}"]
                    ]

                    headers = ["Year", "Total Sales", "Total Revenue"]
                    comparison_table = tabulate(table, headers, tablefmt="grid")

                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2

                    if count_difference > 0:
                        comparison_table += f"\n\nSales Count Difference: {year1} had {abs(count_difference)} more sales than {year2}.\n"
                    elif count_difference < 0:
                        comparison_table += f"\n\nSales Count Difference: {year2} had {abs(count_difference)} more sales than {year1}.\n"
                    else:
                        comparison_table += "\n\nBoth years had the same number of sales."

                    if price_difference > 0:
                        comparison_table += f"\n\nRevenue Difference: {year1} generated ${abs(price_difference):.2f} more in sales revenue than {year2}."
                    elif price_difference < 0:
                        comparison_table += f"\n\nRevenue Difference: {year2} generated ${abs(price_difference):.2f} more in sales revenue than {year1}."
                    else:
                        comparison_table += "\n\nBoth years generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_table)
                except ValueError as ve:
                    dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year' or 'quarter to quarter.")
                    logging.error(f"Date parsing error for year comparison: {ve}")
                return []
            
            if compare_quarters:
                logging.info("quarter comparison...")
                (start_month_1, end_month_1), year1, (start_month_2, end_month_2), year2 = compare_quarters
                
                try:
                    # year1 = int(year1) if year1 else datetime.now().year
                    # year2 = int(year2) if year2 else year 
                    
                    # quarter_map = {'1': 1, '2': 4, '3': 7, '4': 10}  # Starting month of each quarter

                    # start_month_1 = quarter_map[quarter1]
                    # end_month_1 = start_month_1 + 2

                    sales_count_1 = df[(df['PurchaseDate'].dt.year == year1) & 
                                        (df['PurchaseDate'].dt.month >= start_month_1) & 
                                        (df['PurchaseDate'].dt.month <= end_month_1)]['SellingPrice'].count()
                    sales_price_1 = df[(df['PurchaseDate'].dt.year == year1) & 
                                        (df['PurchaseDate'].dt.month >= start_month_1) & 
                                        (df['PurchaseDate'].dt.month <= end_month_1)]['SellingPrice'].sum()
                    sales_count_2 = df[(df['PurchaseDate'].dt.year == year2) & 
                               (df['PurchaseDate'].dt.month >= start_month_2) & 
                               (df['PurchaseDate'].dt.month <= end_month_2)]['SellingPrice'].count()
                    sales_price_2 = df[(df['PurchaseDate'].dt.year == year2) & 
                                       (df['PurchaseDate'].dt.month >= start_month_2) & 
                                       (df['PurchaseDate'].dt.month <= end_month_2)]['SellingPrice'].sum()
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
                    comparison_table = tabulate(table, headers, tablefmt="grid")
                    count_difference = sales_count_1 - sales_count_2
                    price_difference = sales_price_1 - sales_price_2
                    if count_difference > 0:
                        comparison_table += f"\n\nSales Count Difference: {quarter_name_1} of {year1} had {abs(count_difference)} more sales than {quarter_name_2} of {year2}.\n"
                    elif count_difference < 0:
                        comparison_table += f"\n\nSales Count Difference: {quarter_name_2} of {year2} had {abs(count_difference)} more sales than {quarter_name_1} of {year1}.\n"
                    else:
                        comparison_table += "\n\nBoth years had the same number of sales."
                    if price_difference > 0:
                        comparison_table += f"\n\nRevenue Difference: {quarter_name_1} of {year1} generated ${abs(price_difference):.2f} more in sales revenue than {quarter_name_2} of {year2}."
                    elif price_difference < 0:
                        comparison_table += f"\n\nRevenue Difference: {quarter_name_2} of {year2} generated ${abs(price_difference):.2f} more in sales revenue than {quarter_name_1} of {year1}."
                    else:
                        comparison_table += "\n\nBoth years generated the same sales revenue."

                    dispatcher.utter_message(text=comparison_table)
                except Exception as e:
                    dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
                    logging.error(f"Unexpected error in ActionCompareSales for quarter comparison: {e}")
                return []


            dispatcher.utter_message(text="Please provide a valid comparison in the format 'month year to month year' or 'year to year','quarter to quarter'..")
        
        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            logging.error(f"Unexpected error in ActionCompareSales: {e}")
        
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
                dispatcher.utter_message(text="Sales data could not be retrieved from the database. Please try again later.")
                logging.error("Sales data is empty after fetching from the database.")
                return []

            user_message = tracker.latest_message.get('text')
            logging.info(f"User message: {user_message}")

            

            # Extract country from the user message
            country_extracted = extract_country_from_text(user_message)
            logging.info(f"Initial extracted country: {country_extracted}")
    
        
            if not country_extracted:
                dispatcher.utter_message(text=f"Sorry, we do not have sales data for {country_extracted}. Please provide another country.")
                logging.info(f"Country {country_extracted} not found in the dataset.")
                return []
            country = country_extracted[0]

            # Extract years, months, and month-year pairs from the user message
            try:
                years = extract_years(user_message)
                months_extracted = extract_months(user_message)
                month_year_pairs = extract_month_year(user_message)
                quarterly = extract_quarters_from_text(user_message)
                half_year = extract_half_year_from_text(user_message)
                fortnight = extract_fortnight(user_message)
                last_n_months = extract_last_n_months(user_message)
                today = extract_today(user_message)
                last_day = extract_last_day(user_message)
                date = extract_date(user_message)
                # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            except Exception as e:
                dispatcher.utter_message(text="There was an error extracting dates from your message. Please try specifying the month and year clearly.")
                logging.error(f"Date extraction error: {e}")
                return []

            logging.info(f"Processing sales data for country: {country.upper()}")
            table_data = []
            response_message = ""

            try:
                if date:
                    logging.info(f"country sales specific date...{date}...")
    
                    try:
                        daily_sales_count, daily_sales_price = calculate_country_sales_for_specific_date(df, country, date)
                        
                        
                        if daily_sales_count == 0:
                            logging.info(f"No sales found for {country.upper()} on {date}")
                            return 0, 0.0
                        else:
                            response_message = (
                                f"Sales data for {country.upper()} on {date}: "
                                f"{daily_sales_count} sales, generating a total revenue of ${daily_sales_price:,.2f}."
                            )
                    except Exception as e:
                        logging.error(f"Error calculating sales for {country.upper()} on {date}: {e}")
                        response_message = f"Sorry, there was an error retrieving the sales data for {country.upper()} on {date}. Please try again later."
                # Use helper functions to calculate sales count and revenue for the country
                
                elif month_year_pairs:
                    logging.info(f" country sales month year pairs....{month_year_pairs}")
                    for month, year in month_year_pairs:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, year)
                            if sales_count == 0:
                                response_message += f"In {months_reverse[month].capitalize()} {year}, {country.upper()} had no sales.\n"
                            else:
                                 response_message += f"In {months_reverse[month].capitalize()} {year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"         
                        except Exception as e:
                            logging.error(f"Error calculating sales for {month}/{year} in {country.upper()}: {e}")
                            response_message += f"Error calculating sales for {months_reverse[month].capitalize()} {year} in {country.upper()}. Please try again.\n"
                
                elif years:
                    logging.info(f" country sales years....{years}")
                    for year in years:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_year(df, country, year)
                            response_message += f"In {year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                        except Exception as e:
                            logging.error(f"Error calculating sales for year {year} in {country.upper()}: {e}")
                            response_message += f"Error calculating sales for {year} in {country.upper()}. Please try again.\n"

                elif months_extracted:
                    current_year = datetime.now().year
                    logging.info(f" country sales month with current year....{months_extracted}")
                    for month in months_extracted:
                        try:
                            sales_count, total_revenue = calculate_country_sales_by_month_year(df, country, month, current_year)
                            if sales_count == 0:  # No sales in this month
                                response_message += f"In {months_reverse[month].capitalize()} {current_year}, {country.upper()} had no sales.\n"
                            else:
                                response_message += f"In {months_reverse[month].capitalize()} {current_year}, {country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                        except Exception as e:
                            logging.error(f"Error calculating sales for {months_reverse[month].capitalize()} {current_year} in {country.upper}: {e}")
                            response_message += f"Error calculating sales for {months_reverse[month].capitalize()} {current_year} in {country.upper}. Please try again.\n"

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
                            response_message += (
                                f"In {quarter_name} of {year}, {country} recorded "
                                f"{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                            )
                        else:
                            response_message += f"No sales data found for {quarter_name} of {year} in {country.upper()}.\n"

                            
                    except Exception as e:
                        logging.error(f"Error calculating sales for quarter {quarter_name} in {country.upper()}: {e}")
                        response_message += f"Error calculating sales for quarter{quarter_name} in {country.upper()}. Please try again.\n"

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
                        response_message += f"In the {half_name} of {year}, {country.upper()} recorded {half_year_sales_count} sales, generating a total revenue of ${half_year_sales_price:,.2f}.\n"
                    except Exception as e:
                        logging.error(f"Error calculating sales for {half_name} in {country.upper()}: {e}")
                        response_message += f"Error calculating sales for {half_name} of {year} in {country.upper()}. Please try again.\n"

                elif fortnight:
                    start_date, end_date = fortnight
                    logging.info(f"Processing fortnight data for {country.upper()}: {fortnight}")
                    response_message = ""
                    try:
                        # Log the current fortnight being processed
                        logging.info(f"Processing fortnight from {start_date} to {end_date} for {country.upper()}.")
                        
                        # Calculate sales for the country in the given fortnight
                        sales_count, total_revenue = calculate_country_sales_by_fortnight(df, country, start_date, end_date)
                        
                        if sales_count > 0 or total_revenue > 0:
                            # Format dates for user-friendly display
                            start_date_formatted = start_date.date()
                            end_date_formatted = end_date.date()
                            
                            response_message += (
                                f"In the fortnight from ({start_date_formatted} to {end_date_formatted}) of {datetime.now().year}, "
                                f"{country.upper()} recorded {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}.\n"
                            )
                        else:
                            response_message += (
                                f"In the fortnight from ({start_date_formatted} to {end_date_formatted}) of {datetime.now().year}, "
                                f"no sales were recorded for {country.upper()}.\n"
                            )
                    except Exception as e:
                        logging.error(f"Error calculating sales for fortnight from {start_date} to {end_date} in {country.upper()}: {e}")
                        response_message += f"Error calculating sales for fortnight from {start_date} to {end_date} in {country.upper()}. Please try again.\n"
                
                elif last_n_months:
                    logging.info(f" country sales last n months....{last_n_months}")
                    start_date, end_date, num_months = last_n_months
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()

                    
                    try:
                        n_month_sales = df[(df['countryname'].str.lower() == country.lower()) & (df['PurchaseDate'] >= start_date) & (df['PurchaseDate'] <= end_date)]
        
                        sales_count = n_month_sales['SellingPrice'].count()
                        sales_price = n_month_sales['SellingPrice'].sum()
                        
                        response_message += f"In the last {num_months} months ({start_date_formatted} to {end_date_formatted}), {country.upper()} recorded {sales_count} sales, generating a total revenue of ${sales_price:,.2f}.\n"
                    except Exception as e:
                        logging.error(f"Error calculating sales for last {num_months} months ({start_date_formatted} to {end_date_formatted}) in {country.upper()}: {e}")
                        response_message += f"Error calculating sales for last {num_months} months ({start_date_formatted} to {end_date_formatted}) in {country.upper()}. Please try again.\n"

                elif today:
                    today_date = datetime.now().date()
                    logging.info(f" country sales today....{today_date}")
                    try:
                        sales_count, total_revenue = calculate_country_sales_for_today(df, country, today)
                        if sales_count != 0 and total_revenue != 0.0:
                            response_message = (
                                f"Today's {today_date} sales data for {country.upper()}: "
                                f"{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                            )
                            
                        else:
                            response_message = (
                                f"No sales data found for {country.upper()} on {today_date}."
                            )

                    except Exception as e:
                        logging.error(f"Error calculating today's sales for {country.upper()}: {e}")
                        response_message = f"Error calculating today's sales data for {country.upper()}. Please try again later."

                elif last_day:
                    lastday = (datetime.now() - timedelta(days=1)).date()
                    logging.info(f" country sales last day....{lastday}")
                    try:
                        sales_count, total_revenue = calculate_country_sales_for_last_day(df, country, last_day)
                        if sales_count != 0 and total_revenue != 0.0:
                            response_message = (
                                f"last day's {lastday} sales data for {country.upper()}: "
                                f"{sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                            )
                        else:
                            response_message = (
                                f"No sales data found for {country.upper()} on {lastday}."
                            )
                    except Exception as e:
                        logging.error(f"Error calculating last day's sales for {country.upper()}: {e}")
                        response_message = f"Error calculating last day's sales data for {country.upper()}. Please try again later."

                else:
                    # If no specific month or year, return total sales for the country
                    logging.info("total country sales....")
                    try:
                        # Attempt to calculate total sales for the country
                        sales_count, total_revenue = calculate_country_sales(df, country)
                        start_date = df['PurchaseDate'].min().date()
                        end_date = df['PurchaseDate'].max().date()
                        response_message = f" (from {start_date} to {end_date}) In {country.upper()}, there have been a total of {sales_count} sales, generating a total revenue of ${total_revenue:,.2f}."
                    except Exception as e:
                        logging.error(f"Error calculating total sales for {country.upper()}: {e}")
                        response_message = f"Sorry, there was an error retrieving the total sales data for {country.upper()}. Please try again later."
            except Exception as e:
                dispatcher.utter_message(text="An error occurred while calculating sales data. Please try again.")
                logging.error(f"Sales calculation error for country {country.upper()}: {e}")
                return []

            dispatcher.utter_message(text=response_message)
            return []

        except Exception as e:
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again later.")
            logging.error(f"Error fetching or processing sales data: {e}")
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
                # response = {
                #     "status": "error",
                #     "message": "Sorry, I couldn't understand your message. Please try again."
                # }
                # dispatcher.utter_message(json_message=response)
                dispatcher.utter_message(text="Sorry, I couldn't understand your message. Please try again.")
                logging.warning("Received empty or None message from user.")
                return []

            logging.info(f"Received user message: {user_message}")
            country_extracted = extract_country_from_text(user_message)
            #countries_extracted = extract_country_from_text(user_message, df)
            planname = next(tracker.get_latest_entity_values('planname'), None)
            
            if not country_extracted:
                dispatcher.utter_message(text="❌ No valid country found in your query. Please specify a country.")
                return []
    
            # country_names = df['countryname'].dropna().unique().tolist()
            # country = next((c for c in country_names if c.lower() == country_extracted.lower()), None)
            # Check for a valid country input
            # if not country_extracted:
            #     # dispatcher.utter_message(text=f"Sorry, we do not have data for {country_extracted}. Please try another country.")
            #     response = {
            #         "status": "error",
            #         "message": f"Sorry, we do not have data for {country_extracted}. Please try another country."
            #     }
            #     dispatcher.utter_message(json_message=response)
            #     logging.warning(f"Country {country_extracted} not found in the dataset.")
            #     return []
            
            country = country_extracted[0]
            
            
            

            # Extract plans for the specified country
            country_plans = df[df['countryname'].str.lower() == country.lower()]['PlanName'].unique()
            if len(country_plans) == 0:
                # response = {
                #     "status": "error",
                #     "message": f"No plans available for {country.upper()}."
                # }
                # dispatcher.utter_message(json_message=response)
                # return []
                dispatcher.utter_message(text=f"No plans available for {country.upper()}.")
                logging.info(f"No plans found for country: {country.upper()}")
                return []
            # Extract years, months, and month-year pairs from user message
            years = extract_years(user_message)
            months_extracted = extract_months(user_message)
            month_year_pairs = extract_month_year(user_message)
            quarterly = extract_quarters_from_text(user_message)
            half_year = extract_half_year_from_text(user_message)
            fortnight = extract_fortnight(user_message)
            last_n_months = extract_last_n_months(user_message)
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            date = extract_date(user_message)
            #specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
            
            logging.info(f"Processing sales data for country: {country.upper()} and plans: {planname}")
            def format_sales_table(data):
                headers = ["S.No", "Planname", "Sales Count", "Total Revenue"]
                table = [[i + 1, plan, sales, f"${revenue:,.2f}"] for i, (plan, sales, revenue) in enumerate(data)]
                return tabulate(table, headers=headers, tablefmt="grid")
            response_data = []
            response_message = ""
            
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

                        response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {year} ({country} Plans):\n\n"
                        response_message += format_sales_table(response_data) + "\n\n"
                        response_data.clear()  # Clear data for the next iteration
                    else:
                        response_message += f"No sales data found for {months_reverse.get(month, month).capitalize()} {year}.\n\n"


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

                        response_message += f"📅 Sales Overview for {year} ({country} Plans):\n\n"
                        response_message += format_sales_table(response_data) + "\n\n"
                        response_data.clear()
                            
                    else:
                        response_message += f"No sales data found for {years} ({country} Plans):\n\n"

                        

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

                        response_message += f"📅 Sales Overview for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"
                        response_message += format_sales_table(response_data) + "\n\n"
                        response_data.clear()  # Clear data for the next iteration
                    else:
                        response_message += f"No sales data found for {months_reverse.get(month, month).capitalize()} {current_year} ({country} Plans):\n\n"

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

                    response_message += f"📅 Sales Overview for {quarter_name} {year} ({country}):\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found for {quarter_name} of {year} in {country} plans.\n"



            elif date:
                logging.info(f"Processing data for {country} on {date}.")
                for plan in country_plans:
                    daily_sales_count ,daily_sales_price =  calculate_plannane_sales_for_specific_date(df, planname, date)
                    if daily_sales_count > 0 and daily_sales_price > 0:
                        response_data.append((plan, daily_sales_count, daily_sales_price))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message += f"📅 Sales Overview for {date} ({country} Plans):\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found on {date} ({country} Plans):\n\n"


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
                    half_year_sales_count = half_year_sales['SellingPrice'].count()
                    half_year_sales_price = half_year_sales['SellingPrice'].sum()
        
                    # Determine the half-year name
                    
                    
                    # Format the response message
                    response_data = [
                        (row['PlanName'], row['sales_count'], row['total_revenue'])
                        for _, row in grouped_sales.iterrows()
                    ]
                    half_name = "First Half" if start_month == 1 else "Second Half"
                    if response_data:
                        response_data.sort(key=lambda x: x[1], reverse=True)

                        response_message += f"📅 Sales Overview for {half_name} of {year}:\n"
                        response_message += format_fortnight_sales_table(response_data) + "\n\n"
                    else:
                        response_message += f"No sales data found for the {half_name} of {year}\n\n"
                         
                        
                except Exception as e:
                    logging.error(f"Error calculating sales for half-year {half_name} in {country.upper()}: {e}")
                    response_message += (
                        f"Error calculating sales for the period in {country.upper()}. "
                        "Please try again later.\n"
                    )
                



            elif fortnight:
                logging.info(f"Processing fortnight {fortnight} data for {country}.")
                start_date, end_date = fortnight
                for start_date, end_date in fortnight:
                    for plan in country_plans:
                        try:
                            sales_count, total_revenue = calculate_planname_sales_for_fortnight(df, plan, start_date, end_date)
                            if sales_count > 0 or total_revenue > 0:
                                start_date_formatted = start_date.date()
                                end_date_formatted = end_date.date()
                                response_message +=  f"In the fortnight from {start_date_formatted} to {end_date_formatted} "
                        except Exception as e:
                            logging.error(f"Error calculating sales for fortnight {start_date_formatted} to {end_date_formatted} in {country}: {e}")
                            response_message += f"Error calculating sales for fortnight {start_date_formatted} to {end_date_formatted} in {country}. Please try again.\n"
                    if response_data:
                        response_data.sort(key=lambda x: x[1], reverse=True)

                        response_message += f"📅 Sales Overview for Fortnight ({start_date_formatted} to {end_date_formatted}) ({country} Plans):\n\n"
                        response_message += format_fortnight_sales_table(response_data) + "\n\n"
                    else:
                        response_message += f"No sales data found for the period {start_date_formatted} to {end_date_formatted}.\n\n"



            elif last_n_months:
                logging.info(f"Processing last N months {last_n_months} data for {country}.")
                start_date, end_date, num_months = last_n_months
        
               
                for plan in country_plans:
                    filtered_df = df[
                        (df['PlanName'] == planname) &
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]
                    sales_count = filtered_df['SellingPrice'].count
                    total_revenue = filtered_df['SellingPrice'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message += f"📅 Sales Overview for Last {num_months} Months ({start_date_formatted} to {end_date_formatted}){country} Plans :\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found for Last {num_months} Months ({start_date_formatted} to {end_date_formatted}){country} Plans :\n\n"
                        


            elif today:
                today_date = datetime.now().date()
                logging.info(f"Processing today's {today_date} data for {country}.")
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_by_today(df, plan, today)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message += f"📅 Sales Overview for Today ({today_date}) ({country} Plans):\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found for Today ({today_date}) ({country} Plans):\n\n"
                            


            elif last_day:
                last_date = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"Processing last day's {last_date} data for {country}.")
                for plan in country_plans:
                    sales_count, total_revenue = calculate_planname_sales_by_last_day(df, plan, last_day)
                    if sales_count > 0 or total_revenue > 0:
                        response_data.append((plan, sales_count, total_revenue))
                if response_data:
                    response_data.sort(key=lambda x: x[1], reverse=True)

                    response_message += f"📅 Sales Overview for Last Day ({last_date}) ({country} Plans):\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found for Last Day ({last_date}) ({country} Plans):\n\n"
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

                    response_message += f"📊 Total Sales Overview for {country} Plans (from {start_date} to {end_date}):\n\n"
                    response_message += format_sales_table(response_data) + "\n\n"
                    response_data.clear()  # Clear data for the next iteration
                else:
                    response_message += f"No sales data found for ({country} Plans):\n\n"

            # Check if no data was found
            if not response_message.strip():
                dispatcher.utter_message(text=f"No sales data found for the specified criteria in {country}.")
                logging.info("No sales data found for specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except KeyError as e:
            logging.error(f"KeyError: {e}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again.")
        except ValueError as e:
            logging.error(f"ValueError: {e}")
            dispatcher.utter_message(text="There was an issue with the input data format. Please specify the dates correctly.")
        except Exception as e:
            logging.error(f"Unexpected error: {e}")
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
        
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
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive plans
            active_plans_current_year = df[df['PurchaseDate'].dt.year == current_year]['PlanName'].unique()
            past_plans = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['PlanName'].unique()
            active_plans = list(active_plans_current_year)
            inactive_plans = list(set(past_plans) - set(active_plans))
            plans_count = len(active_plans)
            inactive_plans_count = len(inactive_plans)

            # Generate response
            if plans_count == 0 and inactive_plans_count == 0:
                dispatcher.utter_message(text="No active or inactive plans found for the specified criteria.")
                logging.info("No plans found for the specified criteria.")
                return []

            response = f"Total Active Plans: {plans_count}\n\n"
            response += "HERE ARE THE ACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in active_plans) + "\n\n"
            response += f"Total Inactive Plans: {inactive_plans_count}\n\n"
            response += "HERE ARE THE INACTIVE PLANS:\n" + "\n".join(f"- {plan}" for plan in inactive_plans)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactivePlans: {e}")
        
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
                dispatcher.utter_message(text="Error parsing purchase dates. Please check the date format.")
                logging.error(f"Date parsing error: {e}")
                return []

            # Get the current year dynamically
            try:
                current_year = datetime.now().year
            except Exception as e:
                dispatcher.utter_message(text="Could not retrieve the current year.")
                logging.error(f"Error fetching current year: {e}")
                return []

            # Identify active and inactive countries
            active_countries_current_year = df[df['PurchaseDate'].dt.year == current_year]['countryname'].unique()
            past_countries = df[df['PurchaseDate'].dt.year.isin([2021, 2022, 2023])]['countryname'].unique()
            active_countries = list(active_countries_current_year)
            inactive_countries = list(set(past_countries) - set(active_countries))
            countries_count = len(active_countries)
            inactive_countries_count = len(inactive_countries)

            # Generate response
            if countries_count == 0 and inactive_countries_count == 0:
                dispatcher.utter_message(text="No active or inactive countries found for the specified criteria.")
                logging.info("No countries found for the specified criteria.")
                return []

            response = f"Total Active Countries: {countries_count}\n\n"
            response += "HERE ARE THE ACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in active_countries) + "\n\n"
            response += f"Total Inactive Countries: {inactive_countries_count}\n\n"
            response += "HERE ARE THE INACTIVE COUNTRIES:\n" + "\n".join(f"- {country}" for country in inactive_countries)

            dispatcher.utter_message(text=response)
        
        except Exception as e:
            dispatcher.utter_message(text="An unexpected error occurred. Please try again later.")
            logging.error(f"Unexpected error in ActionGetActiveAndInactiveCountries: {e}")
        
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
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)

        logging.info(f"Processing top plans sales data based on user request: {user_message}")
        table_data = []
        response_message = ""
        try:
            # Determine the response based on user input
            if month_year_pairs:
                logging.info(f"top plans for month year :{month_year_pairs}")

                for month, year in month_year_pairs:
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No sales data available for the top plans in the {months_reverse[month]} {year} .\n"
                    else:
                        response_message += f"📈 Top 10 Plans for {months_reverse[month]} {year} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                        

            elif years:
                logging.info(f"top plans for year {years}")
                for year in years:
                    top_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No sales data available for the top plans in the {years}.\n"
                    else:
                        response_message += f"📈 Top 10 Plans for {years} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")



            elif months_extracted:
                logging.info(f"top plans for month with current year :{months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    top_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nlargest(10, 'total_sales').reset_index()
                    if top_plans.empty:
                        response_message += f"No sales data available for the top plans in the {months_reverse[month]} {current_year}.\n"
                    else:
                        response_message += f"📈 Top 10 Plans for {months_reverse[month]} {current_year} :\n"
                        for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                        

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
                    ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans in the {quarter_name} {year}.\n"
                else:
                    response_message += f"📈 Top 10 Plans for {quarter_name} {year}:\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

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
                    ).nlargest(10, 'total_sales').reset_index()
                half_name = "First Half" if start_month == 1 else "Second Half"
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans in the {half_name} of {year} .\n"
                else:
                    response_message += f"📈 Top 10 Plans for {half_name} of {year} :\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

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
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_message += f"📈 Top 10 Plans for last fortnight ({start_date_formatted} to {end_date_formatted}) :\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"top plans for last_day :{lastday}")
                top_plans = df[(df['PurchaseDate']==lastday)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans in the last day {lastday}.\n"
                else:
                    response_message += f"📈 Top 10 Plans for last day ({lastday}).\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
            elif today:
                today_date = datetime.now().date()

                logging.info(f"top plans for last_day :{today_date}")
                top_plans = df[(df['PurchaseDate']==today_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans on {today_date}.\n"
                else:
                    response_message += f"📈 Top 10 Plans for today ({today_date}).\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
                    
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
                    .nlargest(10, 'total_sales')
                    .reset_index()
                )
            
                if top_plans.empty:
                    dispatcher.utter_message(
                        text=f"No sales data available for the top plans on {date}."
                    )
                else:
                    response_message = f"📈 Top 10 Plans for the specific date {date}:\n"
                    table_data = []
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No', 'Plan Name', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

            elif last_n_months:
                logging.info(f"top plans for last n months :{last_n_months}")
                start_date, end_date, num_months = last_n_months
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                top_plans = df[(df['PurchaseDate']>=start_date)&(df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                if top_plans.empty:
                    response_message += f"No sales data available for the top plans in last {last_n_months} months.\n"
                else:
                    response_message += f"📈 Top 10 Plans for last {last_n_months} months.\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

            
            else:
                logging.info("top plans overall")
                top_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nlargest(10, 'total_sales').reset_index()
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if top_plans.empty:
                    response_message += "No sales data available for the top plans.\n"
                else:
                    response_message += f"📈 Top 10 Highest Sales Plans Overall (from {start_date} to {end_date}):\n\n"
                    for idx, row in top_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    


        except Exception as e:
            logging.error(f"Error while processing top sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []

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
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        # specific_date_text =  next(tracker.get_latest_entity_values("specific_date"), None)
        table_data = []
        response_message = ""
        try:
            if month_year_pairs:
                logging.info(f"top plans for month year :{month_year_pairs}")

                for month, year in month_year_pairs:
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No sales data available for the top lowest plans in {months_reverse[month]} {year} .\n"
                    else:
                        response_message += f"📉 Top lowest 10 Plans for {months_reverse[month]} {year} :\n"
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

            elif years:
                logging.info(f"top plans for year :{years}")
                for year in years:
                    lowest_plans = df[df['PurchaseDate'].dt.year == year].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No sales data available for the top lowest plans in {years} .\n"
                    else:
                        response_message += f"📉 Top lowest 10 Plans for {years} :\n"
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                       
            elif months_extracted:
                logging.info(f"top plans for month with current year :{months_extracted}")
                current_year = datetime.now().year
                for month in months_extracted:
                    lowest_plans = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('PlanName').agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).nsmallest(10, 'total_sales').reset_index()
                    if lowest_plans.empty:
                        response_message += f"No sales data available for the top lowest plans in {months_reverse[month]}.\n"
                    else:
                        response_message += f"📉 Top lowest 10 Plans for {months_reverse[month]} :\n"
                        for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                            table_data.append([idx + 1] + row.tolist())
                        table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                        response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                        
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
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
                else:
                    response_message += f"📉 Top Lowest 10 Plans for last fortnight ({start_date_formatted} to {end_date_formatted}) :\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
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
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans in the {quarter_name} {year}.\n"
                else:
                    response_message += f"📉 Top 10 Plans for {quarter_name} {year}:\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
            elif last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"top lowest plans for last_day :{lastday}")
                lowest_plans = df[(df['PurchaseDate']==lastday)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans in the last day {lastday}.\n"
                else:
                    response_message += f"📉 Top Lowest 10 Plans for last day ({lastday}).\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
    
            elif today:
                today_date = datetime.now().date()

                logging.info(f"top lowest plans for last_day :{today_date}")
                lowest_plans = df[(df['PurchaseDate']==today_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans on {today_date}.\n"
                else:
                    response_message += f"📉 Top Lowest 10 Plans for today ({today_date}).\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
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
                    .nsmallest(10, 'total_sales')
                    .reset_index()
                )
            
                if lowest_plans.empty:
                    dispatcher.utter_message(
                        text=f"No sales data available for the top lowest plans on {date}."
                    )
                else:
                    response_message = f"📉  Top lowest 10 Plans for the specific date {date}:\n"
                    table_data = []
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No', 'Plan Name', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

            elif last_n_months:
                logging.info(f"top lowest plans for last n months :{last_n_months}")
                start_date, end_date,num_months = last_n_months
                lowest_plans = df[(df['PurchaseDate']>=start_date)&(df['PurchaseDate']<=end_date)].groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                start_date_formatted = start_date.date()
                end_date_formatted = end_date.date()
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans in last {num_months} months.\n"
                else:
                    response_message += f"📉  Top Lowest 10 Plans for last {num_months} months.\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
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
                    ).nsmallest(10, 'total_sales').reset_index()
                half_name = "First Half" if start_month == 1 else "Second Half"
                if lowest_plans.empty:
                    response_message += f"No sales data available for the top lowest plans in the {half_name} of {year} .\n"
                else:
                    response_message += f"📉 Top lowest 10 Plans for {half_name} of {year} :\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
            else:
                lowest_plans = df.groupby('PlanName').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).nsmallest(10, 'total_sales').reset_index()
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                if lowest_plans.empty:
                    response_message += "No data available for the top lowest plans overall."
                else:
                    response_message += f"📉 Top lowest 10 Plans for overall (from {start_date} to {end_date}).\n"
                    for idx, row in lowest_plans[['PlanName', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Planname', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

        except Exception as e:
            logging.error(f"Error while processing lowest sales plans: {e}")
            dispatcher.utter_message(text="An error occurred while processing your request. Please try again.")
            return []

        dispatcher.utter_message(text=response_message if response_message.strip() else "No sales data found for the specified criteria.")
        return []



        
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
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        # specific_date_text = next(tracker.get_latest_entity_values("specific_date"), None)

        table_data = []
        response_message = ""

        # If month and year are provided, show results for that specific month/year
        if month_year_pairs:
            logging.info(f"top highest sales by country for month year pairs:{month_year_pairs}")
            for month, year in month_year_pairs:
                top_sales  = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for top highest sales by country for {months_reverse[month]} {year}.")
                else:
                    response_message += f"📈 Top 10  Highest Sales Country for {months_reverse[month]} {year}:\n"
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")

        # If only year is provided, show results for the entire year
        elif years:
            logging.info(f"top highest sales by country for years:{years}")
            for year in years:
                top_sales  = df[df['PurchaseDate'].dt.year == year].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for top highest sales by country for {years}.")
                else:
                    response_message += f"📈 top highest sales by country for {years}:\n"
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
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
            total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            if top_sales.empty:
                response_message += f"No sales data found for top highest sales by country for  {quarter_name} {year}."
            else:            
                response_message += f"📈 top highest sales by country for month  for {quarter_name} {year}:\n"
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
            
        # If only month is provided, show results for that month in the current year
        elif months_extracted:
            logging.info(f"top highest sales by country for month extracted:{months_extracted}")
            current_year = datetime.now().year
            for month in months_extracted:
                top_sales  = df[(df['PurchaseDate'].dt.month==month) &(df['PurchaseDate'].dt.year == current_year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for top highest sales by country for {months_reverse[month]} {current_year}.")
                else:    
                    response_message += f"📈top highest sales by country for  {months_reverse[month]} {current_year}:\n"
                    for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
        elif last_n_months:
            logging.info(f"top highest sales by country for last {last_n_months} months.")
            start_date, end_date,num_months = last_n_months
            top_sales  = df[
                (df['PurchaseDate']>= start_date) &
                (df['PurchaseDate'] <= end_date)
            ].groupby('countryname').agg(
            total_sales=('SellingPrice', 'count'),
            total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            start_date_formatted = start_date.date()
            end_date_formatted = end_date.date()
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for top highest sales by country for month  for the last {num_months} months ({start_date_formatted} to {end_date_formatted}).")
                return []
            else:
                response_message += f"📈 top highest sales by country for month for the Last {num_months} Months ({start_date_formatted} to {end_date_formatted}):\n"
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
            
                
        elif half_year:
            logging.info(f"top highest sales by country for half year: {half_year}")
            year,(start_month, end_month) = half_year
            top_sales  = df[
                (df['PurchaseDate'].dt.month >= start_month) &
                (df['PurchaseDate'].dt.month <= end_month) &
                (df['PurchaseDate'].dt.year == year)
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            half_name = "First Half" if start_month == 1 else "Second Half"
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for top highest sales by country for month  for half-year {half_name} of {year}.")
                return []
            
            table_data = []
            for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                table_data.append([idx + 1] + row.tolist())
            table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
            response_message += f"📈 top highest sales by country for month for {half_name} of {year}:\n"
            response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
           

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
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for top highest sales by country for month for the last fortnight ({start_date_formatted} to {end_date_formatted}).")
                return []

            response_message += f"📈 top highest sales by country for month for the Last Fortnight ({start_date_formatted} to {end_date_formatted}) :\n"
            for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                table_data.append([idx + 1] + row.tolist())
            table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
            response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
            
        elif today:
            today_date = datetime.now().date()
            logging.info(f"top highest sales by country for today...{today_date}")
            top_sales  = df[
                df[df['PurchaseDate'].dt.date == today_date]
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for top highest sales by country for month for today {today_date}.")
                return []
            else:
                response_message += f"📈 top highest sales by country for month for Today ({today_date}):\n"
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")

        elif last_day:
            lastday = (datetime.now() - timedelta(days=1)).date()
            logging.info(f"top highest sales by country for the last day.{lastday}")
            top_sales  = df[
                df[df['PurchaseDate'].dt.date == lastday]
            ].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for top highest sales by country for the last day {lastday}.")
                return []
            else:
                response_message += f"📈 top highest sales by country for lastday ({lastday}) :\n"
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")

        
        elif date:
            logging.info(f"top highest sales by country for for specific date: {date}")
            try:
                top_sales = (
                    df[df['PurchaseDate'].dt.date == date]
                    .groupby('PlanName')
                    .agg(
                        total_sales=('SellingPrice', 'count'),
                        total_revenue=('SellingPrice', 'sum')
                    ).sort_values('total_sales', ascending=False).head(10).reset_index()
                )
         
                if top_sales.empty:
                    dispatcher.utter_message(text=f"No sales data found for top highest sales by country for {date}.")
                    return []

                response_message += f"📈 Top 10 Country sales on {date}:\n"
                for idx, row in top_sales[['countryname', 'SalesCount', 'TotalRevenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
            except Exception as e:
                dispatcher.utter_message(text=f"Error processing the specific date: {e}")
                return []

        # If no filters, show overall top 10 highest sales by country
        else:
            logging.info("top highest sales by country for total")
            top_sales  = df.groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')).sort_values('total_sales', ascending=False).head(10).reset_index()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            if top_sales.empty:
                dispatcher.utter_message(text="No sales data found for the top highest sales by country.")
                return []
            else:
                response_message += f"📈 Top 10 Highest Country Overall (from {start_date} to {end_date}):\n\n"
                for idx, row in top_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
        return []


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
        today = extract_today(user_message)
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        
        table_data = []
        response_message = ""

        

        if month_year_pairs:
            logging.info(f"top country for month year :{month_year_pairs}")
            for month, year in month_year_pairs:
                lowest_sales = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(10).reset_index()
                if lowest_sales.empty:
                    response_message += f"No sales data found for Top 10 Lowest Sales Countries for {months_reverse[month]} {year}.\n"
                else:
                    response_message += f"📉 Top lowest 10 Country for {months_reverse[month]} {year} :\n"
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    

        elif years:
            logging.info(f"top country for year :{years}")
            for year in years:
                lowest_sales = df[(df['PurchaseDate'].dt.year == year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(10).reset_index()
                if lowest_sales.empty:
                    response_message += f"No sales data available for the top lowest country in {years} .\n"
                else:
                    response_message += f"📉 Top lowest 10 Country for {years} :\n"
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")

        elif months_extracted:
            current_year = datetime.now().year
            for month in months_extracted:
                lowest_sales = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)].groupby('countryname').agg(
                    total_sales=('SellingPrice', 'count'),
                    total_revenue=('SellingPrice', 'sum')
                ).sort_values('total_sales').head(10).reset_index()
                if lowest_sales.empty:
                    response_message += f"No sales data available for the top lowest country in {months_reverse[month]} {current_year}.\n"
                else:
                    response_message += f"📉 Top lowest 10 Country for {months_reverse[month]} {current_year} :\n"
                    for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                        table_data.append([idx + 1] + row.tolist())
                    table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                    response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
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
            ).sort_values('total_sales').head(10).reset_index()
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country in the last fortnight ({start_date_formatted} to {end_date_formatted}).\n"
            else:
                response_message += f"📉 Top Lowest 10 Country for last fortnight ({start_date_formatted} to {end_date_formatted}) :\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
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
            ).sort_values('total_sales').head(10).reset_index()
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country in the {quarter_name} {year}.\n"
            else:
                response_message += f"📉 Top 10 Lowest Country for {quarter_name} {year}:\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                
        elif last_day:
            lastday = (datetime.now() - timedelta(days=1)).date()
            logging.info(f"top lowest plans for last_day :{lastday}")
            lowest_sales = df[(df['PurchaseDate']==lastday)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(10).reset_index()
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country in the last day {lastday}.\n"
            else:
                response_message += f"📉 Top Lowest 10 Country for last day ({lastday}).\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
        elif today:
            today_date = datetime.now().date()

            logging.info(f"top lowest country for today :{today_date}")
            lowest_sales = df[(df['PurchaseDate']==today_date)].groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(10).reset_index()
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country on today {today_date}.\n"
            else:
                response_message += f"📉 Top Lowest 10 Country for today ({today_date}).\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                
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
                .sort_values('total_sales').head(10).reset_index()
            )
        
            if lowest_sales.empty:
                dispatcher.utter_message(
                    text=f"No sales data available for the top lowest country on {date}."
                )
            else:
                response_message = f"📉  Top lowest 10 country for the specific date {date}:\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No', 'Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                

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
            ).sort_values('total_sales').head(10).reset_index()
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country in last {num_months} months.\n"
            else:
                response_message += f"📉  Top Lowest 10 Country for last {num_months} months.\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                
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
                ).sort_values('total_sales').head(10).reset_index()
            half_name = "First Half" if start_month == 1 else "Second Half"
            if lowest_sales.empty:
                response_message += f"No sales data available for the top lowest country in the {half_name} of {year} .\n"
            else:
                response_message += f"📉 Top lowest 10 Country for {half_name} of {year} :\n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
                    
    
                    
        else:
            logging.info("top country overall")
            lowest_sales = df.groupby('countryname').agg(
                total_sales=('SellingPrice', 'count'),
                total_revenue=('SellingPrice', 'sum')
            ).sort_values('total_sales').head(10).reset_index()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            if lowest_sales.empty:
                response_message += "No data available for the top lowest plans overall."
            else:
                response_message += f"📉 Top lowest 10 Country Overall (from {start_date} to {end_date}) \n"
                for idx, row in lowest_sales[['countryname', 'total_sales', 'total_revenue']].iterrows():
                    table_data.append([idx + 1] + row.tolist())
                table_headers = ['S.No','Country', 'Sales Count', 'Total Revenue']
                response_message += tabulate(table_data, headers=table_headers, tablefmt="grid")
            

        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        dispatcher.utter_message(text=response_message)
        return []




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
        years = extract_years(user_message)  # Implement logic to extract year from message
        month_year_pairs = extract_month_year(user_message)  # Implement logic to extract month from message
        logging.info(f"Extracted countries: {countries}")
        logging.info(f"Extracted month/year pairs: {month_year_pairs}")

        logging.info(f"Extracted year: {years}")


        # Validate that two countries are provided for comparison
        if len(countries) != 2:
            detected_message = f"Detected countries: {', '.join(countries)}" if countries else "No countries detected"
            dispatcher.utter_message(text="Please provide two countries for comparison.")
            return []

        country1, country2 = countries[0], countries[1]

        # Filter data by countries and time period
        try:
            if month_year_pairs:
                logging.info(f"compare country sales {month_year_pairs}")
                for month, year in month_year_pairs:
                    df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.year == year) & (df['PurchaseDate'].dt.month == month)]
                    df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.year == year) & (df['PurchaseDate'].dt.month == month)]
                    comparison_type = f"{months_reverse[month].capitalize()}/{year}"
            elif years:
                logging.info(f"compare country sales {years}")
                # Compare whole year
                for year in years:
                    df_country1 = df[(df['countryname'] == country1) & (df['PurchaseDate'].dt.year == year)]
                    df_country2 = df[(df['countryname'] == country2) & (df['PurchaseDate'].dt.year == year)]
                    comparison_type = f"Year {year}"
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
                comparison_type = f"quarter {quarter_name} of {year}"

            else:
                dispatcher.utter_message(text="Please specify a valid year or month and year for comparison.")
                return []

            # Check if filtered data is empty
            if df_country1.empty or df_country2.empty:
                dispatcher.utter_message(text=f"No sales data found for {country1} or {country2} for {comparison_type}.")
                return []

            # Generate comparison results
            comparison_result = self.compare_sales(df_country1, df_country2, country1, country2, comparison_type)
            dispatcher.utter_message(text=f"Comparison of sales between {country1} and {country2} for {comparison_type}:\n\n{comparison_result}")
        
        except Exception as e:
            logging.error(f"Error during comparison: {e}")
            dispatcher.utter_message(text="An error occurred while comparing sales. Please try again later.")
            return []

        return []

    def compare_sales(self, df_country1, df_country2, country1, country2, comparison_type):
        result = f"Sales Comparison for {comparison_type}:\n\n"

        try:
            # Total sales comparison
            total_sales_amount_country1 = df_country1['SellingPrice'].sum()
            total_sales_amount_country2 = df_country2['SellingPrice'].sum()
            total_sales_count_country1 = df_country1['SellingPrice'].count()
            total_sales_count_country2 = df_country2['SellingPrice'].count()
            total_sales_table = [
                ["Country", "Sales Count", "Sales Price ($)"],
                [country1, total_sales_count_country1, f"${total_sales_amount_country1:,.2f}"],
                [country2, total_sales_count_country2, f"${total_sales_amount_country2:,.2f}"]
            ]
            result += tabulate(total_sales_table, headers="firstrow", tablefmt="grid") + "\n\n"

            # Calculate sales count and amount differences with descriptive messages
            sales_count_difference = abs(total_sales_count_country1 - total_sales_count_country2)
            sales_amount_difference = abs(total_sales_amount_country1 - total_sales_amount_country2)

            if total_sales_amount_country1 > total_sales_amount_country2:
                result += f"{country1} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country2}.\n\n"
            elif total_sales_amount_country1 < total_sales_amount_country2:
                result += f"{country2} had ${sales_amount_difference:,.2f} more revenue and {sales_count_difference} more sales than {country1}.\n\n"
            else:
                result += f"Sales revenue or counts were equal between {country1} and {country2}.\n\n"

            # Generate top and least plans and sources
            result += f"Top 5 plans by sales in {country1}:\n"
            result += self.get_top_plans(df_country1)
            result += f"\nTop 5 plans by sales in {country2}:\n"
            result += self.get_top_plans(df_country2)

            result += f"\nLeast 5 plans by sales in {country1}:\n"
            result += self.get_least_plans(df_country1)
            result += f"\nLeast 5 plans by sales in {country2}:\n"
            result += self.get_least_plans(df_country2)

            result += f"\nTop 5 sources by sales in {country1}:\n"
            result += self.get_top_sources(df_country1)
            result += f"\nTop 5 sources by sales in {country2}:\n"
            result += self.get_top_sources(df_country2)

            result += f"\nTop 2 payment gateways by sales in {country1}:\n"
            result += self.get_top_payment_gateways(df_country1)
            result += f"\nTop 2 payment gateways by sales in {country2}:\n"
            result += self.get_top_payment_gateways(df_country2)

            result += f"\nTop 6 refsite by sales in {country1}:\n"
            result += self.get_top_refsites(df_country1)
            result += f"\nTop 6 refsite by sales in {country2}:\n"
            result += self.get_top_refsites(df_country2)

        except Exception as e:
            logging.error(f"Error in sales comparison: {e}")
            return "An error occurred while generating the sales comparison."

        return result

    def get_top_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            plans_table = [
                ["Plan", "Sales Count", "Sales Price ($)"]
            ]
            plans_table += [[plan, count, f"${sales:,.2f}"] for plan, (sales, count) in plans_counts.iterrows()]
            return tabulate(plans_table, headers="firstrow", tablefmt="grid")
        except Exception as e:
            logging.error(f"Error fetching top plans: {e}")
            return "Could not retrieve top plans."

    def get_least_plans(self, df):
        try:
            plans_counts = df.groupby('PlanName').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nsmallest(5, 'total_sales')
            plans_table = [
                ["Plan", "Sales Count", "Sales Price ($)"]
            ]
            plans_table += [[plan, count, f"${sales:,.2f}"] for plan, (sales, count) in plans_counts.iterrows()]
            return tabulate(plans_table, headers="firstrow", tablefmt="grid")
        except Exception as e:
            logging.error(f"Error fetching least plans: {e}")
            return "Could not retrieve least plans."

    def get_top_sources(self, df):
        try:
            source_counts = df.groupby('source').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(5, 'total_sales')
            sources_table = [
                ["Source", "Sales Count", "Sales Price ($)"]
            ]
            sources_table += [[source, count, f"${sales:,.2f}"] for source, (sales, count) in source_counts.iterrows()]
            return tabulate(sources_table, headers="firstrow", tablefmt="grid")
        except Exception as e:
            logging.error(f"Error fetching top sources: {e}")
            return "Could not retrieve top sources."

    def get_top_payment_gateways(self, df):
        try:
            payment_gateway_counts = df.groupby('payment_gateway').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(3, 'total_sales')
            payment_gateway_table = [
                ["Payment Gateway", "Sales Count", "Sales Price ($)"]
            ]
            payment_gateway_table += [[pg, count, f"${sales:,.2f}"] for pg, (sales, count) in payment_gateway_counts.iterrows()]
            return tabulate(payment_gateway_table, headers="firstrow", tablefmt="grid")
        except Exception as e:
            logging.error(f"Error fetching top payment gateways: {e}")
            return "Could not retrieve top payment gateways."

    def get_top_refsites(self, df):
        try:
            refsite_counts = df.groupby('Refsite').agg(total_sales=('SellingPrice', 'sum'), sales_count=('SellingPrice', 'count')).nlargest(6, 'total_sales')
            refsite_table = [
                ["Refsite", "Sales Count", "Sales Price ($)"]
            ]
            refsite_table += [[refsite, count, f"${sales:,.2f}"] for refsite, (sales, count) in refsite_counts.iterrows()]
            return tabulate(refsite_table, headers="firstrow", tablefmt="grid")
        except Exception as e:
            logging.error(f"Error fetching top refsites: {e}")
            return "Could not retrieve top refsites."




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
        last_day = extract_last_day(user_message)
        date = extract_date(user_message)
        


        if not country_extracted:
            dispatcher.utter_message(text=f"Sorry, we do not have sales data for {country_extracted}. Please provide another country.")
            logging.info(f"Country {country_extracted} not found in the dataset.")
            return []
        country = country_extracted[0]
    

        response_message = f"📊 Sales Overview for {country.upper()}:\n\n"

        
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
                    dispatcher.utter_message(text=f"No sales data found for {months_reverse[month]} {year} in {country.upper()}.")
                    return []
                response_message += f"🔍 Sales Overview for {months_reverse[month]} {year} in {country.upper()}:\n"
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"\n Most Sold Plans in {country.upper()}:\n"
                most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += most_sold_table + "\n"
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country.upper()}:\n"
                least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += least_sold_table + "\n"
                
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
                dispatcher.utter_message(text=f"No sales data found for {quarter_name} {year} in {country.upper()}.")
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Sales Overview for {quarter_name} {year} in {country.upper()}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
            
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += "\n📉 Least Sold Plans:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                
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
                dispatcher.utter_message(text=f"No sales data found for {half_name} {year} in {country.upper()}.")
                return []
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Sales Overview for {half_name} {year} in {country.upper()}:\n"
            response_message += "📈 Most Sold Plans:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += "\n📉 Least Sold Plans:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                
        

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
                dispatcher.utter_message(text=f"No sales data found for the fortnight ({start_date_formatted} to {end_date_formatted}) in {country.upper()}.")
                return []
            response_message += f"🔍 Sales Overview for Last Fortnight ({start_date_formatted} to {end_date_formatted}) in {country.upper()}:\n"
            
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += "📈 Most Sold Plans:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
            
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += "\n📉 Least Sold Plans:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                

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
                dispatcher.utter_message(text=f"No sales data found for today ({today_date}) in {country.upper()}.")
                return []
            response_message += f"🔍 Sales Overview for Today ({today_date}) in {country.upper()}:\n"
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n Least Sold Plans in {country.upper()}:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                
        
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
                dispatcher.utter_message(text=f"No sales data found for yesterday ({lastday}) in {country.upper()}.")
                return
            
        
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Sales Overview for last day ({lastday}) in {country.upper()}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n Least Sold Plans in {country.upper()}:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
            
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
                dispatcher.utter_message(text=f"No sales data found for the last {num_months} months  ({start_date_formatted} to {end_date_formatted})  in {country.upper()}.")
                return []
    
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Sales Overview for the Last {num_months} Months  ({start_date_formatted} to {end_date_formatted}) in {country.upper()}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
        
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n Least Sold Plans in {country.upper()}:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                
        elif date:
            logging.info(f"most and least sold plans for specific date: {date}")
            # Convert the extracted specific date to a pandas datetime object
            
            # Check if the date is valid
            if pd.isna(date):
                dispatcher.utter_message(text=f"Sorry, I couldn't understand the date format. Please provide a valid date.")
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
                dispatcher.utter_message(text=f"No sales data found for {date} in {country.upper()}.")
                return []
            
        
            # Most sold plans for specific date
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Sales Overview for {date} in {country.upper()}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()} on {date}:\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
        
            # Least sold plans for specific date
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n Least Sold Plans in {country.upper()} on {date}:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                

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
                    dispatcher.utter_message(text=f"No sales data found for {year} in {country.upper()}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {year} in {country.upper()}:\n"
                response_message += f"\n Most Sold Plans in {country.upper()}:\n"
                most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += most_sold_table + "\n"
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country}:\n"
                least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += least_sold_table + "\n"
                

        
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
                    dispatcher.utter_message(text=f"No sales data found for {months_reverse[month]} {current_year} in {country.upper()}.")
                    return []
                
                # Most sold plans
                most_sold = top_sales.nlargest(5, 'SalesCount')
                response_message += f"🔍 Sales Overview for {months_reverse[month]} {current_year} in {country.upper()}:\n"
                response_message += f"\n Most Sold Plans in {country.upper()}:\n"
                most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += most_sold_table + "\n"
                
                # Least sold plans
                least_sold = top_sales.nsmallest(5, 'SalesCount')
                response_message += f"\n Least Sold Plans in {country.upper()}:\n"
                least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
                response_message += least_sold_table + "\n"
                
        # If no filters, show overall top 10 highest sold plans by country
        else:
            logging.info("total most and least sold plans")
            top_sales = df[df['countryname'].str.lower() == country.lower()].groupby('PlanName').agg(
                SalesCount=('SellingPrice', 'count'),
                TotalRevenue=('SellingPrice', 'sum')
            ).fillna(0).reset_index()
            if top_sales.empty:
                dispatcher.utter_message(text=f"No sales data found for {country.upper()}.")
                return []
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            # Most sold plans
            most_sold = top_sales.nlargest(5, 'SalesCount')
            response_message += f"🔍 Overall Most Sold Plans in {country.upper()} (from {start_date} to {end_date}):\n"
            response_message += f"\n Most Sold Plans in {country.upper()}:\n"
            most_sold_table = tabulate(most_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += most_sold_table + "\n"
            
            # Least sold plans
            least_sold = top_sales.nsmallest(5, 'SalesCount')
            response_message += f"\n 🔍 Overall Least Sold Plans in {country.upper()}:\n"
            least_sold_table = tabulate(least_sold[['PlanName', 'SalesCount', 'TotalRevenue']], headers=["Plan Name", "Sales Count", "Total Revenue"], tablefmt="grid")
            response_message += least_sold_table + "\n"
                
        if not response_message.strip():
            dispatcher.utter_message(text="No sales data found for the specified criteria.")
            return []

        # Send the formatted message
        dispatcher.utter_message(text=response_message)
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
            dispatcher.utter_message(text=f"Data is missing the following required columns: {', '.join(missing_columns)}")
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
        

        response_message = ""

        try: 
            if month_year_pairs:
                logging.info(f"source, refsite,payment_gateway sales in {month_year_pairs}")
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[month_year_pairs[0][0]].capitalize()} {month_year_pairs[0][1]}:\n\n"
                for month, year in month_year_pairs:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == year)]
                    if  filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df)

            elif years:
                logging.info(f"source, refsite,payment_gateway sales in {years}")
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {years[0]}:\n\n"
                for year in years:
                    filtered_df = df[df['PurchaseDate'].dt.year == year]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {year}.\n"
                    else:
                        response_message += self.process_sales_data(filtered_df)

            elif months_extracted:
                logging.info(f"source, refsite,payment_gateway sales in {months_extracted}")
                current_year = datetime.now().year
                response_message += f"📅 Sales Overview by Source, Payment Gateway, and Refsite for {months_reverse[months_extracted[0]].capitalize()} {current_year}:\n\n"
                for month in months_extracted:
                    filtered_df = df[(df['PurchaseDate'].dt.month == month) & (df['PurchaseDate'].dt.year == current_year)]
                    if filtered_df.empty:
                        response_message += f"No sales data found for {months_reverse[month]} {current_year}.\n" 
                    else:
                        response_message += self.process_sales_data(filtered_df)
            else:
                logging.info("source, refsite,payment_gateway sales ")
                start_date = df['PurchaseDate'].min().date()
                end_date = df['PurchaseDate'].max().date()
                response_message += f"📊 Overall Sales Overview (from {start_date} to {end_date}) by Source, Payment Gateway, and Refsite:\n\n"
                response_message += self.process_sales_data(df)

            # If no data is available after processing, inform the user
            if not response_message.strip():
                dispatcher.utter_message(text="No sales data found for the specified criteria.")
                return []

            # Send the formatted message
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"An error occurred while processing sales data: {str(e)}")
            dispatcher.utter_message(text="An error occurred while processing the sales data. Please try again later.")

        return []

    def process_sales_data(self, filtered_df):
        response = ""

        # Process sources
        if 'source' in filtered_df.columns:
            response += "Sales Overview by Source:\n"
            response += self.get_top_n_sales(filtered_df, 'source')+ "\n"

        # Process payment gateways
        if 'payment_gateway' in filtered_df.columns:
            response += "\nSales Overview by Payment Gateway:\n"
            response += self.get_top_n_sales(filtered_df, 'payment_gateway')+ "\n"

        # Process refsites
        if 'Refsite' in filtered_df.columns:
            response += "\nSales Overview by Refsite:\n"
            response += self.get_top_n_sales(filtered_df, 'Refsite')+ "\n"

        return response

    def get_top_n_sales(self, df, column, n=5):
        """Helper function to calculate top N highest and lowest sales."""
        response = ""

        # Aggregate data by column
        sales_summary = df.groupby(column)['SellingPrice'].agg(['sum', 'count']).reset_index()
        sales_summary.columns = [column, 'TotalRevenue', 'SalesCount']

        # Get top N highest sales
        top_sales = sales_summary.nlargest(n, 'SalesCount')
        top_sales_data = [(row[column], row['TotalRevenue'], row['SalesCount']) for _, row in top_sales.iterrows()]
        response += f"Top {n} Highest Sales by {column.capitalize()}:\n"
        response += self.format_sales_data(top_sales_data, column) + "\n\n"


        # Get top N lowest sales
        lowest_sales = sales_summary.nsmallest(n, 'SalesCount')
        lowest_sales_data = [(row[column], row['TotalRevenue'], row['SalesCount']) for _, row in lowest_sales.iterrows()]
        response += f"Top {n} Lowest Sales by {column.capitalize()}:\n"
        response += self.format_sales_data(lowest_sales_data, column)

        return response
    def format_sales_data(self, data, category):
   
        headers = [category.capitalize(), "Total Revenue ($)", "Sales Count"]
        table = tabulate(data, headers=headers, tablefmt="grid")
        return table
   

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
                dispatcher.utter_message(text="Required columns 'PurchaseDate' or 'SellingPrice' are missing from the dataset.")
                return []

            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate', 'SellingPrice'])  # Drop rows with invalid dates or sales data
            

            if df.empty:
                dispatcher.utter_message(text="No valid sales data available for analysis.")
                return []

            df['MonthYear'] = df['PurchaseDate'].dt.to_period('M')
            df['Year'] = df['PurchaseDate'].dt.year

            # Calculate monthly sales totals and counts
            monthly_data = df.groupby("MonthYear").agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()

            # Prepare the response message
            response_message = f"📊 Sales Metrics Overview (from {start_date} to {end_date}):\n\n"

            # Sales Overview
            response_message += " 📅 Monthly Sales Summary :\n"
            if monthly_data.empty:
                response_message += "No monthly sales data available.\n\n"
            else:
                # for index, row in monthly_data.iterrows():
                #     response_message += (
                #         f" {row['MonthYear']} \n"
                #         f" Total Sales: ${row['TotalSales']:,.2f}\n"
                #         f" Sales Count: {row['SalesCount']}\n\n"
                #     )
                response_message += tabulate(
                    monthly_data, 
                    headers=["MonthYear","TotalSales ($)",  "SalesCount"], 
                    tablefmt="grid"
                )
                response_message += "\n\n"
                    

            # Prepare yearly summary
            yearly_summary = df.groupby('Year').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            # Add yearly data to the response
            response_message += " 📅 Yearly Sales Summary:\n"
            if yearly_summary.empty:
                response_message += "No yearly sales data available.\n\n"
            else:
                response_message += tabulate(
                    yearly_summary, 
                    headers=["Year", "TotalSales ($)", "SalesCount"], 
                    tablefmt="grid"
                )
                # for index, row in yearly_summary.iterrows():
                #     response_message += (
                #         f"{row['Year']}\n"
                #         f"Total Sales: ${row['TotalSales']:,.2f} \n"
                #         f"Sales Count: {row['SalesCount']}\n\n"
                #     )
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"Error while calculating sales metrics: {str(e)}")
            dispatcher.utter_message(text="An error occurred while calculating sales metrics. Please try again later.")

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
                dispatcher.utter_message(text="The required columns for processing are missing in the dataset.")
                return []
            # def format_repeated_email_data(entry):
            #     table = tabulate(
            #         [entry],
            #         headers=['IOrderId', 'Email', 'Repeated Count', 'Plans', 'Purchase Dates'],
            #         tablefmt='grid'
            #     )
               
            #     return table
            def format_repeated_email_data(entry):
                lines = [
                    f"\nEmail: {entry['Email']}",
                    f"Repeated {entry['Count']} times",
                    "Plans and Purchase Dates:"
                ]
                for plan, date, orders_id in zip(entry['Plans'], entry['PurchaseDates'],entry['Order_ids']):
                    lines.append(f"  - Plan: {plan}, Purchase Date: {date},  Order ID: {orders_id}")
                return "\n".join(lines)
                
            def split_and_send_response(data, dispatcher, chunk_size=2000):
                current_chunk = ""
                for item in data:
                    if len(current_chunk) + len(item) > chunk_size:
                        dispatcher.utter_message(text=current_chunk)
                        current_chunk = item
                    else:
                        current_chunk += f"\n{item}\n"
                if current_chunk:
                    dispatcher.utter_message(text=current_chunk)
            response_lines = []


            
        
            # Condition: Month-Year specific
            if month_year_pairs:
                logging.info(f"repeated emails in month year pairs: {month_year_pairs}")
                for month, year in month_year_pairs:
                    filtered_data = df[
                        (df['PurchaseDate'].dt.year == year) &
                        (df['PurchaseDate'].dt.month == month)
                    ]
                    email_counts = filtered_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].index.tolist()
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
                        total_repeated_in_month_year = filtered_data['Email'].nunique()
                   
                        if grouped:
                            for entry in grouped:
                                response_lines.append(format_repeated_email_data(entry))
                            response_lines.append(f"\n📊 Total number of repeated emails in {months_reverse[month].capitalize()} {year}: {total_repeated_in_month_year}")
                    else:
                        response_lines.append(f"No repeated email data found for {months_reverse[month].capitalize()} {year}.\n")

            # Condition: Year specific
            elif years:
                logging.info(f"repeated emails in year : {years}")
                for year in years:
                    filtered_data = df[
                        (df['PurchaseDate'].dt.year == year)
                    ]
                    email_counts = filtered_data['Email'].value_counts()
                    repeated_emails = email_counts[email_counts > 1].index.tolist()
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
                        total_repeated_in_year = filtered_data['Email'].nunique()
                        response_lines.append(f"📅 Repeated Emails in {year}:\n")
                        if grouped:
                            for entry in grouped:
                                response_lines.append(format_repeated_email_data(entry))
                            response_lines.append(f"\n📊 Total number of repeated emails in {year}: {total_repeated_in_year}")
                        
                    
                    else:
                        response_lines.append(f"No repeated email data found for {year}.\n")

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
                    repeated_emails = email_counts[email_counts > 1].index.tolist()
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
                        
                        total_repeated_in_month = filtered_data['Email'].nunique()
                        response_lines.append(f"📅 Repeated Emails in {months_reverse[month].capitalize()} {current_year}:\n")
                        if grouped:
                            for entry in grouped:
                                response_lines.append(format_repeated_email_data(entry))
                            response_lines.append(f"\n📊 Total number of repeated emails in {months_reverse[month].capitalize()} {current_year}: {total_repeated_in_month}")
                    else:
                        response_lines.append(f"No repeated email data found for {months_reverse[month].capitalize()} {current_year}.\n")
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
                    repeated_emails = email_counts[email_counts > 1].index.tolist()
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
                        
                        total_repeated_in_quarter = quarter_data['Email'].nunique()
                        response_lines.append(f"📅 Repeated Email Details for {quarter_name} {year}:\n")
                        for entry in grouped:
                            response_lines.append(format_repeated_email_data(entry))
                        response_lines.append(f"\n📊 Total number of repeated emails in {quarter_name} {year}: {total_repeated_in_quarter}")
                    else:
                        response_lines.append(f"No repeated email data found for {quarter_name} {year}.\n")
                except Exception as e:
                    logging.error(f"Error processing quarterly repeated emails: {e}")
                    dispatcher.utter_message(
                        text="An error occurred while processing quarterly repeated email details. Please try again later."
                    )
                    return []
            # Default: Overall condition
            else:
                logging.info("repeated emails in overall")
                email_counts = df['Email'].value_counts()
                repeated_emails = email_counts[email_counts > 1].index.tolist()
    
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
                    response_lines.append(f"📅 Repeated Emails Overall (from {start_date} to {end_date}) :\n")
                    for entry in grouped:
                        response_lines.append(format_repeated_email_data(entry))
                    response_lines.append(f"\n📊 Total number of repeated emails: {repeated_data['Email'].nunique()}")

            # Split and send response
            
            
            if response_lines:
                split_and_send_response(response_lines, dispatcher)

            return []
            # response_text = "\n".join(response_lines)
        except Exception as e:
            logging.error(f"Error processing repeated email details: {e}")
            dispatcher.utter_message(text="An error occurred while retrieving repeated email details. Please try again later.")
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
                dispatcher.utter_message(text="Error: The sales data is empty or invalid.")
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
            today = extract_today(user_message)
            last_day = extract_last_day(user_message)
            date = extract_date(user_message)

            total_profit_margin = 0.0
            
            #Handle specific date
            if date:
                
                logging.info(f"Profit margin for specific date: {date}")
            

                daily_profit_margin, error_message = extract_profit_margin_sales_for_specific_date(df, date)
                if error_message:  # Check if there's an error message
                    dispatcher.utter_message(error_message)
    
                elif daily_profit_margin > 0:
                    dispatcher.utter_message(f"The profit margin for {date} is ${daily_profit_margin:.2f}.")
                else:
                    dispatcher.utter_message(f"No sales were recorded on {date}.")
                

            # Handle today's profit margin
            if today:
                today_date = datetime.now().date()
                logging.info(f"profit margin of today {today_date}")
                today_profit_margin = df[df['PurchaseDate'].dt.date == today_date]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for today ({today_date}) is ${today_profit_margin:.2f}."
                )
                return []
            if last_day:
                lastday = (datetime.now() - timedelta(days=1)).date()
                logging.info(f"profit margin of last day {lastday}")
                yesterday_profit_margin = df[df['PurchaseDate'].dt.date == lastday]['ProfitMargin'].sum()
                dispatcher.utter_message(
                    text=f"The profit margin for yesterday ({lastday}) is ${yesterday_profit_margin:.2f}."
                )
                return []
            if fortnight:
                logging.info(f"profit margin of fortnight {fortnight}")
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
                dispatcher.utter_message(
                    text=f"The profit margin for the last fortnight (from {start_date_formatted} to {end_date_formatted}) is ${fortnight_profit_margin:.2f}."
                )
                return []
            if last_n_months:
                logging.info(f"profit margin of last  months{last_n_months}")
                try:
                    start_date, end_date, num_months = last_n_months
        
        # Calculate the number of months between start and end dates
                    # num_months = (end_date.year - start_date.year) * 12 + (end_date.month - start_date.month)
        
                    
                    last_months_profit_margin = df[
                        (df['PurchaseDate'] >= start_date) &
                        (df['PurchaseDate'] <= end_date)
                    ]['ProfitMargin'].sum()
                    start_date_formatted = start_date.date()
                    end_date_formatted = end_date.date()
                    if last_months_profit_margin > 0:
                        dispatcher.utter_message(
                            text=f"The profit margin for the last {num_months} months (from {start_date_formatted} to {end_date_formatted}) is ${last_months_profit_margin:.2f}."
                        )
                    else:
                        dispatcher.utter_message(
                            text=f"No profit margin recorded for the last {num_months} months (from {start_date_formatted} to {end_date_formatted})."
                        )
                    
                    return []
                except Exception as e:
                    logging.error(f"Error calculating profit margin: {e}")
                    dispatcher.utter_message(
                        text="Could not process the profit margin for the last months. Please provide a valid date range."
                    )
                    return []

            if quarterly:
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
                    
                    dispatcher.utter_message(
                        text=f"The profit margin for {quarter_name} of {year} is ${quarterly_profit_margin:.2f}."
                    )
                except Exception as e:
                    dispatcher.utter_message(
                        text=f"An error occurred while processing quarterly sales: {str(e)}"
                    )
                return []
                    

            if half_year:
                logging.info(f"Profit margin of half yearly {half_year}")
                year,(start_month, end_month) = half_year
              
                try:
                    # if current_month <= 6:
                    #     start_month, end_month = 1, 6
                    # else:
                    #     start_month, end_month = 7, 12
            
                    # Filter the DataFrame for the specific half of the year
                    half_yearly_profit_margin_data = df[
                        (df['PurchaseDate'].dt.month >= start_month) & 
                        (df['PurchaseDate'].dt.month <= end_month) & 
                        (df['PurchaseDate'].dt.year == year)
                    ]['ProfitMargin'].sum()
            
                
                    # Determine whether it's the first or second half of the year
                    half_name = "First Half" if start_month == 1 else "Second Half"
                    
                    # Log and respond with the calculated profit margin
                    logging.info(f"Profit margin for {half_name} of {year}: ${half_yearly_profit_margin_data:.2f}")
                    dispatcher.utter_message(
                        text=f"The profit margin for the {half_name} of {year} is ${half_yearly_profit_margin_data:.2f}."
                    )
                    
                except Exception as e:
                    logging.error(f"Error calculating profit margin for {half_name} of {year}: {e}")
                    dispatcher.utter_message(
                        text=f"Error calculating profit margin for the {half_name} of {year}. Please try again."
                    )
                return []



            # Handle month-year pairs
            if month_year_pairs:
                logging.info(f"profit margin of month year {month_year_pairs}")
                try:
                    for month, year in month_year_pairs:
                        month_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == year)
                        ]['ProfitMargin'].sum()
                        dispatcher.utter_message(
                            text=f"The profit margin for {months_reverse[month]} {year} is ${month_profit_margin:.2f}."
                        )
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly profit margins: {str(e)}")
                return []

            # Handle years
            if years:
                logging.info(f"profit margin of year {years}")
                try:
                    for year in years:
                        yearly_profit_margin = df[
                            (df['PurchaseDate'].dt.year == year)
                        ]['ProfitMargin'].sum()
                        dispatcher.utter_message(
                            text=f"The total profit margin for {year} is ${yearly_profit_margin:.2f}.")
                        
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing yearly profit margins: {str(e)}")
                return []

            # Handle months in the current year
            if months_extracted:
                logging.info(f"profit margin of month with current year {months_extracted}")
                current_year = pd.to_datetime('today').year
                try:
                    for month in months_extracted:
                        monthly_profit_margin = df[
                            (df['PurchaseDate'].dt.month == month) &
                            (df['PurchaseDate'].dt.year == current_year)
                        ]['ProfitMargin'].sum()
                        dispatcher.utter_message(
                            text=f"The profit margin for {months_reverse[month]} {current_year} is ${monthly_profit_margin:.2f}."
                        )
                except Exception as e:
                    dispatcher.utter_message(text=f"Error occurred while processing monthly profit margins: {str(e)}")
                return []

            # Handle total profit margin
            logging.info("total profit margin")
            total_profit_margin = df['ProfitMargin'].sum()
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()
            dispatcher.utter_message(
                text=f"The overall total profit margin (from {start_date} to {end_date}) is ${total_profit_margin:.2f}."
            )

        except Exception as e:
            dispatcher.utter_message(text=f"An error occurred while processing your request: {str(e)}")
        
        return []

################################################################country sales metric########################

class ActionCalculateCountrySalesMetrics(Action):
    
    def name(self) -> str:
        return "action_calculate_country_sales_metrics"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculateCountrySalesMetrics...")

        global df
        try:
            # Check if the necessary columns exist
            if 'countryname' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns 'countryname' or 'SellingPrice' are missing from the dataset.")
                return []

            df = df.dropna(subset=['countryname', 'SellingPrice','PurchaseDate'])  # Drop rows with invalid country or sales data
            
            if df.empty:
                dispatcher.utter_message(text="No valid sales data available for analysis.")
                return []
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')
            df = df.dropna(subset=['PurchaseDate']) 

            # Calculate country-wise sales totals and counts
            country_data = df.groupby('countryname').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()
            country_data = country_data.sort_values(by='SalesCount', ascending=False)
            country_data = country_data.reset_index(drop=True)
            start_date = df['PurchaseDate'].min().date()
            end_date = df['PurchaseDate'].max().date()


            # Prepare the response message
            response_message = f"📊 Country Sales Metrics Overview (from {start_date} to {end_date}):\n\n"

            # Country Sales Summary
            response_message += " 🌍 Sales by Country:\n"
            if country_data.empty:
                response_message += "No sales data available by country.\n\n"
            else:
                # Create a tabulated summary of country sales
                country_table = tabulate(
                    country_data,
                    headers= ["Country",  "Total Sales","Sales Count"],
                    tablefmt="grid"
                )
                response_message += f"```\n{country_table}\n```\n\n"

            # Display the response message
            dispatcher.utter_message(text=response_message)

        except Exception as e:
            logging.error(f"Error while calculating country sales metrics: {str(e)}")
            dispatcher.utter_message(text="An error occurred while calculating country sales metrics. Please try again later.")

        return []
#####################################################planname sales metrics#####################333
def split_large_message(message: str, max_length=2000):
    """Splits a message into chunks of a specified maximum length."""
    return [message[i:i+max_length] for i in range(0, len(message), max_length)]


class ActionCalculatePlanSalesMetrics(Action):
    
    def name(self) -> str:
        return "action_calculate_plan_sales_metrics"

    def run(self, dispatcher: CollectingDispatcher, tracker: Tracker, domain: Dict[Text, Any]) -> List[Dict[Text, Any]]:
        logging.info("Running ActionCalculatePlanSalesMetrics...")

        global df
        try:
            # Check if the necessary columns exist
            if 'PlanName' not in df.columns or 'SellingPrice' not in df.columns:
                dispatcher.utter_message(text="Required columns 'PlanName' or 'SellingPrice' are missing from the dataset.")
                return []

            df = df.dropna(subset=['PlanName', 'SellingPrice','PurchaseDate'])  # Drop rows with invalid plan or sales data
            df['PurchaseDate'] = pd.to_datetime(df['PurchaseDate'], errors='coerce')

            # Filter data to exclude sales before January 1, 2024
            six_months_ago = datetime.now() - timedelta(days=3 * 30)
            today = datetime.now()
            df = df[(df['PurchaseDate'] >= six_months_ago) & (df['PurchaseDate'] <= today)]
            if df.empty:
                dispatcher.utter_message(text="No valid sales data available for analysis.")
                return []

            # Calculate sales totals and counts by PlanName
            plan_data = df.groupby('PlanName').agg(
                TotalSales=('SellingPrice', 'sum'),
                SalesCount=('SellingPrice', 'count')
            ).reset_index()

            plan_data = plan_data.sort_values(by='SalesCount', ascending=False)
            plan_data = plan_data.reset_index(drop=True)
            start_date = six_months_ago.date()
            end_date = today.date()

            # Prepare the response message
            response_message = f"📊 Plan Sales Metrics Overview (From {start_date} to {end_date}):\n\n"

            # Sales by Plan Summary
            response_message += " 📅 Sales by Plan for last 3 months:\n"
            if plan_data.empty:
                response_message += "No sales data available by plan.\n\n"
            else:
                # Create a tabulated summary of plan sales
                plan_table = tabulate(
                    plan_data,
                    headers=[ "Plan Name","Total Sales", "Sales Count" ],
                    tablefmt="grid"
                )
                response_message += f"```\n{plan_table}\n```\n\n"

            for part in split_large_message(response_message):
                dispatcher.utter_message(text=part)

        except Exception as e:
            logging.error(f"Error while calculating plan sales metrics: {str(e)}")
            dispatcher.utter_message(text="An error occurred while calculating plan sales metrics. Please try again later.")

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
        #user_message = tracker.latest_message.get('text', '').lower()
        
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
        
        # Respond based on the condition
        if condition == "daily":
            table = tabulate(df1, headers='keys', tablefmt='grid', showindex=False)
            dispatcher.utter_message(text=f"Daily Sales Forecast:\n{table}")
        elif condition == "monthly":
            table = tabulate(df2, headers='keys', tablefmt='grid', showindex=False)
            dispatcher.utter_message(text=f"Monthly Sales Forecast:\n{table}")
        elif condition == "yearly":
            table = tabulate(df3, headers='keys', tablefmt='grid', showindex=False)
            dispatcher.utter_message(text=f"Yearly Sales Forecast:\n{table}")
        else:
            dispatcher.utter_message(
                text="I couldn't understand the time frame. Please specify daily, monthly, or yearly."
            )

        return []