# cd app\ 
# streamlit run app.py
import streamlit as st
import xgboost as xgb
import pickle
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import os 
from PIL import Image
import matplotlib.ticker as ticker
import altair as alt
import math 
import joblib
import time
import io
import pyodbc
import numpy as np
import shutil
import plotly.express as px
import plotly.io as pio
import plotly.graph_objects as go
from sklearn.preprocessing import FunctionTransformer
import fetch_data
import key_metrics_v2
import transformation_phase2
import features_phase2_v2

# Force Plotly to use 'json' as the engine instead of 'orjson'
pio.json.config.default_engine = 'json'

# from model_phase2 import run_model,categorize_credit_limit
import transformation_phase2 
data_folder = r"..\.."

st.set_page_config(
    page_title="Credit Limit Optimisation",
    layout="wide",
    initial_sidebar_state="expanded"
)

def col_starts_with(column_start,data):
    return [x for x in data.columns if x.startswith(column_start)]

# Functions 
@st.cache_data
# def read_standardized_CL(file_path = r'Standardized CL\Standardization_based_on_logic.xlsx'):
def read_standardized_CL(file_path = r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\cl_std_v2\cl_std_combined.pkl'):
    # data = pd.read_excel(os.path.join(data_folder,file_path))
    data = pd.read_pickle(file_path)
    data['Customer'] = data['Customer'].map(int)
    data = data[~data['ASM Code'].isin(['DORMANT','LEGAL','WOP'])]
    return data

@st.cache_data
def read_predicted_proba_inc_dec_CL(file_path = r'Model Data\PRED\RESULT.xlsx'):
    data = pd.read_excel(os.path.join(data_folder,file_path))
    data['Customer'] = data['Customer'].map(int)
    return data

@st.cache_resource
def load_state_encoding():
   return {'West Bengal': 839,
    'Orissa': 524,
    'Madhya Pradesh': 1824,
    'Himachal Pradesh': 23,
    'Chhattisgarh': 495,
    'Karnataka': 215,
    'Assam': 119,
    'Uttarakhand': 143,
    'Tamil Nadu': 42,
    'Delhi': 7,
    'Jharkhand': 22,
    'Chandigarh': 1,
    'Tripura': 3,
    'Manipur': 1,
    'Punjab': 1684,
    'Rajasthan': 1601,
    'Gujarat': 1036,
    'Uttar Pradesh': 2454,
    'Andhra Pradesh': 1317,
    'Haryana': 1020,
    'Telangana': 864,
    'Bihar': 273,
    'Maharashtra': 1235}
def indian_numbering_format(x, pos):
    if x >= 10**7:
        return f'{x/10**7:.1f} Crores'
    elif x >= 10**5:
        return f'{x/10**5:.1f} Lakhs'
    elif x >= 10**3:
        return f'{x/10**3:.1f} Thousands'
    else:
        return f'{x:.0f}'
# def user_input_features(state_encoding):
#     st.header("Enter Feature Values:")
    
#     # User selects the state
#     state = st.selectbox("Select State", options=list(state_encoding.keys()))
    
#     # Convert to frequency encoding
#     state_freq = state_encoding.get(state, 0)  # Default to 0 if not found

#     feature_values = {}
#     feature_values = {
#         "State_FreqEnc": state_freq,
#         "Opening Balance": st.number_input("Opening Balance", value=0.0),
#         "Recency": st.number_input("Recency", value=0.0),
#         "days_since_first_activity": st.number_input("Days Since First Activity", value=0.0),
#         "Number of invoices Total": st.number_input("Number of Invoices Total", value=0.0),
#         "Number of Outstanding Invoices": st.number_input("Number of Outstanding Invoices", value=0.0),
#         "Number_of_defaults_total": st.number_input("Number of Defaults Total", value=0.0),
#         "Missed Payment Ratio": st.number_input("Missed Payment Ratio", value=0.0),
#         "Total Profit total": st.number_input("Total Profit Total", value=0.0),
#         "Average Past Due Days Total": st.number_input("Average Past Due Days Total", value=0.0),
#         "debtor_days": st.number_input("Debtor Days", value=0.0),
#         "Last 12 months max sales": st.number_input("Last 12 Months Max Sales", value=0.0),
#         "Last 12 months min sales": st.number_input("Last 12 Months Min Sales", value=0.0),
#         "Average Transaction Size": st.number_input("Average Transaction Size", value=0.0),
#         "Avg Yearly Sales": st.number_input("Avg Yearly Sales", value=0.0),
#         "Avg Monthly Sales_Total": st.number_input("Avg Monthly Sales Total", value=0.0),
#         "Each Month Average Sales": st.number_input("Each Month Average Sales", value=0.0),
#         "Collection Efficiency%": st.number_input("Collection Efficiency%", value=0.0),
#         "Last 12 months max collection": st.number_input("Last 12 Months Max Collection", value=0.0),
#         "Total Amount Defaulted": st.number_input("Total Amount Defaulted", value=0.0),
#     }
    
#     return pd.DataFrame([feature_values])
col1, col2 = st.columns(2)
with col1:
    if st.button("Clear Data Cache"):
        st.cache_data.clear()  # Clears data cache
        st.success("Data Cache Cleared!")
with col2:
    if st.button("Reset & Clear All Files"):
        # Clear all session state variables
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.success("All stored files and session data have been cleared. Please upload new files.")
        time.sleep(0.5)
        st.rerun()  # Refresh the app to start over

def user_input_features(features_file):
    uploaded_file = features_file
    
    if uploaded_file is not None:
        
        if isinstance(uploaded_file, pd.DataFrame):
            df = uploaded_file 

        else:
            if uploaded_file.name.endswith('.pkl'):
                df = pd.read_pickle(uploaded_file)  # Load as Pickle
            elif uploaded_file.name.endswith('.xlsx') or uploaded_file.name.endswith('.xls'):
                df = pd.read_excel(uploaded_file)  # Load as Excel
            elif uploaded_file.name.endswith('.csv') or uploaded_file.name.endswith('.csv'):
                df = pd.read_csv(uploaded_file)  # Load as CSV
            else:
                st.error("Unsupported file format. Please upload a .pkl or .xlsx file.")
        # Check if 'State' column exists
        # if 'State' in df.columns:
        #     df['State_FreqEnc'] = df['State'].map(state_encoding).fillna(0)  # Apply frequency encoding
        # df.drop(columns = 'Customer', inplace = True)
        return df
    else:
        st.warning("Please upload a CSV or an Excel file to proceed.")
        return None

@st.cache_resource
def load_transformer():
    with open("log1p_transformer.pkl", "rb") as f:
        return pickle.load(f)


def preprocess_data_test(df):
    # Load dataset
    # df = pd.read_csv(file_path)
    # Display basic info
    print(df.info())  # Check data types and missing values
    print(df.head())  # View first few rows
    df.set_index(df['Customer'], inplace=True)
    df.drop(columns = ['Customer'], axis = 1, inplace = True)
    # y_true = df['Std_Credit_Limit']
    # df.drop(columns = ['Std_Credit_Limit'], inplace = True)
    numeric_cols = df.select_dtypes(include=['number']).columns
    # scaler = StandardScaler()
    scaler = joblib.load(r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase2\scaler_final.pkl')
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df.update(df[numeric_cols])
    # return df, y_true
    return df

def preprocess_data_view(df):
    # Load dataset
    # df = pd.read_csv(file_path)
    # st.write(df.columns)
    # Display basic info
    print(df.info())  # Check data types and missing values
    print(df.head())  # View first few rows

    # Check for null values
    null_columns = df.columns[df.isnull().any()]  # Get columns with null values
    print("Columns with null values:")
    print(null_columns)
    print("Null values count per column:")
    print(df[null_columns].isnull().sum())
    
    # Data preprocessing
    df = df[(~df['ASM Code'].isin(['WOP', 'BLOCKED', 'LEGAL', 'DORMANT', 'B2B'])) & (df['Std_Credit_Limit'] > 1000)]
    # df = df[(~df['ASM Code'].isin(['WOP', 'BLOCKED', 'LEGAL', 'DORMANT', 'B2B']))]
    df.fillna(0, inplace=True)
    df.drop(columns='ASM Code', inplace=True)
    df.set_index(df['Customer'], inplace=True)
    df.drop(columns='Customer', inplace=True)
    df['Std_Credit_Limit'] = df['Std_Credit_Limit'].astype('int64')
    actual_credit_limits = df['Std_Credit_Limit']
    df.drop(columns = ['Std_Credit_Limit'], inplace = True, axis = 1)
    df['Recency'] = pd.to_numeric(df['Recency'], errors='coerce')
    df['Recency'] = df['Recency'].astype('float64')

    # Map states to regions
    # state_regions = {
    #     "North": ['Punjab', 'Rajasthan', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Uttarakhand', 'Delhi', 'Jammu & Kashmir', 'Chandigarh'],
    #     "South": ['Andhra Pradesh', 'Telangana', 'Karnataka', 'Tamil Nadu', 'Kerala', 'Puducherry'],
    #     "East": ['Bihar', 'West Bengal', 'Orissa', 'Meghalaya', 'Assam', 'Jharkhand', 'Tripura', 'Manipur'],
    #     "West": ['Gujarat', 'Maharashtra', 'Goa', 'Dadra & Nagar Haveli'],
    #     "Central": ['Madhya Pradesh', 'Chhattisgarh'],
    # }
    # flat_state_mapping = {state: region for region, states in state_regions.items() for state in states}
    # df['State_Region'] = df['State'].map(flat_state_mapping)

    # One-hot encoding for State_Region
    # df = pd.get_dummies(df, columns=['State_Region'], prefix='Region')

    # Further preprocessing
    # df['Customer'] = pd.to_numeric(df['Customer'])
    

    # Select relevant columns
    # df = df[['Last 12 months max sales', 'avg_closing_bal_last_12_months', 'max_closing_bal', 'Last 12 months avg collection', 'max collection', 'median collection', 'min_closing_bal', 'net_rev_365', 'avg collection', 'max sales', 'Total Profit', 'Avg Yearly Sales', 'avg_closing_bal', 'Avg Yearly Collections', 'Last 12 months median collection', 'min collection', 'max_missed_payment', 'Average Number of invoices per year', 'Median Monthly Sales_Total','Average Past Due Days Total', 'Last 12 months avg sales', 'Average Transaction Size', 'Std_Credit_Limit']]

    # Preprocessing: Scaling numeric features
    numeric_cols = df.select_dtypes(include=['number']).columns
    scaler = joblib.load(r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase2\scaler_final.pkl')
    df[numeric_cols] = scaler.transform(df[numeric_cols])
    df.update(df[numeric_cols])
    # Save preprocessed data to clipboard
    # df.to_clipboard()
    # return df, actual_credit_limits
    return df


@st.cache_data
def read_material_df():
    data = pd.read_pickle(os.path.join(data_folder,r'Model Data\Customer_monthwise_data_P.pkl'))
    data['Customer'] = data['Customer'].map(int)
    return data

# @st.cache_data
# def read_input_file(file_path = r'Data\PKL_Data_FINAL_v4\features_final.pkl'):
#     data = pd.read_pickle(os.path.join(data_folder,file_path))
#     data['Customer'] = data['Customer'].map(int)
#     return data

@st.cache_data
def read_material_master_df():
    data = pd.read_excel(os.path.join(data_folder,r'Brand Master.XLSX'))
    return data

@st.cache_data
def read_customer_master_df():
    data = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v3\customer_master_clean_v2')
    return data

@st.cache_data
def read_profitability_df():
    data = pd.read_pickle(os.path.join(data_folder,r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_combined.pkl'))
    data['Gross Margin%'] = data['Gross Margin%'] * 100
    return data

@st.cache_data
def read_hierarchy_df():
    data = pd.read_excel(os.path.join(data_folder,r'Hierarchy.XLSX'))
    return data

# Function to apply conditional formatting
def color_negative_red_positive_green(val):
    if isinstance(val,str):return 'color: black'
    color = 'green' if val > 0 else 'red' if val < 0 else 'black'
    return f'color: {color}'

def formatINR(number):
    s, *d = str(number).partition(".")
    r = ",".join([s[x-2:x] for x in range(-3, -len(s), -2)][::-1] + [s[-3:]])
    return "".join([r] + d)

def load_file(uploaded_file):
    """Loads a file based on its type."""
    if uploaded_file is not None:
        file_name = uploaded_file.name
        file_extension = file_name.split(".")[-1]

        if file_extension == "csv":
            return pd.read_csv(uploaded_file), file_name
        elif file_extension in ["xls", "xlsx"]:
            return pd.read_excel(uploaded_file), file_name
        elif file_extension == "pkl":
            return pickle.load(uploaded_file), file_name
        else:
            st.error("Unsupported file format.")
            return None, None
    return None, None



def format_indian_number(value):
    """Format number with Indian-style comma separators."""
    value = round(value, 2)  # Ensure two decimal places
    int_part, decimal_part = str(value).split(".") if "." in str(value) else (str(value), "00")
    
    # Handle Indian comma placement
    if len(int_part) > 3:
        int_part = int_part[:-3] + "," + int_part[-3:]
        int_part = ",".join([int_part[max(i-2, 0):i] for i in range(len(int_part), 0, -2)][::-1])
    
    return f"{int_part}.{decimal_part}"

# Importing Data 
standardized_df = read_standardized_CL()
# prob_df = read_predicted_proba_inc_dec_CL()
# material_df = read_material_df()
profitability_df = read_profitability_df()

#Importing Master Files 
# material_master_df = read_material_master_df()
# customer_master_df = read_customer_master_df()
# Hierarchy_df = read_hierarchy_df()

# Adding ASM code
# material_df['ASM CODE'] = material_df['Customer'].map(dict(zip(standardized_df['Customer'],standardized_df['ASM Code'])))
# ASM_Code_df = material_df[['Customer','State', 'City','ASM CODE','Year','Month','Sales_sum','Collection_sum']]
# ASM_Code_df  = ASM_Code_df.merge(Hierarchy_df[['Customer','Team','CM_ZHIRE-ZNAME','Supplier']],on="Customer",how='inner')
#Material NA
# standardized_df['Top 5 Materials purchased'].fillna("",inplace=True)
#Material Columns
# material_col = col_starts_with('Sales_Material_',material_df)

# Load an image from a file
# Display a banner image at the top of the app
image_path_Deloitte = r"Deloitte.png"
image_Deloitte = Image.open(image_path_Deloitte)
st.image(image_Deloitte, use_container_width=True) 

image_path_Safex = r"Safex.png"
image_Safex = Image.open(image_path_Safex)
# Before displaying model predictions
st.image(image_Safex, use_container_width=True)

# Streamlit UI
st.title('Credit Limit Optimization Model')

# Page navigation
page = st.sidebar.selectbox("Choose a page", ["Feature File Generation from Source Data", "Predictive Model for Customers", "Customer Details"])

# Page 1: Model Execution
if page == "Model to Generate Standardised Credit Limit":
    def load_and_merge_files(uploaded_files):
        """Reads multiple uploaded CSV/Excel files and merges them into a single DataFrame."""
        if not uploaded_files:
            return None

        dataframes = []
        for file in uploaded_files:
            if file.name.endswith('.csv'):
                df = pd.read_csv(file)
            elif file.name.endswith('.xlsx'):
                df = pd.read_excel(file)
            elif file.name.endswith('.pkl'):
                df = pd.read_pickle(file)
            else:
                st.warning(f"⚠️ Unsupported file format: {file.name}")
                continue  # Skip unsupported files

            dataframes.append(df)

        return pd.concat(dataframes, ignore_index=True) if dataframes else None


    # Streamlit UI
    st.title("Credit Limit Standardization")

    # Upload folder inputs
    # transactions_folder = st.text_input("📂 Enter Transactions Folder Path")
    # knockoffs_folder = st.text_input("📂 Enter Knock-offs Folder Path")
    # profitability_folder = st.text_input("📂 Enter Profitability Data Folder Path")
    transactions_files = st.file_uploader("📂 Upload Transaction Files", accept_multiple_files=True, type=['csv', 'xlsx', 'pkl'])
    knockoffs_files = st.file_uploader("📂 Upload Knock-offs Files", accept_multiple_files=True, type=['csv', 'xlsx', 'pkl'])
    profitability_files = st.file_uploader("📂 Upload Profitability Data Files", accept_multiple_files=True, type=['csv', 'xlsx', 'pkl'])
    # Button to load data
    if st.button("Combine Transactions"):
        transactions_df = load_and_merge_files(transactions_files)
        if transactions_df is not None:
            st.success("✅ Transactions Data Loaded Successfully!")

    if st.button("Combine Knock offs"):
        knockoffs_df = load_and_merge_files(knockoffs_files)
        if knockoffs_df is not None:
            st.success("✅ Knock-offs Data Loaded Successfully!")

    if st.button("Combine Profitability Data"):
        profitability_df = load_and_merge_files(profitability_files)
        if profitability_df is not None:
            st.success("✅ Profitability Data Loaded Successfully!")

    # Generate Credit Limit Standardization
    if st.button("🚀 Generate Standardized Credit Limit"):
        if transactions_df is not None and knockoffs_df is not None and profitability_df is not None:
            cl_std_file = transformation_phase2.standardization_CL(transactions_df, knockoffs_df, profitability_df)
            st.success("✅ Credit Limit Standardization Completed!")

            # Save file
            output_file = "Standardized_Credit_Limit.xlsx"
            cl_std_file.to_excel(output_file, index=False)

            # Provide download button
            with open(output_file, "rb") as f:
                st.download_button("📥 Download Standardized Credit Limit File", f, file_name="Standardized_Credit_Limit.xlsx")
        else:
            st.error("⚠️ Please upload all required data folders before generating the standardized credit limit.")
if page == "Feature File Generation from Source Data":
    # st.title("Upload Sales & Collections Data")
    # Sales_df, sales_filename = load_file(st.file_uploader("Upload Sales Data", type=["csv", "xlsx", "pkl"]))
    # Collections_df, collections_filename = load_file(st.file_uploader("Upload Collections Data", type=["csv", "xlsx", "pkl"]))
    
    # st.markdown("**Excecute to predict the correct credit limit**")
    # if st.button('Run Model'):
    #     st.write("Model Excecution Start")
        
    #     if Sales_df is not None and Collections_df is not None:
    #         output_df = main_transformation(Sales_df, Collections_df, step="Customer Standardised Credit Limit")
    #     else:
    #         output_df = "Files not uploaded"
    #     # Create a progress bar
    #     progress_bar = st.progress(0)
    #     from time import sleep
    #     # Simulate a long computation or process
    #     # Streamlit UI
    #     for percent_complete in range(101):
    #         sleep(0.01)  # Simulate time delay
    #         progress_bar.progress(percent_complete)
    #     try:
    #         output_df.to_excel(os.path.join(data_folder,r'Data\app_phase_2\Model_result.xlsx'),index=False)
    #     except:
    #         print("No file")
    #     st.write("Model Excecution completed")
        

        # Initialize session state variables
# Initialize session state variables

    # Initialize session state variables
    st.subheader("🚀 For New Customers")

    session_vars = {
        "sales_fetched": False,
        "collections_fetched": False,
        "customer_master_fetched": False,
        "status": "Waiting for data fetching...",
        "final_ready": False,
        "sales_df": None,
        "collections_df": None,
    }

    for key, default in session_vars.items():
        if key not in st.session_state:
            st.session_state[key] = default

    # Display status
    st.write(f"**📝 Status:** {st.session_state.status}")

    # --- Fetching Data ---
    st.divider()
    st.subheader("📥 Data Fetching")

    col1, col2, col3 = st.columns(3)

    # Fetch Customer Master Data
    with col1:
        if st.button("🗂 Fetch Customer Master Data", disabled=st.session_state.customer_master_fetched):
        # if st.button("🗂 Fetch Customer Master Data"):
            st.session_state.status = "Fetching Customer Master Data..."
            st.rerun()

        if st.session_state.status == "Fetching Customer Master Data...":
            progress_bar = st.progress(20)
            st.session_state.customer_master = fetch_data.fetch_and_filter_customer_data()
            # st.write(st.session_state.customer_master.shape)
            st.session_state.customer_master_fetched = True
            progress_bar.progress(60)
            st.session_state.status = "✅ Customer Master Data Fetched & Cleaned"
            st.session_state.customer_master.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\customer_master.csv", index = False)
            # st.write(st.session_state.customer_master.shape)
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            st.rerun()

    # Fetch Sales Data
    with col2:
        fetch_sales = st.button("📊 Fetch Sales Data", disabled=not st.session_state.customer_master_fetched or st.session_state.sales_fetched)
        # fetch_sales = st.button("📊 Fetch Sales Data")
        if fetch_sales:
            st.session_state.status = "Fetching Sales Data..."
            st.rerun()
        if st.session_state.status == "Fetching Sales Data...":
            progress_bar = st.progress(0)
            progress_bar.progress(20)
            sales_df = fetch_data.fetch_sales_data()
            # st.write(sales_df.shape)
            progress_bar.progress(40)
            sales_df = fetch_data.clean_sales_data(sales_df, st.session_state.customer_master)
            # sales_df = transformation_phase2.clean_sales(sales_df)
            # st.write(sales_df.shape)
            st.session_state.sales_df = sales_df
            st.session_state.sales_fetched = True
            st.session_state.status = "✅ Sales Data Cleaned"
            sales_df.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\sales_df.csv", index = False)
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            st.rerun()
            

    # Fetch Collections Data
    with col3:
        if st.button("💰 Fetch Collections Data", disabled=not st.session_state.customer_master_fetched or st.session_state.collections_fetched):
        # if st.button("💰 Fetch Collections Data"):
        # if st.button("💰 Fetch Collections Data"):
            st.session_state.status = "Fetching Collections Data..."
            st.rerun()

        if st.session_state.status == "Fetching Collections Data...":
            progress_bar = st.progress(20)
            collections_df = fetch_data.fetch_collections_data()
            # st.write(collections_df.shape)
            progress_bar.progress(40)
            collections_df = fetch_data.clean_collections(collections_df, st.session_state.customer_master)
            # collections_df = transformation_phase2.clean_collections(collections_df)
            # st.write(st.session_state.customer_master.shape)
            # st.write(collections_df.shape)
            st.session_state.collections_df = collections_df
            st.session_state.collections_fetched = True
            st.session_state.status = "✅ Collections Data Cleaned"
            collections_df.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\collections_df.csv", index = False)
            progress_bar.progress(100)
            time.sleep(0.5)
            progress_bar.empty()
            st.rerun()
            

    # Enable processing buttons if all data is fetched
    st.session_state.final_ready = (
        st.session_state.customer_master_fetched and
        st.session_state.sales_fetched and
        st.session_state.collections_fetched
    )

    # --- Processing Data ---
    st.divider()
    st.subheader("⚙️ Data Processing")

    col1, col2, col3 = st.columns(3)

    # Generate Transactions
    with col1:
        if st.button("🚀 Generate Transactions", disabled=not st.session_state.final_ready):
            st.session_state.transactions = transformation_phase2.trans_df_creation(st.session_state.sales_df, st.session_state.collections_df, st.session_state.customer_master)
            # st.session_state.transactions = transformation_phase2.trans_df_creation(st.session_state.sales_df, st.session_state.collections_df)
            st.success("✅ Transactions Data Generated!")
            # st.write(st.session_state.transactions.head(5))
            st.session_state.transactions.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\transactions.csv")
            

    # Generate Knockoffs
    with col2:
        # if "knock_offs" not in st.session_state:
        #     st.session_state.knock_offs = None

        if st.button("📌 Generate Knockoffs", disabled=(
                    not st.session_state.final_ready or "transactions" not in st.session_state)):
            st.session_state.knock_offs = transformation_phase2.knockoff_df_creation(st.session_state.transactions)
            st.success("✅ Knockoffs Data Generated!")
            # st.write(st.session_state.knock_offs.head(5))
            st.session_state.knock_offs.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\knockoffs.csv")

    # Generate Profitability
    with col3:
        # if "profit_df" not in st.session_state:
        #     st.session_state.profit_df = None
        if st.button("💹 Generate Profitability", disabled=(
                    not st.session_state.final_ready or 
                    "transactions" not in st.session_state or "knock_offs" not in st.session_state)):
            st.session_state.profit_df = transformation_phase2.profitability(st.session_state.transactions)
            st.success("✅ Profitability Data Generated!")
            # st.write(st.session_state.profit_df.head(5))
            st.session_state.profit_df.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\profitability.csv")
        

    # --- Standardization & Feature Engineering ---
    st.divider()
    st.subheader("📊 Standardization & Feature Engineering")

    col4, col5, col6 = st.columns(3)

    # # Generate Credit Limit Standardization
    # with col4:
    #     if st.button("🔢 Standardize Credit Limits", disabled=(
    #         not st.session_state.final_ready or 
    #         "transactions" not in st.session_state or st.session_state.transactions is None or st.session_state.transactions.empty or
    #         "knock_offs" not in st.session_state or st.session_state.knock_offs is None or st.session_state.knock_offs.empty or
    #         "profit_df" not in st.session_state or st.session_state.profit_df is None or st.session_state.profit_df.empty
    #     )):
    #         st.session_state.cl_std = transformation_phase2.standardization_CL(
    #             st.session_state.transactions, st.session_state.knock_offs, st.session_state.profit_df
    #         )
    #         st.success("✅ Credit Limits Standardized!")
    #         # st.write(st.session_state.cl_std.head(5))
    #         st.session_state.cl_std.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\cl_std.csv")

    # Generate Key Metrics
    with col4:
        if st.button("📈 Generate Key Metrics", disabled=  not st.session_state.final_ready or 
            "transactions" not in st.session_state or st.session_state.transactions is None or st.session_state.transactions.empty or
            "knock_offs" not in st.session_state or st.session_state.knock_offs is None or st.session_state.knock_offs.empty or
            "profit_df" not in st.session_state or st.session_state.profit_df is None or st.session_state.profit_df.empty ):
            # "cl_std" not in st.session_state or st.session_state.profit_df is None or st.session_state.profit_df.empty):
            st.session_state.key_metrics = key_metrics_v2.key_metrics(
                st.session_state.transactions, 
                st.session_state.knock_offs, 
                st.session_state.profit_df
            )
            st.success("✅ Key Metrics Generated!")
            st.session_state.key_metrics.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\key_metrics.csv")

    # Generate Features
    with col5:
        if st.button("✨ Generate Features File", disabled= not st.session_state.final_ready or 
            "transactions" not in st.session_state or st.session_state.transactions is None or st.session_state.transactions.empty or
            "knock_offs" not in st.session_state or st.session_state.knock_offs is None or st.session_state.knock_offs.empty or
            "profit_df" not in st.session_state or st.session_state.profit_df is None or st.session_state.profit_df.empty): #or
            # "cl_std" not in st.session_state or st.session_state.profit_df is None or st.session_state.profit_df.empty):
            st.session_state.features = features_phase2_v2.generate_features(
                st.session_state.transactions,
                st.session_state.knock_offs,
                st.session_state.profit_df,
                # st.session_state.cl_std,
                st.session_state.customer_master,
                st.session_state.key_metrics
            )
            st.success("✅ Features Generated!")
            csv = st.session_state.features.to_csv(index=False)
            st.session_state.features.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\features.csv")
            # csv = pd.DataFrame(csv)
            st.download_button("⬇️ Download Features as CSV", data=csv, file_name="features.csv", mime="text/csv")

# ASM_Customer = 0
# if page == "Predictive Model for Customers":

#     if 'features' not in st.session_state:
#         st.session_state.features = None
#     if 'predictions' not in st.session_state:
#         st.session_state.predictions = None
#     model_file = r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase2\Final_random_forest_model.pkl'
    
#     # transformer = r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase2\log1p_transformer.pkl'
#     if model_file is not None:
#         # log1p_transformer = load_transformer()
#         model = joblib.load(model_file)
#         # print(model)
#         if st.session_state.features is not None:
#             # input_df = st.session_state.features.drop(columns = 'State', axis = 1)
#             # input_df = st.session_state.features.drop(columns = 'Unnamed: 0', axis = 1)
#             user_input_df = st.session_state.features
#             refresh = st.button("Click Here to upload a Features File")
#             # st.write(user_input_df.columns)
#             if refresh:
#                 st.session_state.features = None
#                 # user_input_df = input_df
#         else:
#             input_df = st.file_uploader("Upload a features file, in CSV or Pickle format", type=["csv", "pkl"])
#             user_input_df = user_input_features(input_df)

#             # st.write(user_input_df.shape)
            
#         # st.write(st.session_state.features.shape)
#         # state_encoding = load_state_encoding()
#         st.write("Model Ready for Prediction!")
#         st.title("Credit Limit Prediction")
        
#         # actual_credit_limits = user_input_df.loc[:, 'Std_Credit_Limit']
#         # user_input_df.drop(columns = ['Std_Credit_Limit'], inplace = True)
        
#         if user_input_df is None:
#             st.error("Uploaded file is empty. Please upload a valid CSV or Pickle file.")
#             st.stop()

#         # st.write(user_input_df.columns)
#         # else:
#         #     user_input_df = user_input_df.apply(pd.to_numeric, errors='coerce')
            
#         # if user_input_df.isnull().values.any():
            
#         #     # st.error("Uploaded file contains missing or invalid values. Please check the input data.")
#         #     count = user_input_df.isnull().values.sum()
#         #     user_input_df = user_input_df.fillna(0)
            
#             # st.write(f"**Dealing with {count} Null Values**")
#             # st.stop()  # Stops execution if NaNs are found
#             # st.write(user_input_df.columns)
#             # st.write(len(user_input_df))

#         # # Apply log transformation
#         # try:
#         #     debug_data = np.log1p(user_input_df)
#         #     # st.write("Log1p Transformation Success")
#         # except Exception as e:
#         #     st.error(f"Error during np.log1p: {e}")
#         # # transformed_data = log1p_transformer.transform(np.log1p(user_input_df))
#         # numeric_cols = user_input_df.select_dtypes(include=['number']).columns
#         # categorical_cols = user_input_df.select_dtypes(include=['object']).columns
#         # transformed_data = np.log1p(user_input_df[numeric_cols])
#         # df = pd.concat([transformed_data, user_input_df[categorical_cols]], axis = 0)
#         if st.button("Predict Credit Limit"):
#             # Run the prediction model
#             user_input_df = user_input_df.astype(np.float32)
            
            
#             actual_credit_limits = user_input_df['Std_Credit_Limit']
#             user_input_df.drop(columns = 'Std_Credit_Limit', axis = 1, inplace = True)
#             st.write(user_input_df.columns)
#             # st.write(user_input_df.columns)
#             scaler = StandardScaler()
            
#             # user_input_df_numeric = user_input_df.select_dtypes(include=['number'])
#             user_input_df.fillna(0, inplace = True)
            
#             user_input_df.replace([np.inf, -np.inf], 0, inplace=True)
#             customer_ids = user_input_df['Customer']
#             user_input_df.drop(columns='Customer', axis=1, inplace=True)
#             user_input_df_scaled = scaler.fit_transform(user_input_df)
#             user_input_df_scaled = pd.DataFrame(user_input_df_scaled, columns=user_input_df.columns)
#             user_input_df_scaled['Customer'] = customer_ids.values
#             user_input_df_scaled.set_index('Customer', inplace=True)
            
#             # user_input_df.update(user_input_df)
#             pd.DataFrame(user_input_df_scaled).to_clipboard()
#             # user_input_df.update(user_input_df)

#             # state_regions = {
#             #     "North": ['Punjab', 'Rajasthan', 'Uttar Pradesh', 'Haryana', 'Himachal Pradesh', 'Uttarakhand', 'Delhi', 'Jammu & Kashmir', 'Chandigarh'],
#             #     "South": ['Andhra Pradesh', 'Telangana', 'Karnataka', 'Tamil Nadu', 'Kerala', 'Puducherry'],
#             #     "East": ['Bihar', 'West Bengal', 'Orissa', 'Meghalaya', 'Assam', 'Jharkhand', 'Tripura', 'Manipur'],
#             #     "West": ['Gujarat', 'Maharashtra', 'Goa', 'Dadra & Nagar Haveli'],
#             #     "Central": ['Madhya Pradesh', 'Chhattisgarh'],
#             # }
#             # # Flatten the dictionary to map each state to its region
#             # flat_state_mapping = {state: region for region, states in state_regions.items() for state in states}
#             # # Map the states to their respective regions
#             # user_input_df['State_Region'] = user_input_df['State'].map(flat_state_mapping)

            
            
#             # st.write(user_input_df_scaled.columns)
#             # user_input_df_scaled.set_index('Customer', inplace = True)
            
#             prediction = model.predict(user_input_df_scaled)
            
#             predictions_df = pd.DataFrame({
#                 'Customer': user_input_df_scaled.index,
#                 'Actual_Credit_Limit': actual_credit_limits,
#                 'Predicted_Credit_Limit': prediction
#             })
#             st.session_state.predictions = predictions_df
#             # if st.session_state.features is None:
#             #     df = pd.read_excel(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\app_data\actual_CL.xlsx')
#             #     df['Customer_ID'] = df['Customer_ID'].astype(str)
#             #     predictions_df['Actual Credit Limit'] = predictions_df['Customer_ID'].map(dict(zip(df['Customer_ID'], df['Actual_Credit_Limit'])))
            
            
#             st.success("PREDICTIONS COMPLETED")
#         # **Display the customer selection dropdown only if predictions are available**
#         if st.session_state.predictions is not None:
#             selected_customer = st.selectbox(
#                 "Select a Customer:", 
#                 st.session_state.predictions['Customer'].astype('int64').unique()
#             )

#             # **Show the predicted credit limit dynamically**
#             selected_credit_limit = st.session_state.predictions.loc[
#                 st.session_state.predictions['Customer'].astype('int64') == selected_customer, 'Predicted_Credit_Limit'
#             ].values[0]

#             st.write(f"**Predicted Credit Limit for {selected_customer}:** ₹{selected_credit_limit:,.2f}")            
#             pred_csv1 = st.session_state.predictions.to_csv(index=False)
#             st.download_button(
#                 label="Download Predictions as CSV",
#                 data=pred_csv1,
#                 file_name="credit_limit_predictions.csv",
#                 mime="text/csv",
#                 disabled = st.session_state.predictions.empty,
#                 key="download_features1"
#             )
if 'features' not in st.session_state:
    st.session_state.features = None
if 'predictions' not in st.session_state:
    st.session_state.predictions = None
    st.session_state.test_predictions = None
if page == "Predictive Model for Customers":
    # if 'features' not in st.session_state:
    #     st.session_state.features = None
    # if 'predictions' not in st.session_state:
    #     st.session_state.predictions = None
    model_file = r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase2\Random_forest_model_with20_features_SCL_20K.pkl'
    if model_file is not None:
        model = joblib.load(model_file)
        if st.session_state.features is not None:
            user_input_df = st.session_state.features
            # st.write(user_input_df.columns)
            # st.session_state.features.to_clipboard(index = False)
            user_input_df = preprocess_data_view(user_input_df)
            # user_input_df, actual_credit_limits = preprocess_data_view(user_input_df)
            # refresh = st.button("Click Here to upload a Features File")
            
            # if refresh:
            #     input_df = st.file_uploader("Upload a features file, in CSV or Pickle format", type=["csv", "pkl"])
            #     user_input_df = user_input_features(input_df)
                
            #     # st.write(user_input_df['Customer'])
            #     if user_input_df is not None:
            #         user_input_df = preprocess_data_test(user_input_df)
            #         st.write(user_input_df.shape)
            #     else:
            #         st.warning('Please Upload a CSV or Pickle File')
                
            #     go_back = st.button("Click here to Return")
            #     if go_back:
            #         st.session_state.page = "Predictive Model for Customers"
            
        else:
            input_df = st.file_uploader("Upload a features file, in CSV or Pickle format", type=["csv", "pkl"])
            st.session_state.test_predictions = user_input_features(input_df)
            user_input_df = st.session_state.test_predictions
            # st.write(user_input_df['Customer'])
            if user_input_df is not None:
                user_input_df = preprocess_data_test(user_input_df)
            else:
                st.warning('Please Upload a CSV or Pickle File')

        st.write("Model Ready for Prediction!")
        st.title("Credit Limit Prediction")
        if user_input_df is None:
            st.error("Uploaded file is empty. Please upload a valid CSV or Pickle file.")
            st.stop()

        if st.button("Predict Credit Limit"):
            # user_input_df.update(user_input_df.drop(columns = ['Customer']).astype(np.float32))
            # st.write(user_input_df.index)
            # actual_credit_limits = user_input_df['Std_Credit_Limit']
            # user_input_df.drop(columns='Std_Credit_Limit', axis=1, inplace=True)
            
            # scaler = StandardScaler()
            # user_input_df.fillna(0, inplace=True)
            # user_input_df.replace([np.inf, -np.inf], 0, inplace=True)
            
            # customer_ids = user_input_df['Customer']
            # user_input_df.drop(columns='Customer', axis=1, inplace=True)
            # user_input_df_scaled = scaler.fit_transform(user_input_df)
            # user_input_df_scaled = pd.DataFrame(user_input_df_scaled, columns=user_input_df.columns)
            # user_input_df_scaled['Customer'] = customer_ids.values
            # user_input_df_scaled.set_index('Customer', inplace=True)
            # user_input_df_scaled.to_clipboard()
            prediction = model.predict(user_input_df)
            
            if st.session_state.features is None:
                predictions_df = pd.DataFrame({
                    'Customer': user_input_df.index,
                    # 'Actual_Credit_Limit': actual_credit_limits,
                    'Predicted_Credit_Limit': prediction
                })
            
            else:
                
                predictions_df = pd.DataFrame({
                    'Customer': user_input_df.index,
                    # 'Actual_Credit_Limit': actual_credit_limits,
                    'Predicted_Credit_Limit': prediction
                })
            
            st.session_state.predictions = predictions_df
                
            st.success("PREDICTIONS COMPLETED")
        
        if st.session_state.predictions is not None:
            selected_customer = st.selectbox(
                "Select a Customer:", 
                st.session_state.predictions['Customer'].astype('int64').unique()
            )

            selected_credit_limit = st.session_state.predictions.loc[
                st.session_state.predictions['Customer'].astype('int64') == selected_customer, 'Predicted_Credit_Limit'
            ].values[0]

            st.write(f"**Predicted Credit Limit for {selected_customer}:** ₹{selected_credit_limit:,.2f}")            
            pred_csv1 = st.session_state.predictions.to_csv(index=False)
            # st.session_state.key_metrics.to_csv(r"C:\Users\Deloite\Desktop\Agcare-GIT\Data\dashboard_results\key_metrics.csv")
            st.download_button(
                label="Download Predictions as CSV",
                data=pred_csv1,
                file_name="credit_limit_predictions.csv",
                mime="text/csv",
                disabled=st.session_state.predictions.empty,
                key="download_features1"
            )
    
if page == "Customer Details":
    global STATE
    global CUSTOMER
    cutoff_date = pd.to_datetime('2024-03-31')
    start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')
    
    # view_new_customers = st.button("View New Customers")
    
    
    # file_paths = {
    #     'Customer_level_data': f'C:/Users/Deloite/Desktop/Agcare-GIT/Data/PKL_Data_FINAL_v4/cl_std_v2/cl_std_{STATE}.pkl',
    #     'trans_df': f'C:/Users/Deloite/Desktop/Agcare-GIT/Data/PKL_Data_FINAL_v4/transactions_clean_statewise/transactions_{STATE}.pkl',
    #     'profit_df': f'C:/Users/Deloite/Desktop/Agcare-GIT/Data/PKL_Data_FINAL_v4/profit_df_statewise/profit_df_{STATE}.pkl',
    #     'key_metrics_df': f'C:/Users/Deloite/Desktop/Agcare-GIT/Data/PKL_Data_FINAL_v4/key_metrics/key_metrics_{STATE}.pkl',
    #     'knock_df': f'C:/Users/Deloite/Desktop/Agcare-GIT/Data/PKL_Data_FINAL_v4/knock_df_statewise/knock_df_{STATE}.pkl',
    # }
    # try:
    #     Customer_level_data = st.session_state.key_metrics
    #     trans_df = st.session_state.transactions
    #     profit_df = st.session_state.profit_df
    #     features_df = st.session_state.features
    #     predictions_df = st.session_state.predictions
    #     knock_df = st.session_state.knock_offs
        
    # except FileNotFoundError as e:
    #     st.warning(f"Error loading data: ", {e})

    # st.session_state.states = trans_df['CM_State'].unique()
    # STATE = st.selectbox("Select State", options = states)
    # CUSTOMER = st.selectbox('Customer Number',options=Customer_level_data[Customer_level_data['State']==STATE]['Customer'].unique())
    
    # if view_new_customers:
    # if "transactions" not in st.session_state:
    #     st.error("🚨 No Data to show \n Please run the model first")
    #     st.button("Go Back")
    #     st.stop()
    #     if st.button("Go Back"):
    #         st.session_state.page = "Customer Details"
    # else:
    #     Customer_level_data = st.session_state.cl_std
    #     trans_df = st.session_state.transactions
    #     profit_df = st.session_state.profit_df
    #     key_metrics_df = st.session_state.key_metrics
    #     knock_df = st.session_state.knock_offs
    #     predictions = st.session_state.predictions
    #     features = st.session_state.features
    # st.session_state.states = list(st.session_state.features['State'].unique())
    # STATE = st.selectbox("Select State", st.session_state.states)
    # CUSTOMER = st.selectbox('Customer Number',options=Customer_level_data[Customer_level_data['State']==STATE]['Customer'].unique())

        # if states != st.session_state.states:
        #     st.session_state.states = states
        #     st.session_state.selected_state = states[0]
        #     st.rerun()
        #     STATE = st.selectbox("Select State", states, key = "selected_state")
        # if st.button("Go Back"):
        #     st.session_state.page = "Customer Details"

    if st.session_state.predictions is not None or st.session_state.features is not None:
        st.warning(f'The details in this page are based on the data captured between {start_cutoff.date().strftime('%d/%m/%y')} and {cutoff_date.date().strftime('%d/%m/%y')}')
        if st.session_state.test_predictions is not None:
            Customer_level_data = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\key_metrics_combined_2547_v3.pkl')
            states = Customer_level_data['State'].unique()
            STATE = st.selectbox("Select State", options = states)
            trans_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\trans_df_combined_2547.pkl')
            profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\profit_df_combined_2547.pkl')
            features_df = pd.read_csv(r'c:\Users\Deloite\Desktop\dashboard_demo\test_data_20_features.csv')
            current_credit_limits = pd.read_csv(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\current_credit_limits_2547.csv')
            features_df['Customer'] = features_df['Customer'].astype(str)
            current_credit_limits['Customer'] = current_credit_limits['Customer'].astype(str)
            features_df['Std_Credit_Limit'] = features_df['Customer'].map(dict(zip(current_credit_limits['Customer'], current_credit_limits['Actual'])))
            knock_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\knock_df_combined_2547.pkl')
            customers = features_df['Customer'].unique()
            predictions_df = st.session_state.predictions
            # customers = customers.astype(str)
            CUSTOMER = st.selectbox('Customer Number',options=Customer_level_data[Customer_level_data['State'] == STATE]['Customer'].unique())
            # st.write(Customer_level_data.shape)
        else:
            try:
                Customer_level_data = st.session_state.key_metrics
                trans_df = st.session_state.transactions
                profit_df = st.session_state.profit_df
                features_df = st.session_state.features
                predictions_df = st.session_state.predictions
                knock_df = st.session_state.knock_offs
            
            except FileNotFoundError as e:
                st.warning(f"Error loading data: ", {e})

            st.session_state.states = trans_df['CM_State'].unique()
            states = st.session_state.states
            STATE = st.selectbox("Select State", options = states)
            CUSTOMER = st.selectbox('Customer Number',options=Customer_level_data[Customer_level_data['State']==STATE]['Customer'].unique())
            st.header('Customer')
            st.write(f"Selected Customer: {CUSTOMER}")

        tab1, tab2, tab3, tab4, tab5, tab6, tab7 = st.tabs(["💳 Credit Limit", "🌏 Demographic", "👩‍🌾 Sales", "💰 Collections", "🛒 Material", "📈 Running Outstanding", "📜 GST Status"])
        # Tab 1: 💳 Credit Limit
        with tab1:
            # if view_new_customers:
            #     # predictions = st.session_state.predictions
            #     if "predictions" in st.session_state:
            #         # states = list(st.session_state.features['State'].unique())
            #         # STATE = st.selectbox("Select State", states)
                    
            #         common_customers = set(st.session_state.features['Customer']).intersection(set(st.session_state.predictions['Customer']))
            #         filtered_features = st.session_state.features[st.session_state.features['Customer'].isin(common_customers)]
            #         CUSTOMER = st.selectbox('Customer Number',options=filtered_features[filtered_features['State']==STATE]['Customer'].unique())    
            #         tab1_1,tab1_2 = st.tabs(['Credit Limit in lakhs','Credit Limit'])
            #         with tab1_2:
            #             st.title("💳 Credit Limit")
            #             st.header('Suggested Credit Limit')
            #             standardized_CL = round(st.session_state.predictions[st.session_state.predictions['Customer']==str(CUSTOMER)]['Predicted_Credit_Limit'].values[0])
            #             st.subheader(f"Suggested credit limit: **₹ {formatINR(standardized_CL)}**")
            #             round_off_standardized_CL = round(standardized_CL,-4)
            #             st.subheader(f"Approximate Suggested credit limit: **₹ {formatINR(round_off_standardized_CL)}**")
            #             st.header('Current Credit Limit')
            #             current_CL = Customer_level_data[Customer_level_data['Customer']==str(CUSTOMER)]['Current Credit Limit'].values[0]
            #             st.subheader(f"Current credit limit: **₹ {formatINR(current_CL)}**")
            #         with tab1_1:
            #             st.title("💳 Credit Limit (In Lakhs)")
            #             # Add a +/- button to adjust decimal places
            #             col1,col2 = st.columns(2)
            #             with col1:
            #                 decimal_places = st.number_input('Select Decimal Places', min_value=1, max_value=4, value=2, step=1)

            #             # Convert values to lakhs
            #             standardized_CL_lakhs = standardized_CL / 1e5
            #             round_off_standardized_CL_lakhs = round_off_standardized_CL / 1e5
            #             current_CL_lakhs = current_CL / 1e5

            #             # Display values in lakhs with adjustable decimal places
            #             st.header('Suggested Credit Limit (In Lakhs)')
            #             st.subheader(f"Suggested credit limit: **₹ {standardized_CL_lakhs:.{decimal_places}f} Lakhs**")

            #             st.header('Current Credit Limit (In Lakhs)')
            #             st.subheader(f"Current credit limit: **₹ {current_CL_lakhs:.{decimal_places}f} Lakhs**")
            #     # else:
            #         # STATE = st.selectbox("Select State", states)
            #         # CUSTOMER = st.selectbox('Customer Number',options=Customer_level_data[Customer_level_data['State']==STATE]['Customer'].unique())
            # else:
            
            tab1_1,tab1_2 = st.tabs(['Credit Limit in lakhs','Credit Limit'])
            with tab1_2:
                
                st.title("💳 Credit Limit")
                st.header('Predicted Credit Limit')
                # put predicted credit limit!!!
                predictions_df['Customer'] = predictions_df['Customer'].astype(str)
                features_df['Customer'] = features_df['Customer'].astype(str)
                # st.write(predictions_df)
                # st.write(CUSTOMER)
                # st.write(predictions_df[predictions_df['Customer']==CUSTOMER]['Predicted_Credit_Limit'])
                # st.write(predictions_df[predictions_df['Customer']==CUSTOMER])
                standardized_CL = round(predictions_df[predictions_df['Customer']==CUSTOMER]['Predicted_Credit_Limit'].values[0])
                st.subheader(f"Predicted credit limit: **₹ {formatINR(standardized_CL)}**")
                # round_off_standardized_CL = round(standardized_CL,-4)
                # st.subheader(f"Approximate predicted credit limit: **₹ {formatINR(round_off_standardized_CL)}**")
                # st.header('Current Credit Limit')
                # current_CL = (features_df[features_df['Customer']==str(CUSTOMER)]['Std_Credit_Limit'].values[0])
                # st.subheader(f"Current credit limit: **₹ {formatINR(round(current_CL, 3))}**")
            with tab1_1:
                st.title("💳 Credit Limit (In Lakhs)")
                # st.write(predictions_df['Predicted_Credit_Limit'])
                # Add a +/- button to adjust decimal places
                col1,col2 = st.columns(2)
                with col1:
                    decimal_places = st.number_input('Select Decimal Places', min_value=1, max_value=4, value=2, step=1)

                # Convert values to lakhs
                standardized_CL_lakhs = standardized_CL / 1e5

                # standardized_CL_lakhs_value = standardized_CL_lakhs if isinstance(standardized_CL_lakhs, pd.Series) else standardized_CL_lakhs
                # round_off_standardized_CL_lakhs = round_off_standardized_CL / 1e5
                # current_CL_lakhs = current_CL / 1e5

                # Display values in lakhs with adjustable decimal places
                st.header('Predicted Credit Limit (In Lakhs)')
                st.subheader(f"Predicted credit limit: **₹ {standardized_CL_lakhs:.{decimal_places}f} Lakhs**")

                # st.header('Current Credit Limit (In Lakhs)')
                # st.subheader(f"Current credit limit: **₹ {current_CL_lakhs:.{decimal_places}f} Lakhs**")
        
        # Tab 2: 🌏 Demographic
        with tab2:
            st.header("🌏 Demographic")
            # st.subheader(f"Name: {Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)]['Name'].values[0]}")
            state = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)]['State'].values[0]
            global asm_code
            # filtered_data = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)][['Tenure','last activity']]
            filtered_data = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)][['Tenure','last activity', 'first activity']]
            trans_df.rename(columns = {'CM_ASM Code': 'ASM Code'}, inplace = True)
            filtered_data_2 = features_df[features_df.Customer==str(CUSTOMER)][['ASM Code']] if 'ASM Code' in features_df.columns else trans_df[trans_df.Customer==str(CUSTOMER)][['ASM Code']]
            filtered_data_3 = trans_df[trans_df.Customer==str(CUSTOMER)][['CM_Start Date on']]
            if not filtered_data.empty or not filtered_data_2.empty or not filtered_data_3.empty:
                tenure = filtered_data['Tenure'].values[0] if 'Tenure' in filtered_data.columns else None
                first_activity = pd.to_datetime(filtered_data['first activity'].values[0]) if 'first activity' in filtered_data.columns else None
                asm_code = filtered_data_2['ASM Code'].values[0] 
                created_date = pd.to_datetime(filtered_data_3['CM_Start Date on'].values[0]) if 'CM_Start Date on' in filtered_data_3.columns else None
                last_activity = pd.to_datetime(filtered_data['last activity'].values[0]) if 'last activity' in filtered_data.columns else None
            else:
                tenure, asm_code, created_date, last_activity = None, None, None, None
                st.warning(f"No data found for Customer: {CUSTOMER}")
            ASM_Customer = asm_code
            # Define constants for days in a year and days in a month (approximate)
            days_in_year = 365
            days_in_month = 30
            total_days = tenure * days_in_year
            # Calculate years, months, and days
            # Ensure tenure is a number
            # tenure = int(tenure) # If tenure is an integer string
            # tenure = float(tenure)  # If tenure might have decimal values

            years, remainder = divmod(total_days, days_in_year)

            months, days = divmod(remainder, days_in_month)
            days = int(days)
            
            # Create a string to display the tenure in Years, Months, and Days
            
            col1,col2 = st.columns(2)
            with col1:
                st.subheader(f'ASM Code: {asm_code}')
                st.subheader(f'State: {state}')
                # st.subheader(f'City: {city}')
                
            with col2:
                st.subheader(f"Creation Date: {created_date.strftime('%d/%m/%y')}")
                st.subheader(f"First Activity Date: {first_activity.strftime('%d/%m/%y')}")
                st.subheader(f"Last Activity Date: {last_activity.strftime('%d/%m/%y')}")
                # tenure_string = f"{years} Years, {months} Months, {days} Days"
                # st.subheader(f'Tenure: {tenure_string}')
        
        # Tab 3: 👩‍🌾 Sales
        with tab3:
            # st.write(Customer_level_data.columns)
            st.header("👩‍🌾 Sales")
            avg_bill_settlement_days,last_12_month_min_sales = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)][['avg_bill_settlement_days', 'Last 12 months min sales']].values[0]
            avg_yearly_sales,last_12_month_max_sales = features_df[features_df.Customer==str(CUSTOMER)][['Avg Yearly Sales','Last 12 months max sales']].values[0]
            total_invoices = int(Customer_level_data.loc[Customer_level_data['Customer'] == str(CUSTOMER), 'Number of invoices Total'].fillna(0).values[0] if not Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)].empty else 0)
            outstandings_total = Customer_level_data.loc[Customer_level_data['Customer'] == str(CUSTOMER), 'no_of_outstanding_invoices_total'].fillna(0).values
            outstandings_total = int(outstandings_total[0]) if outstandings_total.size > 0 else 0

            # avg_transactions_size = trans_df[(trans_df['Customer'] == str(CUSTOMER)) & (trans_df['S_Billing Type'] == 'ZINV')] \
                # .groupby('S_Billing Document')['S_Total Sales'].sum().mean()
            
            # avg_transactions_size = round(avg_transactions_size, 2) if not pd.isna(avg_transactions_size) else 0
            avg_transaction_size = Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)]['avg_transaction_size'].values[0]
            opening_bal = trans_df[trans_df['Customer'] == str(CUSTOMER)]['CM_Opening Balances'].iloc[0]
            col1,col2 = st.columns(2)
            with col1:
                # st.write(f"Average Daily Sales: {round(avg_daily_sales,2)}")
                st.markdown(f"**Average Yearly Sales:** ₹ {round(avg_yearly_sales,2)}")
                st.markdown(f"**Last 12 months max sales:** ₹ {round(last_12_month_max_sales,2)}")
                st.markdown(f"**Last 12 months min sales:** ₹ {round(last_12_month_min_sales,2)}")
                st.markdown(f"**Opening Balance:** ₹ {opening_bal}")
            with col2:
                st.markdown(f"**Total Number of Invoices:** {int(total_invoices)}")
                st.markdown(f'**Average Number of Invoices:** {Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)]['avg_number_of_invoices'].values[0]}')
                st.markdown(f"**Number of Outstanding Invoices:** {int(outstandings_total)}")
                st.markdown(f"**Average Transaction Size:** ₹ {int(avg_transaction_size)}")
            
            st.subheader('Sales Trend', divider='blue')
            
            # Prepare the Sales DataFrame
            # trans_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_Punjab.pkl')
            # knock_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_Punjab.pkl')
            sale_df = trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby(['Customer', 'Year', 'Month'])['S_Total Sales'].sum().reset_index()
            sale_df = sale_df[sale_df.Customer == str(CUSTOMER)][['Year', 'Month', 'S_Total Sales']]
            sale_df.rename(columns={'S_Total Sales': 'Total Sales'}, inplace=True)
            sale_df["Month-Year"] = sale_df["Month"].astype(str) + "-" + sale_df["Year"].astype(str)
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Convert the Month column to a categorical type with the specified order
            sale_df['Month'] = pd.Categorical(sale_df['Month'], categories=month_order, ordered=True)
            sale_df.sort_values(by=['Year', 'Month'], inplace=True)
            
            
            fig = px.line(
                sale_df,
                x='Month-Year',
                y='Total Sales',
                title=f"Sales Trend for {CUSTOMER}",
                labels={'value': 'Amount', 'Month-Year': 'Month-Year'},
                # tooltip=[alt.Tooltip('Total Sales', format=',.0f'), alt.Tooltip('Total Collections', format=',.0f')],
            )
            # fig.update_yaxes(tickvals=sale_df['Total Sales'], ticktext=[indian_numbering_format(x) for x in sale_df['Total Sales']])
            fig.update_traces(
                hovertemplate='%{y:,.0f}'  # No decimal places
            )

            # Update the layout for better appearance
            fig.update_layout(
                xaxis_title='Month-Year',
                yaxis_title='Sales Amount',
                legend_title='Legend',
            )

            # Show the Plotly graph in Streamlit
            st.plotly_chart(fig)

            # Show DataFrame in Streamlit
            sales_csv = sale_df.to_csv(index=False)
            st.write("Sales Table")
            sale_df['Year'] = sale_df['Year'].map(str)
            st.dataframe(sale_df.iloc[::-1], hide_index=True,width=700, height=400)
            st.download_button("Download Sales",data=sales_csv, file_name="sales.csv", mime="text/csv")
        with tab4:
            st.header("💰 Collections")
            avg_bill_settlement_days = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)]['avg_bill_settlement_days'].values[0]
            MF_PT_used = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)]['max_payment_term'].values[0]
            avg_yearly_collection = features_df[features_df.Customer==str(CUSTOMER)]['Avg Yearly Collections'].values[0]
            collections_efficiency = features_df[features_df['Customer'] == str(CUSTOMER)][['Collection Efficiency%']]
            debtor_days = Customer_level_data.loc[Customer_level_data['Customer'] == str(CUSTOMER), 'Debtor Days'].fillna(0).values
            debtor_days = int(debtor_days[0]) if len(debtor_days) > 0 else 0

            # st.write(f"Average Daily Sales: {round(avg_daily_sales,2)}")
            st.markdown(f"**Average Yearly Collection:** ₹ {round(avg_yearly_collection,2)}")
            st.markdown(f"**Average Bill Settlement Days:** {int(avg_bill_settlement_days)}")
            st.markdown(f"**Most Frequent Payment terms used:** {MF_PT_used}")
            st.markdown(f"**Debtor Days:** {int(debtor_days)}")
            st.markdown(f"**Collection Efficiency:** {round(collections_efficiency['Collection Efficiency%'].values[0],2)} %")
            st.subheader('Collections Trend', divider='blue')
            
            # Prepare the Collections DataFrame
            collect_df = trans_df[trans_df['C_Document Type'] == 'DZ'].groupby(['Customer', 'Year', 'Month'])['C_Amount in Balance Transaction Currency'].sum().reset_index()
            collect_df = collect_df[collect_df.Customer == str(CUSTOMER)][['Year', 'Month', 'C_Amount in Balance Transaction Currency']]
            collect_df['C_Amount in Balance Transaction Currency'] = collect_df['C_Amount in Balance Transaction Currency'].abs()
            collect_df.rename(columns={'C_Amount in Balance Transaction Currency': 'Total Collection'}, inplace=True)
            collect_df["Month-Year"] = collect_df["Month"].astype(str) + "-" + collect_df["Year"].astype(str)
            month_order = ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec']

            # Convert the Month column to a categorical type with the specified order
            collect_df['Month'] = pd.Categorical(collect_df['Month'], categories=month_order, ordered=True)
            collect_df.sort_values(by=['Year', 'Month'], inplace=True)
            
            fig = px.line(
                collect_df,
                x='Month-Year',
                y='Total Collection',
                title=f"Collections Trend for {CUSTOMER}",
                labels={'value': 'Collections Amount', 'Month-Year': 'Month-Year'},
                # tooltip=[alt.Tooltip('Total Sales', format=',.0f'), alt.Tooltip('Total Collections', format=',.0f')],
            )
            fig.update_traces(
                hovertemplate='%{y:,.0f}'  # No decimal places
            )

            # Update the layout for better appearance
            fig.update_layout(
                xaxis_title='Month-Year',
                yaxis_title='Collections Amount',
                legend_title='Legend',
            )

            # Show the Plotly graph in Streamlit
            st.plotly_chart(fig)
            collect_csv = collect_df.to_csv(index=False)
            st.write("Collection Table")
            collect_df['Year'] = collect_df['Year'].map(str)
            st.dataframe(collect_df.iloc[::-1],hide_index=True,width=700, height=400)
            st.download_button("Download Collections",data=collect_csv, file_name="collections.csv", mime="text/csv")
        # Tab 4: "🛒 Material"
        with tab5:
            st.header("🛒 Material")
            Customer_level_data['Top 5 Materials purchased'].fillna("",inplace=True)
            top_5_materials = Customer_level_data[Customer_level_data.Customer==str(CUSTOMER)]['Top 5 Materials purchased'].values[0]
            if (top_5_materials!=""):
                top_5_materials = top_5_materials.split(", ")

                # Format the output with 1-based indexing
                formatted_materials = [f"{i+1}. {mat}" for i, mat in enumerate(top_5_materials)]

                # Display the formatted list
                st.subheader("Top 5 Materials Purchased:")
                st.write("(Based on Total Sales)")
                # st.write("\n".join(formatted_materials))
                
                material_summary = (
                    trans_df[trans_df['Customer'] == str(CUSTOMER)].groupby('S_Material', as_index=False)
                    .agg(Total_Sales=('S_Total Sales', 'sum'))
                )

                # Filter for only the top 5 materials
                top_5_df = material_summary[material_summary['S_Material'].isin(top_5_materials)].reset_index(drop = True)

                # Sort in descending order of occurrences (optional)
                
                # Filter for only the top 5 materials
                top_5_df = material_summary[material_summary['S_Material'].isin(top_5_materials)]
                top_5_df = top_5_df.sort_values(by='Total_Sales', ascending=False)
                top_5_df.reset_index(drop = True, inplace = True)
                top_5_df.index +=1
                top_5_df.rename(
                    columns = {
                        'S_Material': 'Material Code',
                        'Total_Sales': 'Total Sales'
                    },
                    inplace = True
                )
                # Display as a DataFrame in Streamlit
                st.dataframe(top_5_df)
                # mat_df = trans_df[trans_df.Customer==str(CUSTOMER)][['Year']+list(top_5_materials)].groupby('Year').sum()
                # mat_df.index = list(map(lambda x: x.replace('Sales_Material_',""),mat_df.index))
                # mat_df = mat_df.reset_index().rename(columns= {'index':"Material"})
                # mat_df.index = pd.RangeIndex(start=1, stop=len(mat_df)+1, step=1)
                # st.dataframe(mat_df.style.applymap(color_negative_red_positive_green),width=900, height=300)            
            else:
                st.write("No Material Found")
            
            # st.subheader("Materials")
            # mat_all_df = Customer_level_data[Customer_level_data.Customer==CUSTOMER][['Year'] + material_col].groupby('Year').sum().T
            # mat_all_df.reset_index(inplace=True)
            # mat_all_df.rename(columns={'index':'Matrial'},inplace=True)
            # mat_all_df['Matrial'] = mat_all_df['Matrial'].str.replace('Sales_Material_',"")
            # mat_all_df = pd.merge(mat_all_df,Customer_level_data[['Main Material','Brand','Technical','Material Description']],left_on='Matrial',right_on='Main Material',how="inner" ).drop_duplicates('Main Material',keep='first')
            # mat_all_df.drop(columns=['Main Material'],inplace=True)
            # mat_all_df.reset_index(drop=True,inplace=True)
            # mat_all_df = mat_all_df[list(filter(lambda i: isinstance(i,str) , mat_all_df.columns)) + list(filter(lambda i: isinstance(i,int) , mat_all_df.columns))]
            # mat_all_df.index = pd.RangeIndex(start=1, stop=len(mat_all_df)+1, step=1)
            # # Apply the conditional formatting
            # st.dataframe(mat_all_df.style.applymap(color_negative_red_positive_green))

            
        # Tab 5: "📈 Running Outstanding"
        with tab6:
            st.header('📈 Running Outstanding')
            # Utilization Number
            def utilization_calculate(RO,CL):
                ''' 
                    RO = Running Outstanding
                    CL = Credit Limit
                '''
                if RO<=0:
                    RO=0
                if CL <= 0:
                    return 0
                return 100*(RO/CL)
            # Filter DataFrame for the selected customer
            # utilization_df = Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)][['Year', 'Month', 'Max utilization %', 'Avg Utilization %','RO_mean', 'RO_max']]
            trans_df['Credit Limit'] = pd.to_numeric(trans_df['Credit Limit'])
            trans_df['Running Outstanding'] = pd.to_numeric(trans_df['Running Outstanding'])
            trans_df['Utilization %'] =  trans_df[['Running Outstanding','Credit Limit']].apply(lambda row: utilization_calculate(row['Running Outstanding'],row['Credit Limit']), axis=1)
            # utilization_df = Customer_level_data[Customer_level_data['Customer'] == str(CUSTOMER)][['Year', 'Month', 'Max utilization %', 'Avg Utilization %','RO_mean', 'RO_max']]
            max_util = trans_df[trans_df['Customer'] == str(CUSTOMER)].groupby(['Year', 'Month'])['Utilization %'].max().reset_index() 
            max_util.rename(columns={'Utilization %': 'utilization_max'}, inplace = True)
            avg_util = trans_df[trans_df['Customer'] == str(CUSTOMER)].groupby(['Year', 'Month'])['Utilization %'].mean().reset_index()
            avg_util.rename(columns={'Utilization %': 'utilization_avg'}, inplace = True)
            RO_Mean = trans_df[trans_df['Customer'] == str(CUSTOMER)].groupby(['Year', 'Month'])['Running Outstanding'].mean().reset_index()
            RO_Mean.rename(columns={'Running Outstanding': 'RO_mean'}, inplace = True)
            RO_MAX = trans_df[trans_df['Customer'] == str(CUSTOMER)].groupby(['Year', 'Month'])['Running Outstanding'].max().reset_index()
            RO_MAX.rename(columns={'Running Outstanding': 'RO_max'}, inplace = True)
            RO_MAX['Date'] = pd.to_datetime(RO_MAX['Year'].astype(str) + RO_MAX['Month'], format='%Y%b')
            RO_Mean['Date'] = pd.to_datetime(RO_Mean['Year'].astype(str) + RO_Mean['Month'], format='%Y%b')
            # utilization_df_month = utilization_df['Month'].values[-1]
            # st.subheader("Max Utilization%", divider='blue')
            # st.text(f"Current utilization:")
            # st.text(f"Last 3 month max utilization: {round(max_util.tail(3)['Utilization %'].max(),2)}%")
            # st.text(f"Last 6 month max utilization: {round(max_util.tail(6)['Utilization %'].max(),2)}%")
            # st.text(f"Last 12 month max utilization: {round(max_util.tail(12)['Utilization %'].max(),2)}%")
            # st.text(f"Max utilization throughout the tenure: {round(max_util['Utilization %'].max(),2)}%")
            # Convert 'Month' and 'Year' to datetime format
            max_util['Date'] = pd.to_datetime(max_util['Year'].astype(str) + max_util['Month'], format='%Y%b')
            avg_util['Date'] = pd.to_datetime(avg_util['Year'].astype(str) + avg_util['Month'], format='%Y%b')
            # Sort the data by Date
            max_util = max_util.sort_values(by='Date')

            # Create a Plotly line chart
            # fig = px.line(max_util, x='Date', y=['Utilization %'],
            #             labels={'value': 'Utilization', 'Date': 'Timeline'},
            #             title='Utilization')

            # # Update the layout for better appearance
            # fig.update_layout(xaxis_title='Date',
            #                 yaxis_title='Utilization',
            #                 legend_title='Utilization Metrics')

            # # Show the plot in Streamlit
            # st.plotly_chart(fig)
            
            RO_MAX.sort_values(by = 'Date', inplace = True)
            # st.dataframe(RO_MAX)
            st.subheader(f"Max Running Outstanding (overall tenure): ₹ {round(RO_MAX['RO_max'].max(),2)}")
            st.subheader(f"Current Max Running Outstanding (monthly) : ₹ {round(RO_MAX['RO_max'].values[-1],2)}")
            # st.subheader(f"Maximum running outstanding Monthwise")
            utilization_df = max_util[['Year', 'Month', 'utilization_max']] \
            .merge(avg_util[['Year', 'Month', 'utilization_avg']], on=['Year', 'Month'], how='outer') \
            .merge(RO_MAX[['Year', 'Month', 'RO_max']], on=['Year', 'Month'], how='outer') \
            .merge(RO_Mean[['Year', 'Month', 'RO_mean']], on=['Year', 'Month'], how='outer').reset_index()
            # Create a Plotly line chart
            fig_ro = px.line(RO_MAX, x='Date', y=['RO_max'],
                        labels={'value': 'Running Outstanding', 'Date': 'Timeline'},
                        title='Maximum Running Outstanding - Monthwise',
                        # tooltip=[alt.Tooltip('RO_max', format=',.0f')]
                        )
            fig_ro.update_traces(
                hovertemplate='%{y:,.0f}'  # No decimal places
            )
            # Update the layout for better appearance
            fig_ro.update_layout(xaxis_title='Date',
                            yaxis_title='Running Outstanding',
                            legend_title='Running Outstanding')
            # Show the plot in Streamlit
            st.plotly_chart(fig_ro)
            # st.subheader(f"Max Running Outstanding: {round(utilization_df['RO_max'].values[-1],2)}")
            # st.subheader(f"Maximum running outstanding")
            
            # st.text(f"current maximum running outstanding (monthly) : {round(utilization_df['RO_max'].values[-1],2)}")
            # Create a Plotly line chart
            # fig_ro = px.line(utilization_df, x='Date', y=['RO_max'],
            #             labels={'value': 'Running Outstanding', 'Date': 'Timeline'},
            #             title='Running outstanding over the year')

            # Update the layout for better appearance
            # fig_ro.update_layout(xaxis_title='Date',
            #                 yaxis_title='Running Outstanding',
            #                 legend_title='Running Outstanding')

            # Show the plot in Streamlit
            # st.plotly_chart(fig_ro)
            
            # utilization_df = max_util.sort_values(by = 'Date',ascending=False)
            # utilization_df = utilization_df[['Year', 'Month', 'utilization_max', 'utilization_avg','RO_mean', 'RO_max']].reset_index(drop=True)
            # utilization_df = utilization_df[['utilization_max', 'utilization_avg']].reset_index(drop=True)
            
            # st.dataframe(utilization_df,hide_index=True,width=900, height=400)
        # with tab6:
        #     # Subheader for the section with a blue divider
        #     st.subheader("💰 Customer Profitability", divider='blue')
        #     profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Punjab.pkl')
        #     # Filter DataFrame for the selected customer
        #     filtered_df = profit_df[profit_df['Customer'] == str(CUSTOMER)]
        #     filtered_df['Fiscal Year'] = filtered_df['Fiscal Year'].astype(str)
        #     filtered_df.sort_values(by='Fiscal Year', inplace=True)
        #     # filtered_df = filtered_df.dropna(subset=['Gross Margin%'])
            
        #     fig_line = go.Figure()

        #     # Add line trace
        #     fig_line.add_trace(
        #         go.Scatter(
        #             x=filtered_df['Fiscal Year'], 
        #             y=filtered_df['Gross Margin%'], 
        #             mode='lines+markers',  # Line + dots on each point
        #             name='Gross Margin %',
        #             marker=dict(color='blue'),
        #             line=dict(width=2)  # Adjust line thickness if needed
        #         )
        #     )
        #     # Update layout for the line chart
        #     fig_line.update_layout(
        #         title=f'Gross Margin % for Customer: {CUSTOMER}',
        #         xaxis_title='Fiscal Year',
        #         yaxis_title='Gross Margin %',
        #         legend_title='Metrics',
        #         yaxis=dict(tickfont=dict(color='blue')),
        #     )
            
        #     # Show the bar chart in Streamlit
        #     st.plotly_chart(fig_line)
        #     filtered_df = filtered_df.dropna(subset=['Net Sale Amount', 'Net Return'])
        #     fig = px.line(
        #         filtered_df, 
        #         x='Fiscal Year', 
        #         y=['Net Sale Amount', 'Net Return'], 
        #         title='Net Sale Amount and Net Returns Over Fiscal Years',
        #         markers=True  # Adds dots at data points
        #     )

        #     # Customize layout
        #     fig.update_layout(
        #         xaxis_title='Fiscal Year',
        #         yaxis_title='Amount',
        #         legend_title='Metrics'
        #     )

        #     # Display the chart in Streamlit
        #     st.plotly_chart(fig)
            
        #     # filtered_df = filtered_df[['Fiscal Year', 'Net Return', 'Net Sale Amount','Rate Diff', 'COGS','Gross Margin','Gross Margin%','ZINV', 'ZIRE', 'ZWCR','ZCCL', 'ZCSI', 'ZICL', 'ZICR', 'ZIDG','ZIDR', 'ZIFC']].sort_values('Fiscal Year').reset_index(drop=True)
            
        #     st.dataframe(filtered_df)
        # with tab6:
        #     st.header("📊 ASM Details")
        #     trans_df = trans_df.loc[:, ~trans_df.columns.duplicated()]
        #     ASM_Code_df = trans_df[trans_df['Customer'] == str(CUSTOMER)][['Customer','Year', 'Month', 'Date', 'CM_Name 1', 'CM_State', 'CM_City','CM_ASM Code','CM_Team Details', 'CM_Supplier', 'CM_ZHIRE-ZNAME', 'S_Total Sales', 'C_Amount in Balance Transaction Currency']]
            
        #     ASM_CODE = asm_code
        #     st.subheader(f"Selected ASM Code: {ASM_CODE}",divider='blue')
        #     st.subheader(f"Name: {ASM_Code_df[ASM_Code_df['CM_ASM Code']==ASM_CODE]['CM_ZHIRE-ZNAME'].values[0]}",divider='blue')
        #     # ASM_Code_df['Collection_sum'] = ASM_Code_df['Collection_sum'].abs()
        #     # ASM_Code_df['Date'] = pd.to_datetime(ASM_Code_df['Year'].astype(str) + ASM_Code_df['Month'], format='%Y%b')
        #     # # Sort the data by Date
        #     # ASM_Code_df = ASM_Code_df.sort_values(by='Date')
        #     # # Create a Plotly line chart
        #     # fig = px.line(ASM_Code_df[ASM_Code_df['ASM CODE']==ASM_CODE], x='Date', y=['Sales_sum','Collection_sum'],
        #     #             labels={'value': 'Sales', 'Date': 'Date'},
        #     #             title='Sales')

        #     # # Update the layout for better appearance
        #     # fig.update_layout(xaxis_title='Date',
        #     #                 yaxis_title='Sales',
        #     #                 legend_title='Sales Trend')
        #     # # Show the plot in Streamlit
        #     # st.plotly_chart(fig)
            
        #     # Create a Plotly line chart
            
        #     # fig = px.bar(ASM_Code_df[ASM_Code_df['ASM CODE']==ASM_CODE], x='Date', y=['Sales_sum','Collection_sum'],
        #     #             labels={'value': 'Collection', 'Date': 'Timeline'},
        #     #             title='Collection')

        #     # # Update the layout for better appearance
        #     # fig.update_layout(xaxis_title='Date',
        #     #                 yaxis_title='Sales & Collection',
        #     #                 legend_title='Sales & Collection Trend')
            
        #     # # Show the plot in Streamlit
        #     # st.plotly_chart(fig)
            
        #     col1,col2 = st.columns(2)
        #     ASM_STATE = STATE
        #     Customer_Name = ASM_Code_df['CM_Name 1'].values[0]
        #     st.write(f'Selected State : {ASM_STATE}')
        #     # st.write(ASM_Code_df['CM_City'].unique())
        #     with col1:
        #         CITY = st.selectbox('Select City:', options = list(ASM_Code_df[(ASM_Code_df['CM_ASM Code']==ASM_CODE) & (ASM_Code_df['CM_State']==STATE)]['CM_City'].unique()))
        #     with col2:
        #         Team = st.selectbox('Select Team:', options = list(ASM_Code_df[(ASM_Code_df['CM_ASM Code']==ASM_CODE) & (ASM_Code_df['CM_State']==STATE)]['CM_Team Details'].unique()))
        #     # Create a Plotly line chart
        #     asm_filter_df = ASM_Code_df[(ASM_Code_df['CM_ASM Code']==ASM_CODE) & (ASM_Code_df['CM_State']==ASM_STATE) & (ASM_Code_df['CM_City']==CITY) & (ASM_Code_df['CM_Team Details']==Team)]
        #     ASM_Code_df.rename(columns = {
        #             'S_Total Sales': 'Sales',
        #             'C_Amount in Balance Transaction Currency': 'Collections'
        #     }, inplace = True)
        #     if len(asm_filter_df) == 0:
        #         st.write("Data Not Availabe")
        #     else:
                
        #         fig = px.bar(ASM_Code_df[(ASM_Code_df['CM_ASM Code']==ASM_CODE) & (ASM_Code_df['CM_State']==ASM_STATE) & (ASM_Code_df['CM_City']==CITY) & (ASM_Code_df['CM_Team Details']==Team)], x='Date', y=['Sales','Collections'],
        #                 labels={'value': 'Collection', 'Date': 'Timeline'},
        #                 title=f'Sales & Collection for {Customer_Name} from City: {CITY} & Team: {Team}')
                
        #         # Update the layout for better appearance
        #         fig.update_layout(xaxis_title='Date',
        #                         yaxis_title='Sales & Collection',
        #                         legend_title='Sales & Collection Trend')
                
        #         # Show the plot in Streamlit
        #         st.plotly_chart(fig)
            
            
        #     'Credit Limit Bin' 
        #     # Create a Plotly line chart
        #     Customer_level_data['Credit Limit Bin'] = Customer_level_data['Standardised Credit Limit'].apply(categorize_credit_limit)
        #     bin_order = ['<1000', '1000-10,000','10,000-2,00,000','2,00,000-5,00,000', '5,00,000-10,00,000','10,00,000-50,00,000', '50,00,000-1,00,00,000', '10,00,00,000+',]
        #     # ASM_Code_df = pd.DataFrame()
        #     ASM_Code_df['Credit Limit Bin'] = ASM_Code_df['Customer'].map(dict(zip(Customer_level_data['Customer'],Customer_level_data['Credit Limit Bin'])))
        #     ASM_Code_df_count = ASM_Code_df[ASM_Code_df['CM_ASM Code']==ASM_CODE]['Credit Limit Bin'].value_counts()    
        #     ASM_Code_df_count = ASM_Code_df_count.reindex(index=bin_order)
        #     ASM_Code_df_count = ASM_Code_df_count.reset_index()
        #     fig = px.bar(ASM_Code_df_count, x='Credit Limit Bin', y='count',
        #                 labels={'value': 'Credit Limit Bin', 'Count': 'Count'},
        #                 title='Credit Limit Distribution')
        #     # Update the layout for better appearance
        #     fig.update_layout(xaxis_title='Credit Limit Bin',
        #                     yaxis_title='Count',
        #                     legend_title='Collection Trend')
        #     # Show the plot in Streamlit
        #     st.plotly_chart(fig)
        #     st.dataframe(ASM_Code_df_count)
            
        # with tab8:
        #     # Filter DataFrame for the selected customer
        #     # profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Punjab.pkl')
        #     filtered_df = profit_df[profit_df['Customer'] == str(CUSTOMER)]
        #     filtered_df['Fiscal Year'] = filtered_df['Fiscal Year'].astype(str)
        #     filtered_df.sort_values(by='Fiscal Year', inplace=True)
        #     # filtered_df = filtered_df.dropna(subset=['Gross Margin%'])
        #     payment_terms = knock_df['Payment Term']
        #     if not payment_terms.empty:
        #         term_counts = payment_terms.value_counts()
        #         if not term_counts.empty:
        #             max_payment_term = term_counts.idxmax()
        #         else:
        #             max_payment_term = None    
        #     else:
        #         # Handle the case when no data is found for this customer
        #         max_payment_term = None
            
            
        #     avg_bal = key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)]['average_closing_bal'].values[0] if not key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)].empty else 0
            
            
        #     # invoices_365 = len(trans_df[trans_df['Customer'] == str(CUSTOMER)][(trans_df['Time Stamp'] >= threshold) & (trans_df['Time Stamp'] <= last_sale_activity)]['S_Billing Document'].unique())
        #     outstandings_total = key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)]['Number of Outstanding Invoices total'].values[0] if not key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)].empty else 0
        #     avg_transactions_size = round((trans_df[(trans_df['Customer'] == str(CUSTOMER)) & (trans_df['S_Billing Type'] == 'ZINV')].groupby('S_Billing Document')['S_Total Sales'].sum().mean()), 2)
        #     mean_past_due_days = knock_df[knock_df['Customer'] == str(CUSTOMER)]['Past Due Days'].mean() 
        #     mean_bill_settlement_days = knock_df[knock_df['Customer'] == str(CUSTOMER)]['Bill Settlement Days'].mean()
            
        #     collections_efficiency = features_phase2_v2.collection_efficiency(trans_df[trans_df['Customer'] == str(CUSTOMER)])
        #     debtor_days = key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)]['debtor_days'].values[0] if not key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)].empty else 0
        #     opening_bal = trans_df[trans_df['Customer'] == str(CUSTOMER)]['CM_Opening Balances'].iloc[0]
        #     knock_df['Default_Flag'] = knock_df.apply(key_metrics.defaults, axis=1)
        #     total_amt_defaulted =  knock_df[(knock_df['Customer'] == str(CUSTOMER)) & knock_df['Default_Flag'] == 1]['Sales Amount'].sum()
        #     st.header("📒 Key Information")
        #     # st.subheader("📜 Invoices & Payment Behavior")
        #     col1, col2, col3, col4 = st.columns(4)
        #     col1.metric("Total Number of Invoices", int(key_metrics_df[key_metrics_df['Customer'] == str(CUSTOMER)]['Number of invoices_total'].values[0]))
        #     # col2.metric("Invoices (365 Days)", invoices_365])
        #     col2.metric("Opening Balances", f"₹ {opening_bal}")
        #     col3.metric("Max Payment Term", max_payment_term)
        #     col4.metric("Avg Past Due Days", int(mean_past_due_days))
        #     st.markdown("---")
        #     col1, col2, col3, col4 = st.columns(4)
        #     col1.metric("Debtor Days", int(debtor_days))
        #     col2.metric("Total Number of Defaults", int(knock_df[knock_df['Customer'] == str(CUSTOMER)]['Default_Flag'].sum()))
        #     col3.metric("Avg Bill Settlement Days", int(mean_bill_settlement_days))
        #     col4.metric("Avg No. of Invoices per Year", )
        #     st.markdown("---")
        #     col1, col2, col3, col4 = st.columns(4)
        #     col1.metric("Outstanding Invoices", f"₹ {outstandings_total}")
        #     col2.metric("Avg Transaction Size", f"₹ {avg_transactions_size}")
        #     col3.metric("Collection Efficiency", f"₹ {collections_efficiency['Collection_Efficiency%'].iloc[0]}")
        #     col4.metric("Total Amount Defaulted", f"₹ {round(total_amt_defaulted, 2)}")
        #     st.markdown("---")
        with tab7:
            
            st.markdown(f"**Filing Consistency:**")
            st.markdown(f"**Average Filing Delay:**")
            st.markdown(f"**Risk Flag:**")
    else:
        # st.warning('🚨 No Data to show. \n Please create the feature file and generate the predictions to visualise the data.')
        st.warning('🚨 No Data Available\n\nTo proceed, please follow these steps:\n\n1. **Generate the Feature File**: Create the necessary feature file to begin the process.\n2. **Predict Credit Limits**: Use the generated feature file to predict the credit limits.\n3. **Visualize the Details**: Display the predicted credit limits and customer details for better analysis.')
# if page == 'Upload Data':
#     # Title
#     st.header("Upload Sales & Collection Files")

#     # Upload the first file
#     file1 = st.file_uploader("Upload the Sales file", type=["csv", "xlsx", "txt"])

#     # Upload the second file
#     file2 = st.file_uploader("Upload the Collection file", type=["csv", "xlsx", "txt"])
    
#     # Check if both files are uploaded
#     if file1 is not None and file2 is not None:
#         st.success("Both files have been uploaded successfully!")

#         # Display the names of the uploaded files
#         st.write("Sales uploaded:", file1.name)
#         st.write("Collection uploaded:", file2.name)
        
#         if file1.name.endswith(".csv"):
#             Sales_df = pd.read_csv(file1)
#         elif file1.name.endswith(".xlsx"):
#             Sales_df = pd.read_excel(file1)
        
#         if file2.name.endswith(".csv"):
#             Collection_df = pd.read_csv(file2)
#         elif file2.name.endswith(".xlsx"):
#             Collection_df = pd.read_excel(file2)


#         if st.button('Run Model'):
#             # Execute the transformations
#             st.write("Processing data...")
#             st.write("Sales Data Preview")
#             st.dataframe(Sales_df.head())

#             st.write("Collection Data Preview")
#             st.dataframe(Collection_df.head())
#             # Execute the transformations
#             st.write("Processing data...")

#             for step_name, df in main_transformation(Sales_df, Collection_df):
#                 if step_name == 'Customer Standardised Credit Limit':
#                     st.write(f"**{step_name} Done**")
#                     df = standardized_df[standardized_df['Customer'].isin([11001329,11006789,11007425,21000717,21002225,21002569,21003043,21005342,21006834,21009025,31003198,31004001,31005539])][['Customer', 'State', 'Created Date', 'First Activity', 'Last Activity',
#                                                                                                     'ASM Code', 'Tenure', 'No. of Days For average calculation',
#                                                                                                     'Avg Utilization %', 'Avg Daily Sales', 'Avg Yearly Sales',
#                                                                                                     'Average Yearly collections', 'Average sales during Kharif season',
#                                                                                                     'Average sales during Rabi season', 'last 12 months max collection',
#                                                                                                     'last 12 months max credit note', 'Most Frequent Payment Term used',
#                                                                                                     'Customer Profitability', 'Top 5 Materials purchased',
#                                                                                                     'Previous Credit Limit', 'Current Credit Limit', 'Max utilization %',
#                                                                                                     'Average Bill Settlement Days','last 12 months max sales',
#                                                                                                     'last 12 months min sales','Standardised Credit Limit']]
#                     df['Customer'] = df['Customer'].map(str)
#                     st.dataframe(df)  # Display each DataFrame with its head (you can customize the number of rows displayed)
#                     st.write("---")  # Add a separator between steps
#                 else:
#                     st.write(f"**{step_name} Done**")
#                     st.dataframe(df)
#                     st.write("---")  # Add a separator between steps
                
                

    
            
#     else:
#         st.warning("Please upload both files to proceed.")    