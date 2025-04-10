import os
import pandas as pd
import numpy as np
from tqdm import tqdm
from datetime import datetime
import warnings
import math
import warnings
warnings.filterwarnings("ignore")

# Display settings
pd.set_option('display.max_columns', None)

# Paths and data folder
data_folder = r"..\..\Data"


# states = [
#     'Uttar Pradesh', 'Telangana',
#     'Bihar', 'Maharashtra', 'Meghalaya', 'West Bengal', 'Orissa',
#     'Madhya Pradesh', 'Himachal Pradesh', 'Chhattisgarh', 'Karnataka',
#     'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand', 'Others'
# ]
# states = ['Andhra Pradesh']
states = [
    'Punjab', 'Rajasthan', 'Gujarat', 'Uttar Pradesh', 'Andhra Pradesh', 'Haryana', 'Telangana',
    'Bihar', 'Maharashtra', 'West Bengal', 'Orissa',
    'Madhya Pradesh', 'Himachal Pradesh', 'Chhattisgarh', 'Karnataka',
    'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand', 'Others'
]
# states = ['Maharashtra', 'West Bengal', 'Orissa',
#     'Madhya Pradesh', 'Himachal Pradesh', 'Chhattisgarh', 'Karnataka',
#     'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand', 'Others']
# states = ['Andhra Pradesh']


def cal_payment_behaviour_multiplier(mean_past_due_days, credit_cash):
    conditions = [
    credit_cash == 'cash',
    mean_past_due_days <= 30,  
    mean_past_due_days > 30
    
    ]
    values = [
        0.8,                    
        1.2,
        90 / mean_past_due_days
        
    ]
    Payment_Behaviour_Multiplier = np.select(conditions, values, default=np.nan).round(1)
    
    return Payment_Behaviour_Multiplier


def cal_debtor_days_multiplier(debtor_days):
    debtor_days_multiplier = 0
    # Ensure debtor_days is an integer
    if isinstance(debtor_days, pd.Timedelta):
        debtor_days = debtor_days.days  # Extract the number of days from Timedelta
    
    if debtor_days <= 120:
        return 1
    return debtor_days / 120

def cal_past_due_days_multiplier(mean_past_due_days):
    past_due_days_multiplier = 0
    if mean_past_due_days <= 30:
        past_due_days_multiplier = 1
    else:
        past_due_days_multiplier = round((mean_past_due_days/30), 2)
    return past_due_days_multiplier

def calculate_default(row):
    due_amount = row['Running Outstanding']
    zcsi_billing_type = row['S_Billing Type']
    
    # Handle missing or invalid values
    if pd.isna(due_amount) or pd.isna(zcsi_billing_type):
        return 0  # Ignore rows with missing data
    
    # Ignore default if Due Amount < 10,000
    if due_amount < 10000:
        return 0
    
    # Check ZCSI billing type condition
    if zcsi_billing_type == "ZCSI":
        return 0  # Ignore default if recent billing type is ZCSI
    
    # Otherwise, count as default
    return 1

def cal_default_multiplier(number_of_defaults):
    default_multiplier = 0
    # Default Multiplier=max(1−(defaults×0.2),0.2)
    if number_of_defaults == 0:
        default_multiplier = 1
    else:
        default_multiplier = max(1-(number_of_defaults*0.2),0.2)
    return default_multiplier

def number_of_outstanding_invoices(trans_df, knock_df):
    last_activity = trans_df['Time Stamp'].max()
    amt = trans_df.loc[trans_df['Time Stamp'] == last_activity, 'Running Outstanding'].iloc[0] if not trans_df[trans_df['Time Stamp'] == last_activity].empty else 0
    bill = trans_df.loc[trans_df['Time Stamp'] == last_activity, 'S_Billing Document'].iloc[0] if not trans_df[trans_df['Time Stamp'] == last_activity].empty else None
    
    if amt < 0:
        invoices = 0
    else:
        if bill is not None:
            knock_off_filtered = knock_df[knock_df['knockoff'] == 1]   
            if not knock_off_filtered.empty:
                last_knock_off_date = knock_off_filtered['Billing Date'].max()
            else:
                last_knock_off_date = None
            if last_knock_off_date is not None:
                knockoffs = knock_df[knock_df['Billing Date'] > last_knock_off_date]
                invoices = len(knockoffs['Sale_ID'].unique())
            else:
                invoices = 0
        else:
            invoices = 0
    return invoices

def name_cluster(cluster):
    if cluster == 0:
        cluster_name = 'Steady Customers'
    elif cluster == 1:
        cluster_name = 'Dormant/Low Value Customers'
    elif cluster == 2:
        cluster_name = 'Loyal Champions'
    elif cluster == 3:
        cluster_name = 'Occasional Spenders'
    else:
        cluster_name = None
    return cluster_name

def get_fiscal_year(timestamp):
        if pd.isna(timestamp):
            return 0
        # If the date is before April (Jan, Feb, Mar), return the previous year
        if timestamp.month < 4:
            return timestamp.year - 1
        # Otherwise, return the current year
        else:
            return timestamp.year
        
def defaults(row):
    # Ensure that Posting Date and Due Date are not missing
    if pd.isna(row['Posting Date']) or pd.isna(row['Due Date']):
        return 0  # Return 0 (no default) if any of the dates are missing
    
    row['Threshold'] = row['Due Date'] + pd.Timedelta(days=90)
    
    # Check if the condition is met and the Billing Type is not 'ZCSI'
    if (row['Posting Date'] > row['Threshold']) and (abs(row['Sales Amount'] - row['Collections Amount']) > 10000):
        if row['Billing Type'] == 'ZIDR':
            default = 0
        else:
            default = 1
    else:
        default = 0
    return default

w1 = 65
w2 = 35

def gst_multiplier(GST_Status):
    if GST_Status == 'Unregistered':
        return 0.05
    return 1


def profitability_last_year(Trans_df):
    from datetime import datetime

    def get_financial_year(date):
        """
        Get the financial year for a given date.

        Args:
            date (datetime.date): The date to check.

        Returns:
            str: The financial year in the format "YYYY-YYYY".
        """

        if date.month < 4:
            return f"{date.year - 1}-{date.year}"
        else:
            return f"{date.year}-{date.year + 1}"
    
    import fiscalyear
    def get_fiscal_year_foreign(timestamp):
        fiscalyear.START_MONTH = 4
        if pd.isna(timestamp):
                return 0
        fiscal_date = fiscalyear.FiscalDate(timestamp.year, timestamp.month, timestamp.day)
        return fiscal_date.fiscal_year
    
    def get_fiscal_year(timestamp):
        if pd.isna(timestamp):
            return 0
        # If the date is before April (Jan, Feb, Mar), return the previous year
        if timestamp.month < 4:
            return timestamp.year - 1
        # Otherwise, return the current year
        else:
            return timestamp.year

    Result_df_list = []
    
    Trans_df = Trans_df[Trans_df['CM_ZHIRE-ZNAME'] != "B2B"]
    # print(Trans_df[Trans_df['S_Item Description'].str.startswith("Rate", na = False)]['S_Billing Type'].value_counts()/len(Trans_df[Trans_df['S_Item Description'].str.startswith("Rate", na = False)]))
    #  Addin Financial Year
    Trans_df['Fiscal Year']= Trans_df['S_Billing Date'].apply(get_fiscal_year)
    Trans_df['Fiscal Year_I'] = Trans_df['S_Billing Date'].apply(get_financial_year)
    Trans_df['Material_P_Key'] = Trans_df['Fiscal Year'].astype(str) + " + " + Trans_df['S_Material'].astype(str)
    
    # Adding Costing Sheet
    # Loading and processing the Costing DataFrame
    costing_df = pd.read_excel(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\Costing Sheet.XLSX')
    # costing_df = pd.read_excel(r'C:\Users\darkhandelwal\Desktop\AgCare-Git\Data\Costing Sheet.XLSX')
    costing_df['Fiscal Year'] = costing_df['Fiscal Year'].astype(str)
    costing_df['Primary Key'] = costing_df['Fiscal Year'] + " + " + costing_df['Material']
    # Calculating the missing material prices
    costing_df['Net Value'] = costing_df['Net Value'].astype(float)
    mean_cost_sulphur = costing_df[costing_df['Material'].str.contains('SULPH')]['Net Value'].mean()
    mean_cost_bifen = costing_df[costing_df['Material'].str.contains('BIFEN')]['Net Value'].mean()
    # Mapping and filling the missing values
    price_map = costing_df[costing_df['Material'] == 'F-BAHAR_ORMAN0_C6'][['Fiscal Year', 'Net Value']].set_index('Fiscal Year').to_dict()['Net Value']
    costing_df.loc[costing_df['Material'] == 'F-BAHAR_ORMAN0_L5', 'Net Value'] = costing_df['Fiscal Year'].map(price_map)
    # Handling the case where 'F-BAHAR_ORMAN0_L5' is not present
    if not costing_df['Material'].str.contains('F-BAHAR_ORMAN0_L5').any():
        a_data = costing_df[costing_df['Material'] == 'F-BAHAR_ORMAN0_C6'].copy()
        b_data = a_data.copy()
        b_data['Material'] = 'F-BAHAR_ORMAN0_L5'
        costing_df = pd.concat([costing_df, b_data], ignore_index=True)
    # Correctly setting the 'Primary Key' with .loc to avoid SettingWithCopyWarning
    mask = costing_df['Material'] == 'F-BAHAR_ORMAN0_L5'
    costing_df.loc[mask, 'Primary Key'] = costing_df.loc[mask, 'Fiscal Year'].astype(str) + " + " + costing_df.loc[mask, 'Material']
    # fill in the prices of missing years
    years = list(range(2018, 2025))
    materials = costing_df['Material'].unique()
    full_index = pd.MultiIndex.from_product([materials, years], names=['Material', 'Fiscal Year'])
    full_df = pd.DataFrame(index=full_index).reset_index()
    costing_df['Fiscal Year'] = costing_df['Fiscal Year'].astype(str)
    full_df['Fiscal Year'] = full_df['Fiscal Year'].astype(str)
    costing_df = pd.merge(full_df, costing_df, on=['Material', 'Fiscal Year'], how='left')
    costing_df['Net Value'] = costing_df.groupby('Material')['Net Value'].ffill()
    # After filling, ensure that the 'Primary Key' is correctly updated for the newly filled rows
    costing_df['Primary Key'] = costing_df['Fiscal Year'] + " + " + costing_df['Material']
    
    Trans_df['Net Value per unit'] = 0
    Trans_df['Net Value per unit'] = Trans_df['S_Net Value.1']/Trans_df['S_Billing Qty in SKU']
    # raw_df = Trans_df[Trans_df['S_Material'].str.match(r'^\d+$', na=False)][['Customer','Fiscal Year','S_Material','S_Net Value.1','Net Value per unit']].groupby(['Fiscal Year','S_Material'])['Net Value per unit'].mean()
    Trans_df['CP_per_unit'] = Trans_df['Material_P_Key'].map(dict(zip(costing_df['Primary Key'],costing_df['Net Value'])))
    Trans_df['COGS'] = Trans_df['CP_per_unit'] * Trans_df['S_Billing Qty in SKU']
    # return Trans_df
    # Billing Type Pivot
    pivot_df = pd.pivot_table(data=Trans_df[~(Trans_df['S_Material'].str.match(r'^\d+$', na=False)) | (Trans_df['Material_P_Key']=='F-BAGS_GIFT0_H2')],
                                values='S_Net Value.1',
                                index=['Customer','Fiscal Year'],
                                columns=['S_Billing Type'],
                                aggfunc='sum',
                                fill_value=0)
    
    # COGS Pivot
    cogs_pivot_df = pd.pivot_table(data=Trans_df[(Trans_df['S_Billing Type']=='ZINV') & (~(Trans_df['S_Material'].str.match(r'^\d+$', na=False)) | (Trans_df['Material_P_Key']=='F-BAGS_GIFT0_H2') | (Trans_df['Fiscal Year']!='nan-nan'))],
                                values='COGS',
                                index=['Customer','Fiscal Year'],
                                # columns=['CP_per_unit'],
                                aggfunc='sum',
                                fill_value=0)
    cogs_pivot_df = cogs_pivot_df.reset_index()
    cogs_pivot_df = cogs_pivot_df[cogs_pivot_df['Fiscal Year']!='nan-nan']
    # cogs_pivot_df.rename(columns = {'CM_Fiscal Year': 'Fiscal Year'}, inplace = True)
    # cogs_pivot_df['Fiscal Year'] = cogs_pivot_df['Fiscal Year'].astype(str)
    
    pivot_df['Net Return'] = pivot_df.get('ZIRE', 0) + pivot_df.get('ZWCR',0) + pivot_df.get('ZCSI', 0)
    pivot_df['Net Sale Amount'] = pivot_df.get('ZINV', 0) - pivot_df['Net Return']
    
    # Rate Difference
    Rate_Diff_Sales = Trans_df[Trans_df['S_Billing Type'].isin(['ZICR', 'ZIDR', 'ZWDR', 'ZWCR'])][['Customer', 'Fiscal Year', 'S_Net Value.1']]
    # Rate_Diff_Sales = Trans_df[(Trans_df['S_Item Description'].str.startswith("Rate", na = False)) | (Trans_df['S_Billing Type'] == 'ZICR')][['Customer', 'Fiscal Year', 'S_Net Value.1']]
    Rate_Diff_Sales = Rate_Diff_Sales.groupby(['Customer', 'Fiscal Year'])['S_Net Value.1'].sum().reset_index()
    Rate_Diff_Sales.rename(columns={'S_Net Value.1':'Rate Diff'},inplace=True)
    pivot_df = pivot_df.reset_index()
    Result_df = pivot_df.merge(Rate_Diff_Sales,how='left',on=['Customer','Fiscal Year'])
    # Result_df['Fiscal Year'] = Result_df['Fiscal Year'].astype(str)
    Result_df = Result_df.merge(cogs_pivot_df,how='left',on=['Customer','Fiscal Year'])
    # Check for required columns and set default values if missing
    if 'Net Sale Amount' not in Result_df.columns:
        print("Warning: 'Net Sale Amount' column is missing. Defaulting to 0.")
        Result_df['Net Sale Amount'] = 0
    if 'Rate Diff' not in Result_df.columns:
        print("Warning: 'Rate Diff' column is missing. Defaulting to 0.")
        Result_df['Rate Diff'] = 0
    Result_df.fillna(0, inplace= True)
    # Calculate 'Net Revenue'
    Result_df['Net Revenue'] = Result_df['Net Sale Amount'] - Result_df['Rate Diff']
    # Check for 'COGS' column and set default if missing
    if 'COGS' not in Result_df.columns:
        print("Warning: 'COGS' column is missing. Defaulting to 0.")
        Result_df['COGS'] = 0

    # Calculate Gross Margin
    Result_df['Gross Margin'] = Result_df['Net Revenue'] - Result_df['COGS']

    # Calculate Gross Margin%
    Result_df['Gross Margin%'] = Result_df.apply(
        lambda row: row['Gross Margin'] / abs(row['Net Sale Amount']) if row['Net Sale Amount'] != 0 else 1,
        axis=1
    )

    Result_df_list.append(Result_df)
    Final_result_df = pd.concat(Result_df_list,axis=0)
    #using Financial year as 'Fiscal Year'
    return Final_result_df

def collection_efficiency(trans_df):
    # Calculate total collections and total sales by customer
    Total_Collections = trans_df.groupby('Customer', as_index=False)['C_Amount in Balance Transaction Currency'].sum()
    Total_Sales = trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer', as_index=False)['S_Total Sales'].sum()
    
    # Merge collections and sales into one DataFrame
    sale_collection = pd.merge(Total_Collections, Total_Sales, on='Customer', how='outer')

    # Fill missing sales or collections with 0 before calculating efficiency
    sale_collection['C_Amount in Balance Transaction Currency'] = sale_collection['C_Amount in Balance Transaction Currency'].fillna(0)
    sale_collection['S_Total Sales'] = sale_collection['S_Total Sales'].fillna(0)

    # Calculate Collection Efficiency with zero-division handling
    sale_collection['Collection_Efficiency%'] = (
        (abs(sale_collection['C_Amount in Balance Transaction Currency']) / sale_collection['S_Total Sales'].replace(0, float('nan'))) * 100
    ).fillna(0).round(2)  # Replace inf with NaN and fill NaN with 0

    return sale_collection

def key_metrics(trans_df1, knock_df1, profit_df1):
# states = ['Andhra Pradesh']
# customer_data = {}
# for state in tqdm(states, desc="Processing States"):
    # trans_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_{state}.pkl')
    # knock_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_{state}.pkl')
    # profit_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_{state}.pkl')
    # break
    # customers = ['11005411', '11003891', '11005794', '11001090', '11000576']
    # customers = ['11003796']
    # trans_df1 = trans_df1[trans_df1['Customer'].isin(customers)]
    all_customer_data = []
    start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')
    cutoff_date = pd.to_datetime('2024-03-31')
    # collections_efficiency = collection_efficiency(trans_df1)
    # trans_df1 = trans_df1[trans_df1['S_Canceled'] != 'X']
    # trans_df1 = trans_df1[~trans_df1['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
    # trans_df1 = trans_df1[trans_df1['CM_ZHIRE-ZNAME'] != "B2B"]
    knock_df1['Billing Date'] = pd.to_datetime(knock_df1['Billing Date'])
    knock_df1['Posting Date'] = pd.to_datetime(knock_df1['Posting Date'])
    knock_df1 = knock_df1[(knock_df1['Billing Date'] >= start_cutoff) & (knock_df1['Posting Date'] >= start_cutoff)]
    knock_df1 = knock_df1[(knock_df1['Billing Date'] <= cutoff_date) & (knock_df1['Posting Date'] <= cutoff_date)]
    # profit_df = profit_df[profit_df['Fiscal Year'] >= 2022]
    grouped_customers = trans_df1.groupby('Customer')
    for customer, group_df in tqdm(grouped_customers, delay = 2):
        last_12_months_min_sales = group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].min().tail(12).min()
        state = group_df.iloc[0]['CM_State']
        top_5_materials_purchased = ", ".join([mat for mat in group_df.groupby('S_Material')['S_Total Sales'].max().sort_values(ascending=False).index if mat.startswith('F-')][0:5])
        group_df['Time Stamp'] = pd.to_datetime(group_df['Time Stamp'], errors='coerce')
        # group_df['Time Stamp'] = group_df['Time Stamp'].dt.strftime('%d-%m-%Y')
        # group_df['Time Stamp'] = group_df['Time Stamp'].strftime('%d-%m-%Y')
        group_df = group_df[group_df['Time Stamp'] <= cutoff_date]
        group_df = group_df[group_df['Time Stamp'] >= start_cutoff]
        knock_df = knock_df1[knock_df1['Customer'] == customer]
        knock_df['cash/credit'] = np.where(knock_df['Payment Term'] == '0001', 'cash', 'credit')
        knock_df = knock_df[knock_df['cash/credit'] != 'cash']
        avg_bill_settlement_days = knock_df['Bill Settlement Days'].mean()
        profit_df = profit_df1[profit_df1['Customer'] == customer]
        tenure = round(((pd.to_datetime(group_df['Time Stamp']).max() - pd.to_datetime(group_df['Time Stamp']).min()).days / 365), 2) if not group_df.empty else 0
        # def format_tenure(tenure):
        # # Check the decimal part of the tenure
        #     if tenure - int(tenure) >= 0.6:
        #         return int(tenure) + 1
        #     return int(tenure)
        # tenure = format_tenure(tenure)
        # Filter sales and returns
        
        payment_terms = knock_df['Payment Term']
        if not payment_terms.empty:
            term_counts = payment_terms.value_counts()
            if not term_counts.empty:
                max_payment_term = term_counts.idxmax()
            else:
                max_payment_term = None    
        else:
            # Handle the case when no data is found for this customer
            max_payment_term = None
        invoices = group_df[(group_df['S_Billing Type'] == 'ZINV') & (group_df['S_Canceled'] != 'X') & (~group_df['S_Description'].isin(['Cancelled Cred Memo', 'Cancelled Invoice']))]
        invoices_total = len(invoices['S_Billing Document'].unique())
        avg_number_of_invoices = round((invoices_total/tenure), 2) if tenure != 0 else 0
        if avg_number_of_invoices > invoices_total:
            avg_number_of_invoices = invoices_total
        elif avg_number_of_invoices < 1:
            avg_number_of_invoices = 1
        else:
            avg_number_of_invoices = round(avg_number_of_invoices)
        no_of_outstanding_invoices_total = number_of_outstanding_invoices(group_df, knock_df[knock_df['Customer'] == customer])
        group_df['Fiscal Year']= group_df['S_Billing Date'].apply(get_fiscal_year)
        first_sale_activity = invoices['S_Billing Date'].min() if pd.notna(invoices['S_Billing Date'].min()) else pd.NaT
        last_sale_activity = invoices['S_Billing Date'].max() if pd.notna(invoices['S_Billing Date'].max()) else pd.NaT
        # Find first and last activity
        first_activity = group_df['Time Stamp'].min()
        last_activity = group_df['Time Stamp'].max()
        # Convert dates to string or handle NaT
        last_sale_activity = pd.to_datetime(last_sale_activity) if isinstance(last_sale_activity, str) else last_sale_activity
        # last_sale_activity = last_sale_activity.strftime('%d-%m-%Y') if pd.notna(last_sale_activity) else None
        # first_sale_activity = first_sale_activity.strftime('%d-%m-%Y') if pd.notna(first_sale_activity) else None
        first_activity = pd.to_datetime(first_activity).strftime('%d-%m-%Y') if pd.notna(first_activity) else None
        last_activity = pd.to_datetime(last_activity)
        # Set threshold based on tenure
        if tenure > 1:
            threshold = pd.to_datetime(pd.to_datetime(last_sale_activity) - pd.Timedelta(days=365))
        else:
            threshold = pd.to_datetime(first_sale_activity) 
        # threshold = pd.to_datetime(threshold)

        activity = pd.to_datetime(last_activity) - threshold
        # # Handle Running Outstanding
        prev_activity = None
        threshold_yearmonth = threshold.to_period('M') if pd.notna(threshold) else None
        group_df['yearmonth'] = group_df['Time Stamp'].dt.to_period('M')
        if pd.notna(threshold):
            if threshold_yearmonth not in group_df['yearmonth']:
                prev_rows = group_df[group_df['Time Stamp'] <= threshold]
                if not prev_rows.empty:
                    prev_activity = pd.to_datetime(prev_rows.iloc[-1]['Time Stamp'], format = "%d-%m-%Y")
                else:
                    prev_activity = threshold
        else:
            prev_activity = None
        # Convert dates to string or handle None
        
        # threshold = threshold.strftime('%d-%m-%Y') if pd.notna(threshold) else None
        # prev_activity = prev_activity.strftime('%d-%m-%Y') if prev_activity else None

        # invoices1 = invoices[(invoices['Time Stamp'] >= threshold) & (invoices['Time Stamp'] <= last_sale_activity)]
        # No_of_Invoices_Created_365 = len(invoices1['S_Billing Document'].unique())
        # Calculate running outstanding and sales

        # if pd.notna(prev_activity):
        #     running_df = group_df[(group_df['Time Stamp'] >= prev_activity) & (group_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
        #     Total_Sales = invoices[(invoices['Time Stamp'] > prev_activity) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
        #     # Net_Returns = returns[(returns['Time Stamp'] > prev_activity) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
        #     running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
        #     grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
        # else:
        #     grouped_running = pd.DataFrame()
        
        if prev_activity:
            running_df = group_df[(group_df['Time Stamp'] >= prev_activity) & (group_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
            Total_Sales = invoices[(invoices['Time Stamp'] > prev_activity) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            # Net_Returns = returns[(returns['Time Stamp'] > prev_activity) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
            
            grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
        else:
            running_df = group_df[(group_df['Time Stamp'] >= threshold) & (group_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
            Total_Sales = invoices[(invoices['Time Stamp'] > threshold) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            # Net_Returns = returns[(returns['Time Stamp'] > threshold) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
            grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
        
        # Generate YearMonth and calculate average balance
        if not grouped_running.empty:
            # Create a complete range of months and merge with the grouped data
            all_months = pd.period_range(start=grouped_running['YearMonth'].min(), end=grouped_running['YearMonth'].max(), freq='M')
            all_months_df = pd.DataFrame({'YearMonth': all_months})
            merged_running = pd.merge(all_months_df, grouped_running, on='YearMonth', how='left')
            merged_running['Running Outstanding'] = merged_running['Running Outstanding'].fillna(method='ffill')

            # Calculate the last 12 months (handle cases with fewer than 12 rows)
            last_12_months = merged_running.tail(12)  # Ensure we only take the last 12 months
            active_months = activity/30
            avg_bal = round((last_12_months['Running Outstanding'].sum() / 12), 2)  if activity.days >= 365 else round((merged_running['Running Outstanding'].sum() / merged_running.shape[0]), 2)# Use the correct denominator
        else:
            avg_bal = 0
        
        if pd.notna(threshold) and pd.notna(last_sale_activity):
                trans_df_last_year = group_df[(group_df['S_Billing Date'] >= threshold) & (group_df['S_Billing Date'] <= last_sale_activity)]
        else:
            trans_df_last_year = pd.DataFrame()
        total_profit = profit_df['Gross Margin'].sum()
        total_profit_percent = profit_df['Gross Margin%'].mean()
        net_revenue_total = profit_df['Net Revenue'].sum()
        avg_transactions_size = group_df[(group_df['S_Billing Type'] == 'ZINV') & (group_df['S_Canceled'] != 'X') & (~group_df['S_Description'].isin(['Cancelled Cred Memo', 'Cancelled Invoice']))].groupby('S_Billing Document')['S_Total Sales'].sum().mean()
        if trans_df_last_year.empty:
            net_rev_365 = 0 if profit_df.empty else profit_df['Net Revenue'].values[0]
        else:
            df1 = profitability_last_year(trans_df_last_year)
            df1.fillna(0, inplace = True)
            net_sale_amt_365 = df1['Net Sale Amount']
            net_rev_365 = df1['Net Revenue'].sum()
            
        if net_rev_365 == 0:
            debtor_days = 0
        else:
            try:
                avg_days = avg_bal / net_rev_365

                # Handle extreme cases or invalid values
                if avg_days > 1e6 or pd.isna(avg_days):
                    debtor_days = 0
                else:
                    if isinstance(avg_days, pd.Timedelta):  # If avg_days is a Timedelta object
                        avg_days = avg_days.total_seconds() / (24 * 60 * 60)  # Convert to days

                    # Ensure `activity` is numeric
                    multiplier = 365 if tenure > 1 else activity.days
                    if not isinstance(multiplier, (int, float)):
                        raise ValueError("The multiplier (activity) must be numeric.")

                    # Calculate debtor days
                    debtor_days = round(avg_days * multiplier, 2)
            # except OverflowError:
            #     debtor_days = 0  # Handle overflow gracefully
            except Exception as e:
                print(f"An error occurred: {e}")
                debtor_days = None
        customer_entry = ({
            'Customer': customer,
            'State': state,
            'Tenure': tenure,
            'first activity': first_activity,
            'last activity': last_activity,
            'avg_bill_settlement_days':  int(avg_bill_settlement_days) if avg_bill_settlement_days is not None and not math.isnan(avg_bill_settlement_days) else 0,
            'Number of invoices Total': invoices_total,
            'no_of_outstanding_invoices_total': no_of_outstanding_invoices_total,
            'avg_transaction_size': avg_transactions_size,
            'avg_number_of_invoices': avg_number_of_invoices,
            'Activity': activity,
            'Last 12 months min sales': last_12_months_min_sales,
            'max_payment_term': max_payment_term,
            'Average Bal in the last 365 days': avg_bal,
            'Net Revenue 365': net_rev_365,
            'Debtor Days': debtor_days,
            'Top 5 Materials purchased':top_5_materials_purchased
        })
        all_customer_data.append(customer_entry)
    # break

    scoring_df = pd.DataFrame(all_customer_data)
    return scoring_df
# trans_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\trans_df_combined_2547.pkl')
# knock_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\knock_df_combined_2547.pkl')
# profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\customer_details_demo_data\profit_df_combined_2547.pkl')


#  pd.DataFrame(all_customer_data).to_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\scoring_v2\scoring_v2_{state}.pkl')


# customer_data = {}
# # Process data for each state
# for state in tqdm(states, desc="Processing States"):
#     group_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_{state}.pkl')
#     knock_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_{state}.pkl')
#     profit_df1 = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_{state}.pkl')
#     all_customer_data = []
#     for customer in tqdm(trans_df1['Customer'].unique()):
        
#         trans_df = trans_df1[trans_df1['Customer'] == customer]
#         knock_df = knock_df1[knock_df1['Customer'] == customer]
#         profit_df = profit_df1[profit_df1['Customer'] == customer]
#         # calculating max payment term for each customer
#         # payment_terms = knock_df['Payment Term']
#         # if not payment_terms.empty:
#         #     term_counts = payment_terms.value_counts()
#         #     if not term_counts.empty:
#         #         max_payment_term = term_counts.idxmax()
#         #     else:
#         #         max_payment_term = None    
#         # else:
#         #     # Handle the case when no data is found for this customer
#         #     max_payment_term = None
#         # asm code
#         # asm_code = trans_df['CM_ASM Code']
#         # penalty
#         # penalty = 30 if asm_code.isin(['WOP', 'BLOCKED', 'LEGAL', 'DORMANT', 'B2B', 'VACANT_GUN', 'VAC_GHAZIP', 'VACNT_GOLA']).any() else 0
#         # cash or credit customer
#         # cash_credit = "cash" if (knock_df['Payment Term'] == '0001').all() else "credit"
#         # calculating mean past due days for each customer
#         # mean_past_due_days = knock_df['Past Due Days'].mean()
#         # net_sale_amt = profit_df['Net Sale Amount'].sum()
#         # cogs = profit_df['COGS'].sum()
#         # calculating Payment_Behaviour_Multiplier for each customer
#         # Payment_Behaviour_Multiplier = cal_payment_behaviour_multiplier(mean_past_due_days, cash_credit)
#         # identifying cluster to which the customer belongs
#         # cluster = rfm_results.loc[rfm_results['Customer'] == customer, 'Cluster'].iloc[0] if not rfm_results.loc[rfm_results['Customer'] == customer, 'Cluster'].empty else 0
#         # cluster_name = name_cluster(cluster)
#         # locally max profit in the cluster
#         # max_profit_segment_local = round(rfm_results[(rfm_results['State'] == state) & (rfm_results['Cluster'] == cluster)]['Monetary Value'].max(), 2) if pd.notna(cluster) else 0
#         # nationally max profit in the cluster
#         # max_profit_segment_national = round((rfm_results[(rfm_results['Cluster'] == cluster)]['Monetary Value'].max()), 2) if pd.notna(cluster) else 0
#         # total profit of each customer
#         # total_profit = round((rfm_results[rfm_results['Customer'] == customer]['Monetary Value']), 2).values[0] if not rfm_results[rfm_results['Customer'] == customer]['Monetary Value'].empty else 0
#         # profitability score local
#         # profitability_score_local = round(((total_profit / max_profit_segment_local) * 100 * Payment_Behaviour_Multiplier), 2) if max_profit_segment_local != 0 else 1
#         # profitability score national
#         # profitability_score_national = round(((total_profit / max_profit_segment_national) * 100 * Payment_Behaviour_Multiplier), 2) if max_profit_segment_national != 0 else 1
#         # debtor days calculation
#         tenure = (trans_df['Time Stamp'].max() - trans_df['Time Stamp'].min()).days / 365 if not trans_df.empty else 0
#         def format_tenure(tenure):
#         # Check the decimal part of the tenure
#             if tenure - int(tenure) >= 0.6:
#                 return int(tenure) + 1
#             return int(tenure)
#         tenure = format_tenure(tenure)
#         # Filter sales and returns
#         invoices = trans_df[(trans_df['Customer'] == customer) & (trans_df['S_Billing Type'] == 'ZINV')]
#         trans_df['Fiscal Year']= trans_df['S_Billing Date'].apply(get_fiscal_year)
#         # Rate_Diff_Sales = trans_df[trans_df['S_Billing Type'].isin(['ZICR', 'ZIDR', 'ZWDR', 'ZWCR'])][['Customer', 'Fiscal Year', 'S_Net Value.1']]
#         # # Rate_Diff_Sales = trans_df[(trans_df['S_Item Description'].str.startswith("Rate", na = False)) | (trans_df['S_Billing Type'] == 'ZICR')][['Customer', 'Fiscal Year', 'S_Net Value.1']]
#         # Rate_Diff_Sales = Rate_Diff_Sales.groupby(['Customer', 'Fiscal Year'])['S_Net Value.1'].sum().reset_index()
#         # Rate_Diff_total = Rate_Diff_Sales['S_Net Value.1'].sum()
#         # no_of_outstanding_invoices_total = number_of_outstanding_invoices(trans_df, knock_df)
#         # # avg_number_of_invoices = round(((No_of_Invoices_Created+no_of_outstanding_invoices)/2), 2)
#         # No_of_Invoices_Created_total = len(invoices['S_Billing Document'].unique())
        
#         # # updated average number of invoices
#         # avg_number_of_invoices = round((No_of_Invoices_Created_total/tenure), 2) if tenure != 0 else 0
#         # returns = trans_df[(trans_df['Customer'] == customer) & (trans_df['S_Billing Type'].isin(['ZIRE', 'ZWCR', 'ZCSI']))]
#         # # Find first and last sale activity
#         first_sale_activity = invoices['S_Billing Date'].min() if pd.notna(invoices['S_Billing Date'].min()) else pd.NaT
#         # if not pd.isna(first_sale_activity):
#         last_sale_activity = invoices['S_Billing Date'].max()
#         # Find first and last activity
#         first_activity = trans_df['Time Stamp'].min()
#         last_activity = trans_df['Time Stamp'].max()
#         # Convert dates to string or handle NaT
#         last_sale_activity = pd.to_datetime(last_sale_activity) if isinstance(last_sale_activity, str) else last_sale_activity
#         last_sale_activity = last_sale_activity.strftime('%d-%m-%Y') if pd.notna(last_sale_activity) else None
#         first_activity = first_activity.strftime('%d-%m-%Y') if pd.notna(first_activity) else None
#         last_activity = last_activity.strftime('%d-%m-%Y')
#         # Set threshold based on tenure
#         if tenure > 1:
#             threshold = pd.to_datetime(last_sale_activity) - pd.Timedelta(days=365)
#         else:
#             threshold = pd.to_datetime(first_sale_activity) 
#         # threshold = pd.to_datetime(threshold)
#         activity = pd.to_datetime(last_activity) - threshold
#         # # Handle Running Outstanding
#         # prev_activity = None
#         # threshold_yearmonth = threshold.to_period('M') if pd.notna(threshold) else None
#         # trans_df['yearmonth'] = trans_df['Time Stamp'].dt.to_period('M')
#         # if pd.notna(threshold):
#         #     if threshold_yearmonth not in trans_df['yearmonth']:
#         #         prev_rows = trans_df[trans_df['Time Stamp'] <= threshold]
#         #         if not prev_rows.empty:
#         #             prev_activity = pd.to_datetime(prev_rows.iloc[-1]['Time Stamp'])
#         #         else:
#         #             prev_activity = threshold
#         # else:
#         #     prev_activity = None
#         # # Convert dates to string or handle None
#         # threshold = threshold.strftime('%d-%m-%Y') if pd.notna(threshold) else None
#         # prev_activity = prev_activity.strftime('%d-%m-%Y') if prev_activity else None
#         # # invoices1 = invoices[(invoices['Time Stamp'] >= threshold) & (invoices['Time Stamp'] <= last_sale_activity)]
#         # # No_of_Invoices_Created_365 = len(invoices1['S_Billing Document'].unique())
#         # # Calculate running outstanding and sales
#         # if prev_activity:
#         #     running_df = trans_df[(trans_df['Time Stamp'] >= prev_activity) & (trans_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
#         #     Total_Sales = invoices[(invoices['Time Stamp'] > prev_activity) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
#         #     # Net_Returns = returns[(returns['Time Stamp'] > prev_activity) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
#         #     running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
#         #     grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
#         # else:
#         #     running_df = trans_df[(trans_df['Time Stamp'] >= threshold) & (trans_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
#         #     Total_Sales = invoices[(invoices['Time Stamp'] > threshold) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
#         #     # Net_Returns = returns[(returns['Time Stamp'] > threshold) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
#         #     running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
#         #     grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
        
#         # # Generate YearMonth and calculate average balance
#         # if not grouped_running.empty:
#         #     # Create a complete range of months and merge with the grouped data
#         #     all_months = pd.period_range(start=grouped_running['YearMonth'].min(), end=grouped_running['YearMonth'].max(), freq='M')
#         #     all_months_df = pd.DataFrame({'YearMonth': all_months})
#         #     merged_running = pd.merge(all_months_df, grouped_running, on='YearMonth', how='left')
#         #     merged_running['Running Outstanding'] = merged_running['Running Outstanding'].fillna(method='ffill')
            
#         #     # Calculate the last 12 months (handle cases with fewer than 12 rows)
#         #     last_12_months = merged_running.tail(12)  # Ensure we only take the last 12 months
#         #     active_months = activity/30
#         #     avg_bal = round((last_12_months['Running Outstanding'].sum() / 12), 2)  if tenure > 1 else round((merged_running['Running Outstanding'].sum() / merged_running.shape[0]), 2)# Use the correct denominator
#         # else:
#         #     avg_bal = 0
#         # --- Preliminaries: Convert and cache date values ---
#         threshold = pd.to_datetime(threshold)
#         last_activity_dt = pd.to_datetime(last_activity)
#         threshold_yearmonth = threshold.to_period('M') if pd.notna(threshold) else None

#         # Ensure trans_df has a 'yearmonth' column computed only once
#         if 'yearmonth' not in trans_df.columns:
#             trans_df['yearmonth'] = trans_df['Time Stamp'].dt.to_period('M')

#         # --- Determine previous activity date (if any) ---
#         prev_activity = None
#         if pd.notna(threshold):
#             # If threshold's period is not in trans_df['yearmonth'], get the last row with Time Stamp <= threshold.
#             if threshold_yearmonth not in trans_df['yearmonth'].values:
#                 prev_rows = trans_df.loc[trans_df['Time Stamp'] <= threshold]
#                 prev_activity = prev_rows.iloc[-1]['Time Stamp'] if not prev_rows.empty else threshold

#         # Optionally, create string versions if needed downstream
#         threshold_str = threshold.strftime('%d-%m-%Y') if pd.notna(threshold) else None
#         prev_activity_str = prev_activity.strftime('%d-%m-%Y') if prev_activity is not None else None

#         # --- Determine the range for "Running Outstanding" ---
#         activity_range_start = prev_activity if prev_activity is not None else threshold
#         activity_range_end = last_activity_dt

#         # Filter transactions (running_df) and invoices only once using .loc[]
#         running_df = trans_df.loc[(trans_df['Time Stamp'] >= activity_range_start) &
#                                 (trans_df['Time Stamp'] <= activity_range_end)].copy()

#         Total_Sales = invoices.loc[(invoices['Time Stamp'] > activity_range_start) &
#                                 (invoices['Time Stamp'] <= activity_range_end), 'S_Net Value.1'].sum()

#         # --- Compute average balance if there is data ---
#         if not running_df.empty:
#             # Use existing 'yearmonth' column in running_df
#             if 'yearmonth' not in running_df.columns:
#                 running_df['yearmonth'] = running_df['Time Stamp'].dt.to_period('M')
            
#             # Group by yearmonth and take the last value of 'Running Outstanding'
#             grouped_running = running_df.groupby('yearmonth', as_index=False)['Running Outstanding'].last()
            
#             # Create a complete range of months between the minimum and maximum periods
#             all_months = pd.period_range(start=grouped_running['yearmonth'].min(), 
#                                         end=grouped_running['yearmonth'].max(), freq='M')
#             all_months_df = pd.DataFrame({'yearmonth': all_months})
            
#             # Merge the complete month range with the grouped data and forward-fill missing values
#             merged_running = pd.merge(all_months_df, grouped_running, on='yearmonth', how='left')
#             merged_running['Running Outstanding'].fillna(method='ffill', inplace=True)
            
#             # Extract only the last 12 months (or fewer if not available)
#             last_12_months = merged_running.tail(12)
            
#             # Compute avg_bal using different denominators based on tenure
#             if tenure > 1:
#                 avg_bal = round(last_12_months['Running Outstanding'].sum() / 12, 2)
#             else:
#                 avg_bal = round(merged_running['Running Outstanding'].sum() / merged_running.shape[0], 2)
#         else:
#             avg_bal = 0

#         # Now avg_bal holds the average running outstanding for the appropriate period.

#         if pd.notna(threshold) and pd.notna(last_sale_activity):
#             trans_df_last_year = trans_df[(trans_df['S_Billing Date'] >= threshold) & (trans_df['S_Billing Date'] <= last_sale_activity)]
#         else:
#             trans_df_last_year = pd.DataFrame()
#         total_profit = profit_df['Gross Margin'].sum()
#         total_profit_percent = profit_df['Gross Margin%'].mean()
#         net_revenue_total = profit_df['Net Revenue'].sum()

#         if trans_df_last_year.empty:
#             net_rev_365 = 0 if profit_df.empty else profit_df['Net Revenue'].values[0]
#             # Rate_Diff_365 = 0 if profit_df.empty else profit_df['Rate Diff'].values[0]
#             # cogs_365 = 0 if profit_df.empty else profit_df['COGS'].values[0]
#             # net_sale_amt_365 = 0 if profit_df.empty else profit_df['Net Sale Amount'].values[0]
#             # gross_margin_365 = 0 if profit_df.empty else profit_df['Gross Margin'].values[0]
#             # gross_margin_365_percent = 0 if profit_df.empty else profit_df['Gross Margin%'].values[0]
#             # total_profit_365 = 0 if profit_df.empty else profit_df['Gross Margin'].sum()
#             # total_profit_percent_365 = 0 if profit_df.empty else profit_df['Gross Margin%'].mean()
#         else:
#             df1 = profitability_last_year(trans_df_last_year)
#             df1.fillna(0, inplace = True)
#             net_sale_amt_365 = df1['Net Sale Amount']
#             # net_rev_365 = Total_Sales - Net_Returns
#             net_rev_365 = df1['Net Revenue'].sum()
#             # Rate_Diff_365 = df1['Rate Diff'].sum()
#             # cogs_365 = df1['COGS'].sum()
#             # net_sale_amt_365 = df1['Net Sale Amount'].sum()
#             # gross_margin_365 = df1['Gross Margin'].sum()
#             # gross_margin_365_percent = df1['Gross Margin%'].sum()
#             # total_profit_365 = df1['Gross Margin'].sum()
#             # total_profit_percent_365 = df1['Gross Margin%'].mean()

#         if net_rev_365 == 0:
#             debtor_days = None
#         else:
#             try:
#                 avg_days = avg_bal / net_rev_365

#                 # Handle extreme cases or invalid values
#                 if avg_days > 1e6 or pd.isna(avg_days):
#                     debtor_days = 0
#                 else:
#                     if isinstance(avg_days, pd.Timedelta):  # If avg_days is a Timedelta object
#                         avg_days = avg_days.total_seconds() / (24 * 60 * 60)  # Convert to days

#                     # Ensure `activity` is numeric
#                     multiplier = 365 if tenure > 1 else activity.days
#                     if not isinstance(multiplier, (int, float)):
#                         raise ValueError("The multiplier (activity) must be numeric.")

#                     # Calculate debtor days
#                     debtor_days = round(avg_days * multiplier, 2)
#             except OverflowError:
#                 debtor_days = 0  # Handle overflow gracefully
#             except Exception as e:
#                 print(f"An error occurred: {e}")
#                 debtor_days = 0

#         # calculating debtor days multiplier
#         # debtor_days_multiplier = cal_debtor_days_multiplier(debtor_days) if pd.notna(debtor_days) else 1
#         # print(f"Customer: {customer}, Debtor Days Multiplier: {debtor_days_multiplier}")
#         # calculating past_due_days_multiplier
#         # past_due_days_multiplier = cal_past_due_days_multiplier(mean_past_due_days)
#         # calculating number of defaults
#         # knock_df['Billing Date'] = pd.to_datetime(knock_df['Billing Date'])
#         # knock_df_last_year = knock_df[(knock_df['Billing Date'] >= threshold) & (knock_df['Billing Date'] <= last_activity)]
#         # # number_of_defaults_last_year = merged_df_last_year['Is_Default'].sum()
#         # # number_of_defaults = merged_df['Is_Default'].sum()
#         # knock_df['Default'] = knock_df.apply(defaults, axis=1)
#         # knock_df_last_year['Default'] = knock_df_last_year.apply(defaults, axis = 1)
#         # # defaults total
#         # grouped_defaults = knock_df.groupby('Sale_ID')['Default'].sum().reset_index()
#         # total_defaults = 0
#         # for idx in grouped_defaults.index:
#         #     if grouped_defaults.loc[idx, 'Default'] >= 1:
#         #         total_defaults += 1
#         # number_of_defaults = total_defaults
#         # # defaults for last year
#         # grouped_defaults_last_year = knock_df_last_year.groupby('Sale_ID')['Default'].sum().reset_index()
#         # total_defaults_last_year = 0
#         # for idx in grouped_defaults_last_year.index:
#         #     if grouped_defaults_last_year.loc[idx, 'Default'] >= 1:
#         #         total_defaults_last_year += grouped_defaults_last_year.loc[idx, 'Default']
#         # number_of_defaults_last_year = total_defaults_last_year
#         # # Calculate default multiplier
#         # # default_multiplier = cal_default_multiplier(number_of_defaults)
        
#         # # calcualte risk score
#         # # risk_score = round((((debtor_days_multiplier*default_multiplier*past_due_days_multiplier)/3)*100), 2)
#         # # calculate final score
#         # # final_score_local = round((((w1*profitability_score_local) + (w2 * (100 - risk_score)))/100), 2)
#         # # final_score_national = round((((w1*profitability_score_national) + (w2 * (100 - risk_score)))/100), 2)
#         # avg_bill_settlement_days = knock_df['Bill Settlement Days'].mean()
        
#         customer_data.update({
#             'Customer': customer,
#             # 'State': trans_df['CM_State'].values[0],
#             # # 'GST_Status': gst_status.values[0],
#             # 'start date': pd.to_datetime(trans_df['CM_Start Date on'].values[0]).strftime('%Y-%m-%d'),
#             # 'First Activity': first_activity,
#             # 'Last Activity': last_activity,
#             # 'First Sale Activity': first_sale_activity.date().strftime('%Y-%m-%d'),
#             # 'Last Sale Activity': last_sale_activity,
#             # 'activity': activity.days,
#             # 'threshold': threshold,
#             # 'Tenure': tenure,
#             # 'ASM Code': asm_code.values[0],
#             # 'cash or credit': cash_credit,
#             # 'Number of invoices_total': No_of_Invoices_Created_total,
#             # 'Number of invoices_365': No_of_Invoices_Created_365,
#             # 'Number of Outstanding Invoices total': no_of_outstanding_invoices_total,
#             # 'Average Number of Invoices per year': avg_number_of_invoices,
#             # 'max payment term used': max_payment_term,
#             # 'average past due days': int(mean_past_due_days),
#             # 'payment behaviour multiplier': Payment_Behaviour_Multiplier,
#             # 'Cluster Group': cluster_name,
#             # 'Rate Diff (ZICR, ZIDR, ZWDR, ZWCR) (total)': Rate_Diff_total,
#             # 'Rate Diff (ZICR, ZIDR, ZWDR, ZWCR) (365)': Rate_Diff_365,
#             # 'Net Sale Amount (total)': net_sale_amt,
#             # 'Net Sale Amount (365)': net_sale_amt_365,
#             # 'COGS_total': cogs,
#             # 'COGS_365':cogs_365,
#             # 'Net Revenue Total': float(net_revenue_total),
#             # 'GM_365/TotalProfit_365': gross_margin_365,
#             # 'GM%_365/TotalProfit%_365': gross_margin_365_percent,
#             # 'GM/total profit(NetRev-COGS)': float(total_profit),
#             # 'GM%/total_profit_percent(GM/NetRev)': float(total_profit_percent),
#             # 'Average Bill Settlement Days': float(avg_bill_settlement_days),
#             # 'max profit segment_local': float(max_profit_segment_local),
#             # 'max profit segment_national': float(max_profit_segment_national),
#             # 'profitability_score_local': float(profitability_score_local),
#             # 'profitability_score_national': float(profitability_score_national),
#             'average_closing_bal': float(avg_bal),
#             'net_rev_365': float(net_rev_365),
#             'debtor_days': debtor_days,
#             # 'number_of_defaults_365': number_of_defaults_last_year,
#             # 'number_of_defaults_total': number_of_defaults,
#             # 'debtor_days_multiplier': float(debtor_days_multiplier),
#             # 'Past Due Days Multiplier': float(past_due_days_multiplier), 
#             # 'default_multiplier': float(default_multiplier),
#             # 'risk_score': risk_score.values[0],
#             # 'final_score_local': final_score_local.values[0],
#             # 'final_score_national': final_score_national.values[0],
#             # 'Legal/WOP Consideration': penalty
#         })  
        
#         all_customer_data.append(customer_data)
    

#     break


# customer_data_df = pd.DataFrame(all_customer_data)
