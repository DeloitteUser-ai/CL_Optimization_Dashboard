import pandas as pd 
import os
from tqdm import tqdm
from datetime import datetime, timedelta
import numpy as np
# import scoring
import uuid
data_folder = r"Data"

states = ['Andhra Pradesh', 'Gujarat', 'Telangana', 'West Bengal', 'Punjab', 'Haryana',
            'Bihar', 'Maharashtra', 'Rajasthan', 'Orissa', 'Madhya Pradesh', 'Uttar Pradesh', 'Himachal Pradesh',
            'Chhattisgarh', 'Karnataka', 'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand', 'others']

customer_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\AgCare\app_phase_2_new\Data\customer_master_clean_v2')

customer_df.rename(columns = {
    'CM_Opening_Balance' : 'CM_Opening Balances',
    'CD_Credit Limit': 'CM_Credit Limit',
    'CM_ZName': 'CM_ZHIRE-ZNAME',
    'CM_State': 'CM_State_Code',
    'CM_Region_Name': 'CM_State'    
}, inplace = True)

# customer_gst_df = pd.read_pickle(r'C:\Users\darkhandelwal\Desktop\AgCare-Git\Data\PKL_Data_FINAL_v4\customer+gst_df.pkl')
# customer_gst_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\customer+gst_df.pkl')
# rfm_results = pd.read_excel(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\Phase2_RFM_Results.xlsx', sheet_name="RFM_results")
# customer_gst_df['Compliance Risk Trend'] = ''
# customer_gst_df['Filing Date'] = pd.to_datetime(customer_gst_df['Filing Date'], errors='coerce')
# customer_gst_df['Reporting Period'] = pd.to_datetime(customer_gst_df['Reporting Period'], errors='coerce')


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
def defaults(row):
    # Ensure that Posting Date and Due Date are not missing
    row['Posting Date'] = pd.to_datetime(row['Posting Date'])
    row['Due Date'] = pd.to_datetime(row['Due Date'])
    
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



def generate_features(trans_df, knock_df, profit_df, customer_df, scoring_df):
    cutoff_date = pd.to_datetime('2024-03-31')
    # start_date = cutoff_date - timedelta(days=6*30)
    start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')
    features_list = []
    customer_dict = customer_df.set_index('Customer')[['CM_Name 1', 'CM_ASM Code', 'CM_State']].to_dict('index')
# for state in tqdm(states):
#     profit_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_{state}.pkl')
#     trans_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_{state}.pkl')
    trans_df = trans_df.loc[:, ~trans_df.columns.duplicated()]
#     # trans_df = trans_df[trans_df['Customer'] == '11001090']
#     knock_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_{state}.pkl')
#     # trans_df.fillna(0, inplace = True)

    trans_df.loc[:, ~trans_df.columns.isin(['S_Billing Date', 'C_Posting Date'])] = trans_df.loc[:, ~trans_df.columns.isin(['S_Billing Date', 'C_Posting Date'])].fillna(0)
    knock_df.loc[:, ~knock_df.columns.isin(['Billing Date', 'Posting Date'])] = knock_df.loc[:, ~knock_df.columns.isin(['Billing Date', 'Posting Date'])].fillna(0)

    # knock_df.fillna(0, inplace = True)
    profit_df.fillna(0, inplace = True)
    knock_df['Billing Date'] = pd.to_datetime(knock_df['Billing Date'])
    knock_df['Posting Date'] = pd.to_datetime(knock_df['Posting Date'])
    knock_df['cash/credit'] = np.where(knock_df['Payment Term'] == '0001', 'cash', 'credit')
    knock_df = knock_df[knock_df['cash/credit'] != 'cash']

    # scoring_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\scoring_v2\scoring_v2_{state}.pkl')
    # scoring_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\scoring_v2\scoring_v2_5_customers.pkl')
    # cl_std_details = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\cl_std_v2\cl_std_{state}.pkl')

    trans_df = trans_df[trans_df['Time Stamp'] <= cutoff_date]
    trans_df = trans_df[trans_df['Time Stamp'] >= start_cutoff]

    trans_df = trans_df[trans_df['Customer'].isin(profit_df['Customer'])]

    knock_df = knock_df[(knock_df['Billing Date'] >= start_cutoff) & (knock_df['Posting Date'] >= start_cutoff)]
    knock_df = knock_df[(knock_df['Billing Date'] <= cutoff_date) & (knock_df['Posting Date'] <= cutoff_date)]
    profit_df = profit_df[profit_df['Fiscal Year'] >= 2022]
    # profit_df = profit_df[profit_df['Fiscal Year'] <= 2024]

    trans_df = trans_df[trans_df['S_Canceled'] != 'X']
    trans_df = trans_df[~trans_df['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
    trans_df = trans_df[trans_df['CM_ZHIRE-ZNAME'] != "B2B"]
    knock_df['Default'] = knock_df.apply(defaults, axis=1) if not knock_df.empty else None
    trans_df['Last Activity'] = trans_df['Customer'].map(trans_df.groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['First Activity'] = trans_df['Customer'].map(trans_df.groupby('Customer')['Time Stamp'].min().to_dict())
    
    trans_df['Sale First Activity'] = trans_df['Customer'].map(trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['Time Stamp'].min().to_dict())
    trans_df['Sale Last Activity'] = trans_df['Customer'].map(trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['Sale First Activity']= pd.to_datetime(trans_df['Sale First Activity'])
    trans_df['Sale Last Activity']= pd.to_datetime(trans_df['Sale Last Activity'])
    
    trans_df['Collect First Activity'] = trans_df['Customer'].map(trans_df[trans_df['C_Document Type'] == 'DZ'].groupby('Customer')['Time Stamp'].min().to_dict())
    trans_df['Collect Last Activity'] = trans_df['Customer'].map(trans_df[trans_df['C_Document Type'] == 'DZ'].groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['Collect First Activity']= pd.to_datetime(trans_df['Collect First Activity'])
    trans_df['Collect Last Activity']= pd.to_datetime(trans_df['Collect Last Activity'])
    # trans_df['Tenure_days'] = (trans_df['Last Activity'] - start_cutoff).dt.days
    trans_df['Sale Tenure_days'] = (trans_df['Sale Last Activity'] - trans_df['Sale First Activity']).dt.days
    trans_df['Collect Last Activity'] = pd.to_datetime(trans_df['Collect Last Activity'], errors='coerce')
    trans_df['Collect First Activity'] = pd.to_datetime(trans_df['Collect First Activity'], errors='coerce')
    trans_df['Collect Tenure_days'] = (trans_df['Collect Last Activity'] - trans_df['Collect First Activity']).dt.days
    
    # trans_df['Running Outstanding'] = pd.to_numeric(trans_df['Running Outstanding'])
    # trans_df['Credit Limit'] = pd.to_numeric(trans_df['Credit Limit'])
    # trans_df['Utilization %'] = trans_df.apply(lambda row: utilization_calculate(row['Running Outstanding'], row['Credit Limit']), axis=1)
    # avg_utilization = trans_df.groupby('Customer')['Utilization %'].mean().reset_index()
    # max_utilization = trans_df.groupby('Customer')['Utilization %'].max().reset_index()
    trans_df['S_Billing Date'] = pd.to_datetime(trans_df['S_Billing Date'], errors='coerce')
    collections_efficiency = collection_efficiency(trans_df)
    knock_df['Billing Date'] = pd.to_datetime(knock_df['Billing Date'], errors='coerce')
    # customers = ['11005411', '11003891', '11005794', '11001090', '11000576']
    # customers = ['11005411', '11001090']
    # customers = ['11005411']
    # trans_df = trans_df[trans_df['Customer'].isin(customers)]
    # Group by Customer_ID and calculate the maximum billing date for each customer
    max_date_df = trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['S_Billing Date'].max().reset_index()
    # Filter customers whose max billing date is after 31st March 2023
    filtered_customers = max_date_df[max_date_df['S_Billing Date'] > pd.to_datetime('31-03-2023', dayfirst=True)]
    trans_df = trans_df[trans_df['Customer'].isin(filtered_customers['Customer'].unique())]
    grouped_customers = trans_df.groupby('Customer')

    for customer, group_df in tqdm(grouped_customers, delay = 2):
    # for customer in grouped_customers:
        # group_df = grouped_customers[grouped_customers['Customer'] == customer]
        group_df['yearmonth'] = group_df['Time Stamp'].dt.to_period('M')
        last_sale_activity = group_df[group_df['S_Billing Type'] == 'ZINV']['S_Billing Date'].max()

        first_sale_activity = group_df[group_df['S_Billing Type'] == 'ZINV']['S_Billing Date'].min()
        # profit_df = profit_df[profit_df['Customer']== customer]
        last_one_year = last_sale_activity - pd.Timedelta(days=365)
        trans_df_last_year = group_df[(group_df['Time Stamp']>=last_one_year)]
        collections_efficiency_last_year = collection_efficiency(trans_df_last_year)
        last_sale_activity = pd.to_datetime(last_sale_activity, errors='coerce')
        if pd.notna(last_sale_activity):
            knock_df_last_year = knock_df[(knock_df['Customer'] == customer) & (knock_df['Billing Date'] >= last_one_year)]
        else:
            knock_df_last_year = pd.DataFrame()  # or handle appropriately
        
        # Get customer details
        customer_details = customer_dict.get(customer, {})
        # gst_details = gst_dict.get(customer, {})
        # rfm_details = rfm_dict.get(int(customer), {})
        
        last_activity = group_df['Time Stamp'].max()
        first_activity = group_df['Time Stamp'].min()
        # tenure = (trans_df['Time Stamp'].max() - trans_df['Time Stamp'].min()).days / 365 if not trans_df.empty else 0
        tenure = round(((group_df['Time Stamp'].max() - group_df['Time Stamp'].min()).days / 365), 2) if not group_df.empty else 0
        # def format_tenure(tenure):
        # # Check the decimal part of the tenure
        #     if tenure - int(tenure) >= 0.5:
        #         return int(tenure) + 1
        #     return int(tenure)
        # tenure = format_tenure(tenure)
        year_wise_sales = group_df.groupby('Year')['S_Total Sales'].sum().reset_index()
        # knock_df['Default'] = knock_df.apply(defaults, axis=1)
        # knock_df['Threshold'] = knock_df['Due Date'] + pd.Timedelta(days=90)
        # knock_df['Default'] = ((knock_df['Posting Date'] > knock_df['Threshold']) &
            # (knock_df['Sales Amount'] - knock_df['Collections Amount']).abs() > 10000 ) & (knock_df['Billing Type'] != 'ZIDR')
        grouped_defaults = knock_df[knock_df['Customer'] == customer].groupby('Sale_ID')['Default'].sum().reset_index()
        total_defaults = 0
        for idx in grouped_defaults.index:
            if grouped_defaults.loc[idx, 'Default'] >= 1:
                total_defaults += 1
        number_of_defaults = total_defaults
        top_materials = [
                mat for mat in group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('S_Material')['S_Total Sales'].max().sort_values(ascending=False).index
                if mat.startswith('F-')
        ]
        # no. of invoices
        invoices = group_df[group_df['S_Billing Type'] == 'ZINV']
        invoices_total = len(invoices['S_Billing Document'].unique())
        avg_number_of_invoices = round((invoices_total/tenure), 2) if tenure != 0 else 0
        if avg_number_of_invoices > invoices_total:
            avg_number_of_invoices = invoices_total
        elif avg_number_of_invoices < 1:
            avg_number_of_invoices = 1
        else:
            avg_number_of_invoices = round(avg_number_of_invoices)
        no_of_outstanding_invoices_total = number_of_outstanding_invoices(group_df, knock_df[knock_df['Customer'] == customer])

        # for total
        monthly_sales = group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Total Sales'].sum().reset_index()

        monthly_transactions = group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Billing Document'].count().reset_index()
        monthly_sales['S_Total Sales'] = pd.to_numeric(monthly_sales['S_Total Sales'], errors='coerce')
        monthly_transactions['S_Billing Document'] = pd.to_numeric(monthly_transactions['S_Billing Document'], errors='coerce')
        monthly_sales.fillna(0, inplace=True)
        monthly_transactions.replace(0, 1, inplace=True)  # Avoid division by zero
        average_each_month = monthly_sales.copy()
        average_each_month['Average Sales per Transaction'] = monthly_sales['S_Total Sales'] / monthly_transactions['S_Billing Document']
        # for the last year
        monthly_sales_last_year = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Total Sales'].sum().reset_index()
        monthly_transactions_last_year = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Billing Document'].count().reset_index()
        monthly_sales_last_year['S_Total Sales'] = pd.to_numeric(monthly_sales_last_year['S_Total Sales'], errors='coerce')
        monthly_transactions_last_year['S_Billing Document'] = pd.to_numeric(monthly_transactions_last_year['S_Billing Document'], errors='coerce')
        monthly_sales_last_year.fillna(0, inplace=True)
        monthly_transactions_last_year.replace(0, 1, inplace=True)  # Avoid division by zero
        average_each_month_last_year = monthly_sales_last_year.copy()
        average_each_month_last_year['Average Sales per Transaction'] = monthly_sales_last_year['S_Total Sales'] / monthly_transactions_last_year['S_Billing Document']
        
        not_defaults = abs(invoices_total - total_defaults)

        avg_defaults = total_defaults/tenure if tenure != 0 else total_defaults
        knock_df['yearmonth'] = knock_df['Billing Date'].fillna(knock_df['Posting Date']).dt.to_period('M')
        avg_monthly_default = knock_df[knock_df['Customer'] == customer].groupby('yearmonth')['Default'].sum().mean()
        if avg_defaults == 0:
            avg_monthly_default = 0
        elif avg_defaults < 1:
            avg_monthly_default = 1
        else:
            avg_monthly_default = round(avg_monthly_default)

        max_default_value = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].max()
        min_default_value = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].min()
        total_profit = profit_df['Gross Margin'].sum()

        avg_missed_payment_val = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].mean()


        # total_sales_last_12_months = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum()
        # if trans_df_last_year.empty:
        #     total_profit_last_12_months = 0 if profit_df.empty else profit_df['Gross Margin'].sum()
        # else:
        #     df1 = scoring.profitability_last_year(trans_df_last_year)
        #     total_profit_last_12_months = df1['Gross Margin'].sum()
        # number_of_transactions = len(group_df['S_Billing Document'].unique())
        # Append features for the customer
        # avg_transactions = group_df.groupby('S_Billing Document')['S_Total Sales'].sum()
        total_sales = group_df['S_Total Sales'].abs().sum()
        threshold = last_sale_activity - pd.Timedelta(days = 365)
        threshold = pd.to_datetime(threshold)
        std_credit_limit = group_df[(group_df['Time Stamp'] >= threshold) & (group_df['Time Stamp'] <= last_sale_activity)]['Running Outstanding'].max()

        features_list.append({
            'Customer': customer,
            # 'last_sales_activity': last_sale_activity,
            'ASM Code': customer_details.get('CM_ASM Code', ''),
            'Recency': int(str(cutoff_date - last_sale_activity).split(" days")[0].strip()) if not pd.isna(last_sale_activity) else 0,
            'Total Profit': profit_df[profit_df['Customer'] == customer]['Gross Margin'].sum(),
            'Profit Ratio': round((profit_df[profit_df['Customer'] == customer]['Gross Margin'].sum()/total_sales), 2) if total_sales != 0 else 0,
            'Average Past Due Days Total': round((knock_df[knock_df['Customer'] == customer]['Past Due Days'].mean()), 0),
            'Median Past Due Days Total': round((knock_df[knock_df['Customer'] == customer]['Past Due Days'].median()), 0),
            'debtor_days': int(scoring_df[scoring_df['Customer']== customer]['Debtor Days'].values[0]) if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            'net_rev_365': scoring_df[scoring_df['Customer']== customer]['Net Revenue 365'].values[0] if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            'Last 12 months max sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].max().tail(12).max(),
            'Last 12 months avg sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().tail(12).mean(),
            'Last 12 months median sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().tail(12).median(),
            'max sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].max(),
            'avg sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].mean(),
            'Avg Yearly Sales': (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / (group_df['Sale Tenure_days'].iloc[0] / 365))
            if group_df['Sale Tenure_days'].iloc[0] >= 365 else group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            'Avg Yearly Collections': abs((group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum() / (group_df['Collect Tenure_days'].iloc[0] / 365)))
            if group_df['Collect Tenure_days'].iloc[0] >= 365 else group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum(),
            'Avg Monthly Sales_Total': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / (group_df['Sale Tenure_days']/30).values[0] if group_df['Sale Tenure_days'].iloc[0] > 0 else group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            'Collection Efficiency%': collections_efficiency.loc[collections_efficiency['Customer'] == customer, 'Collection_Efficiency%'].iloc[0],
            'Last 12 months max collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().max()).tail(12).max(),
            'Last 12 months avg collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().sum()).tail(12).mean(),
            'max collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().max(),
            'avg collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().mean(),
            'Std_Credit_Limit': std_credit_limit
            # 'State': customer_details.get('CM_State', ''),
            
            # 'avg_closing_bal_last_12_months': scoring_df[scoring_df['Customer']== customer]['Average Bal in the last 365 days'].values[0] if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            # 'max_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().max(),
            
            
            # 'median collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().median(),
            # 'min_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().min(),
            
        
            # 'avg_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().mean(),
            
            # 'Last 12 months median collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().sum()).tail(12).median(),
            # 'min collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().min(),
            # 'max_missed_payment': max_default_value,
            # 'Average Number of invoices per year': avg_number_of_invoices,
            # 'Median Monthly Sales_Total': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().median(),
            
            
            # 'Average Transaction Size': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('S_Billing Document')['S_Total Sales'].sum().mean(),
            

            # 'Customer Name': customer_details.get('CM_Name 1', ''),
            # 
            # 'Tenure': tenure,
            # 
            # 'Sale Tenure Days': group_df['Sale Tenure_days'].iloc[0],
            # 'Total Sales' : group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            # 'Total Collection' : group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum(),
            # 'Zone': group_df['CM_ZONE'].iloc[0],
            # 'Opening Balance': group_df['CM_Opening Balances'].iloc[0],
            # 
            # 'F_old': rfm_details.get('Frequency', ''),
            # 'M_old': rfm_details.get('Monetary Value', ''),
            # 'F': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_total'].values[0],
            # 'M': trans_df[(trans_df['Customer'] == customer) & (trans_df['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum(),
            # 'Frequency_365': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_365'].values[0],
            # 'Monetary Value_365': trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum(),
            # 'days_since_first_activity': pd.to_datetime(cutoff_date) - first_activity,
            # 'cash/credit': "cash" if (knock_df[knock_df['Customer'] == customer]['Payment Term'] == '0001').all() else "credit",
            # 'cash/credit_365': "none" if knock_df_last_year.empty else ("cash" if (knock_df_last_year[knock_df_last_year['Customer'] == customer]['Payment Term'] == '0001').all() else "credit"),
            # 'Most Frequent Payment Term': scoring_df[scoring_df['Customer'] == customer]['max payment term used'].values[0],
            # 'Number of invoices Total': invoices_total,
            
            # 'Number of invoices 365': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_365'].values[0],
            # 'Number of Outstanding Invoices': no_of_outstanding_invoices_total,
            # 'Number_of_defaults_total': total_defaults,
            # 'avg_defaults': int(avg_defaults),
            # 'avg_monthly_defaults': avg_monthly_default,
            # 'Number_of_defaults_365': scoring_df[scoring_df['Customer']== customer]['number_of_defaults_365'].values[0],
            # 'Missed Payment Ratio': round((total_defaults/invoices_total), 2) if invoices_total != 0 else -1,
            
            # 'min_missed_payment': min_default_value,
            # 'Average Missed Payment Value': avg_missed_payment_val,
            # 'Missed Payment Ratio_365': round(((scoring_df[scoring_df['Customer']== customer]['number_of_defaults_365'].values[0])/(scoring_df[scoring_df['Customer']== customer]['Number of invoices_365'].values[0])), 2),
            
            # 
            # 'Profit Ratio last 12 months': round((total_profit_last_12_months/total_sales_last_12_months), 2),
            # 'Total Profit% total': scoring_df[scoring_df['Customer'] == customer]['GM%/total_profit_percent(GM/NetRev)'].values[0],
            # 'Total Profit in 365 days': scoring_df[scoring_df['Customer'] == customer]['GM_365/TotalProfit_365'].values[0],
            # 'Total Profit % 365': scoring_df[scoring_df['Customer'] == customer]['GM%_365/TotalProfit%_365'].values[0],
            # 'Average Past Due Days in 365': knock_df_last_year['Past Due Days'].mean() if not knock_df_last_year.empty else 0,
            
            # 
            # 'Net Sale Amount (365)': scoring_df[scoring_df['Customer'] == customer]['Net Sale Amount (365)'].values[0],
            # 'average_closing_bal in 365 days': scoring_df[scoring_df['Customer']== customer]['average_closing_bal'].values[0],
            # 'Net Revenue in 365 days': scoring_df[scoring_df['Customer']== customer]['net_rev_365'].values[0],
            # 
            
            
            
            
            
            # 'Legal/WOP Consideration': scoring_df[scoring_df['Customer']== customer]['Legal/WOP Consideration'].values[0],
            
            # 'Last 12 months min sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].min().tail(12).min(),
            
            
            
            # 'min sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].min(),
            
            # 'median sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].median(),
            
            # 'Average Transaction Size 365': trans_df_last_year[(trans_df_last_year['S_Billing Type'] == 'ZINV')].groupby('S_Billing Document')['S_Total Sales'].sum().mean(),
            # 'Top Material purchased': top_materials[-1] if len(top_materials) >= 1 else '',
            # 'Top 2nd Material purchased': top_materials[-2] if len(top_materials) >= 2 else '',
            # 'Top 3rd Material purchased': top_materials[-3] if len(top_materials) >= 3 else '',
            
            
            # 'Median Sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].median(),
            # 'Avg Monthly Sales_365': (trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum() / max(12, 1)) if group_df['Sale Tenure_days'].iloc[0] >= 365 else (group_df['S_Total Sales'].sum() / max((group_df['Sale Tenure_days'] / 30).values[0], 1)),  
            
            
            # 'Each Month Average Sales_365': average_each_month_last_year['Average Sales per Transaction'].mean() if not average_each_month_last_year.empty else 0,
            # 'Each Month Average Sales': average_each_month['Average Sales per Transaction'].mean() if not average_each_month.empty else 0,
            # 'Avg Daily Sales': (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / group_df['Sale Tenure_days'].iloc[0])
            # if group_df['Sale Tenure_days'].iloc[0] != 0 else 0,
            
            # 'Collection Efficiency%_365' : (
            #     collections_efficiency_last_year.loc[collections_efficiency_last_year['Customer'] == customer, 'Collection_Efficiency%'].iloc[0]
            #     if not collections_efficiency_last_year.loc[collections_efficiency_last_year['Customer'] == customer, 'Collection_Efficiency%'].empty
            #     else None
            # ),
            
            # 'Last 12 months min collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().min()).tail(12).min(),
            
            
            
            
            
            
            # 'Total Amount Defaulted': knock_df[(knock_df['Customer'] == customer) & knock_df['Default'] == 1]['Sales Amount'].sum(),
            # 'default-to-non_default': round((total_defaults/not_defaults), 2) if not_defaults != 0 else 0,
            # 'Total Amount Defaulted_365': 0 if knock_df_last_year.empty else knock_df_last_year[(knock_df_last_year['Customer'] == customer) & knock_df_last_year['Default'] == 1]['Sales Amount'].sum(),
            # 'Default Rate%': round((knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].sum() / (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() or 1)) * 100, 2),
            # 'Default Rate%_365': 0 if knock_df_last_year.empty else round((knock_df_last_year[(knock_df_last_year['Customer'] == customer) & (knock_df_last_year['Default'] == 1)]['Sales Amount'].sum() / (trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum() or 1)) * 100, 2),
            
        })

    features = pd.DataFrame(features_list)
    return features


def generate_features_old_model(trans_df, knock_df, profit_df, customer_df, scoring_df):
    cutoff_date = pd.to_datetime('2024-03-31')
    # start_date = cutoff_date - timedelta(days=6*30)
    start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')
    features_list = []
    customer_dict = customer_df.set_index('Customer')[['CM_Name 1', 'CM_ASM Code', 'CM_State']].to_dict('index')
# for state in tqdm(states):
#     profit_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_{state}.pkl')
#     trans_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_{state}.pkl')
    trans_df = trans_df.loc[:, ~trans_df.columns.duplicated()]
#     # trans_df = trans_df[trans_df['Customer'] == '11001090']
#     knock_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_{state}.pkl')
#     # trans_df.fillna(0, inplace = True)

    trans_df.loc[:, ~trans_df.columns.isin(['S_Billing Date', 'C_Posting Date'])] = trans_df.loc[:, ~trans_df.columns.isin(['S_Billing Date', 'C_Posting Date'])].fillna(0)
    knock_df.loc[:, ~knock_df.columns.isin(['Billing Date', 'Posting Date'])] = knock_df.loc[:, ~knock_df.columns.isin(['Billing Date', 'Posting Date'])].fillna(0)

    # knock_df.fillna(0, inplace = True)
    profit_df.fillna(0, inplace = True)
    knock_df['Billing Date'] = pd.to_datetime(knock_df['Billing Date'])
    knock_df['Posting Date'] = pd.to_datetime(knock_df['Posting Date'])
    knock_df['cash/credit'] = np.where(knock_df['Payment Term'] == '0001', 'cash', 'credit')
    knock_df = knock_df[knock_df['cash/credit'] != 'cash']
    
    # scoring_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\scoring_v2\scoring_v2_{state}.pkl')
    # scoring_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\scoring_v2\scoring_v2_5_customers.pkl')
    # cl_std_details = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\cl_std_v2\cl_std_{state}.pkl')
    trans_df = trans_df[trans_df['Customer'].isin(profit_df['Customer'])]
    # trans_df = trans_df[trans_df['Time Stamp'] <= cutoff_date]
    trans_df = trans_df[trans_df['Time Stamp'] >= start_cutoff]
    knock_df = knock_df[(knock_df['Billing Date'] >= start_cutoff) & (knock_df['Posting Date'] >= start_cutoff) ]
    profit_df = profit_df[profit_df['Fiscal Year'] >= 2022]
    trans_df = trans_df[trans_df['S_Canceled'] != 'X']
    trans_df = trans_df[~trans_df['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
    trans_df = trans_df[trans_df['CM_ZHIRE-ZNAME'] != "B2B"]

    knock_df['Default'] = knock_df.apply(defaults, axis=1) if not knock_df.empty else None
    trans_df['Last Activity'] = trans_df['Customer'].map(trans_df.groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['First Activity'] = trans_df['Customer'].map(trans_df.groupby('Customer')['Time Stamp'].min().to_dict())
    
    trans_df['Sale First Activity'] = trans_df['Customer'].map(trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['Time Stamp'].min().to_dict())
    trans_df['Sale Last Activity'] = trans_df['Customer'].map(trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['Sale First Activity']= pd.to_datetime(trans_df['Sale First Activity'])
    trans_df['Sale Last Activity']= pd.to_datetime(trans_df['Sale Last Activity'])
    
    trans_df['Collect First Activity'] = trans_df['Customer'].map(trans_df[trans_df['C_Document Type'] == 'DZ'].groupby('Customer')['Time Stamp'].min().to_dict())
    trans_df['Collect Last Activity'] = trans_df['Customer'].map(trans_df[trans_df['C_Document Type'] == 'DZ'].groupby('Customer')['Time Stamp'].max().to_dict())
    trans_df['Collect First Activity']= pd.to_datetime(trans_df['Collect First Activity'])
    trans_df['Collect Last Activity']= pd.to_datetime(trans_df['Collect Last Activity'])
    # trans_df['Tenure_days'] = (trans_df['Last Activity'] - start_cutoff).dt.days
    trans_df['Sale Tenure_days'] = (trans_df['Sale Last Activity'] - trans_df['Sale First Activity']).dt.days
    trans_df['Collect Last Activity'] = pd.to_datetime(trans_df['Collect Last Activity'], errors='coerce')
    trans_df['Collect First Activity'] = pd.to_datetime(trans_df['Collect First Activity'], errors='coerce')
    trans_df['Collect Tenure_days'] = (trans_df['Collect Last Activity'] - trans_df['Collect First Activity']).dt.days
    
    

    # trans_df['Running Outstanding'] = pd.to_numeric(trans_df['Running Outstanding'])
    # trans_df['Credit Limit'] = pd.to_numeric(trans_df['Credit Limit'])
    # trans_df['Utilization %'] = trans_df.apply(lambda row: utilization_calculate(row['Running Outstanding'], row['Credit Limit']), axis=1)
    # avg_utilization = trans_df.groupby('Customer')['Utilization %'].mean().reset_index()
    # max_utilization = trans_df.groupby('Customer')['Utilization %'].max().reset_index()
    trans_df['S_Billing Date'] = pd.to_datetime(trans_df['S_Billing Date'], errors='coerce')
    collections_efficiency = collection_efficiency(trans_df)
    knock_df['Billing Date'] = pd.to_datetime(knock_df['Billing Date'], errors='coerce')
    # customers = ['11005411', '11003891', '11005794', '11001090', '11000576']
    # customers = ['11005411', '11001090']
    # customers = ['11005411']
    # trans_df = trans_df[trans_df['Customer'].isin(customers)]
    # Group by Customer_ID and calculate the maximum billing date for each customer
    max_date_df = trans_df[trans_df['S_Billing Type'] == 'ZINV'].groupby('Customer')['S_Billing Date'].max().reset_index()
    # Filter customers whose max billing date is after 31st March 2023
    filtered_customers = max_date_df[max_date_df['S_Billing Date'] > pd.to_datetime('31-03-2023', dayfirst=True)]
    trans_df = trans_df[trans_df['Customer'].isin(filtered_customers['Customer'].unique())]
    grouped_customers = trans_df.groupby('Customer')

    for customer, group_df in tqdm(grouped_customers, delay = 2):
    # for customer in grouped_customers:
        # group_df = grouped_customers[grouped_customers['Customer'] == customer]
        group_df['yearmonth'] = group_df['Time Stamp'].dt.to_period('M')
        last_sale_activity = group_df[group_df['S_Billing Type'] == 'ZINV']['S_Billing Date'].max()
        
        first_sale_activity = group_df[group_df['S_Billing Type'] == 'ZINV']['S_Billing Date'].min()
        # profit_df = profit_df[profit_df['Customer']== customer]
        last_one_year = last_sale_activity - pd.Timedelta(days=365)
        trans_df_last_year = group_df[(group_df['Time Stamp']>=last_one_year)]
        collections_efficiency_last_year = collection_efficiency(trans_df_last_year)
        last_sale_activity = pd.to_datetime(last_sale_activity, errors='coerce')
        if pd.notna(last_sale_activity):
            knock_df_last_year = knock_df[(knock_df['Customer'] == customer) & (knock_df['Billing Date'] >= last_one_year)]
        else:
            knock_df_last_year = pd.DataFrame()  # or handle appropriately
        
        # Get customer details
        customer_details = customer_dict.get(customer, {})
        # gst_details = gst_dict.get(customer, {})
        # rfm_details = rfm_dict.get(int(customer), {})
        
        last_activity = group_df['Time Stamp'].max()
        first_activity = group_df['Time Stamp'].min()
        # tenure = (trans_df['Time Stamp'].max() - trans_df['Time Stamp'].min()).days / 365 if not trans_df.empty else 0
        tenure = round(((group_df['Time Stamp'].max() - group_df['Time Stamp'].min()).days / 365), 2) if not group_df.empty else 0
        # def format_tenure(tenure):
        # # Check the decimal part of the tenure
        #     if tenure - int(tenure) >= 0.5:
        #         return int(tenure) + 1
        #     return int(tenure)
        # tenure = format_tenure(tenure)
        year_wise_sales = group_df.groupby('Year')['S_Total Sales'].sum().reset_index()
        # knock_df['Default'] = knock_df.apply(defaults, axis=1)
        # knock_df['Threshold'] = knock_df['Due Date'] + pd.Timedelta(days=90)
        # knock_df['Default'] = ((knock_df['Posting Date'] > knock_df['Threshold']) &
            # (knock_df['Sales Amount'] - knock_df['Collections Amount']).abs() > 10000 ) & (knock_df['Billing Type'] != 'ZIDR')
        grouped_defaults = knock_df[knock_df['Customer'] == customer].groupby('Sale_ID')['Default'].sum().reset_index()
        total_defaults = 0
        for idx in grouped_defaults.index:
            if grouped_defaults.loc[idx, 'Default'] >= 1:
                total_defaults += 1
        number_of_defaults = total_defaults
        top_materials = [
                mat for mat in group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('S_Material')['S_Total Sales'].max().sort_values(ascending=False).index
                if mat.startswith('F-')
        ]
        # no. of invoices
        invoices = group_df[group_df['S_Billing Type'] == 'ZINV']
        invoices_total = len(invoices['S_Billing Document'].unique())
        avg_number_of_invoices = round((invoices_total/tenure), 2) if tenure != 0 else 0
        if avg_number_of_invoices > invoices_total:
            avg_number_of_invoices = invoices_total
        elif avg_number_of_invoices < 1:
            avg_number_of_invoices = 1
        else:
            avg_number_of_invoices = round(avg_number_of_invoices)
        no_of_outstanding_invoices_total = number_of_outstanding_invoices(group_df, knock_df[knock_df['Customer'] == customer])

        # for total
        monthly_sales = group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Total Sales'].sum().reset_index()

        monthly_transactions = group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Billing Document'].count().reset_index()
        monthly_sales['S_Total Sales'] = pd.to_numeric(monthly_sales['S_Total Sales'], errors='coerce')
        monthly_transactions['S_Billing Document'] = pd.to_numeric(monthly_transactions['S_Billing Document'], errors='coerce')
        monthly_sales.fillna(0, inplace=True)
        monthly_transactions.replace(0, 1, inplace=True)  # Avoid division by zero
        average_each_month = monthly_sales.copy()
        average_each_month['Average Sales per Transaction'] = monthly_sales['S_Total Sales'] / monthly_transactions['S_Billing Document']
        # for the last year
        monthly_sales_last_year = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Total Sales'].sum().reset_index()
        monthly_transactions_last_year = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV'].groupby('Month')['S_Billing Document'].count().reset_index()
        monthly_sales_last_year['S_Total Sales'] = pd.to_numeric(monthly_sales_last_year['S_Total Sales'], errors='coerce')
        monthly_transactions_last_year['S_Billing Document'] = pd.to_numeric(monthly_transactions_last_year['S_Billing Document'], errors='coerce')
        monthly_sales_last_year.fillna(0, inplace=True)
        monthly_transactions_last_year.replace(0, 1, inplace=True)  # Avoid division by zero
        average_each_month_last_year = monthly_sales_last_year.copy()
        average_each_month_last_year['Average Sales per Transaction'] = monthly_sales_last_year['S_Total Sales'] / monthly_transactions_last_year['S_Billing Document']
        
        not_defaults = abs(invoices_total - total_defaults)

        avg_defaults = total_defaults/tenure if tenure != 0 else total_defaults
        knock_df['yearmonth'] = knock_df['Billing Date'].fillna(knock_df['Posting Date']).dt.to_period('M')
        avg_monthly_default = knock_df[knock_df['Customer'] == customer].groupby('yearmonth')['Default'].sum().mean()
        if avg_defaults == 0:
            avg_monthly_default = 0
        elif avg_defaults < 1:
            avg_monthly_default = 1
        else:
            avg_monthly_default = round(avg_monthly_default)

        max_default_value = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].max()
        min_default_value = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].min()
        total_profit = profit_df['Gross Margin'].sum()

        avg_missed_payment_val = knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].mean()


        # total_sales_last_12_months = trans_df_last_year[trans_df_last_year['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum()
        # if trans_df_last_year.empty:
        #     total_profit_last_12_months = 0 if profit_df.empty else profit_df['Gross Margin'].sum()
        # else:
        #     df1 = scoring.profitability_last_year(trans_df_last_year)
        #     total_profit_last_12_months = df1['Gross Margin'].sum()
        # number_of_transactions = len(group_df['S_Billing Document'].unique())
        # Append features for the customer
        # avg_transactions = group_df.groupby('S_Billing Document')['S_Total Sales'].sum()
        total_sales = group_df['S_Total Sales'].abs().sum()
        threshold = last_sale_activity - pd.Timedelta(days = 365)
        threshold = pd.to_datetime(threshold)
        std_credit_limit = group_df[(group_df['Time Stamp'] >= threshold) & (group_df['Time Stamp'] <= last_sale_activity)]['Running Outstanding'].max()

        features_list.append({
            'Customer': customer,
            'ASM Code': customer_details.get('CM_ASM Code', ''),
            'Last 12 months max sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].max().tail(12).max(),
            'avg_closing_bal_last_12_months': scoring_df[scoring_df['Customer']== customer]['Average Bal in the last 365 days'].values[0] if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            'max_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().max(),
            'Last 12 months avg collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().sum()).tail(12).mean(),
            'max collection': group_df['C_Amount in Balance Transaction Currency'].abs().max(),
            'median collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().median(),
            'min_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().min(),
            'net_rev_365': scoring_df[scoring_df['Customer']== customer]['Net Revenue 365'].values[0] if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            'avg collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().mean(),
            'max sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].max(),
            'Total Profit': profit_df[profit_df['Customer'] == customer]['Gross Margin'].sum(),
            'Avg Yearly Sales': (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / (group_df['Sale Tenure_days'].iloc[0] / 365))
            if group_df['Sale Tenure_days'].iloc[0] >= 365 else group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            'avg_closing_bal': group_df.groupby('yearmonth')['Running Outstanding'].last().mean(),
            'Avg Yearly Collections': (group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum() / (group_df['Collect Tenure_days'].iloc[0] / 365))
            if group_df['Collect Tenure_days'].iloc[0] >= 365 else group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum(),
            'Last 12 months median collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().sum()).tail(12).median(),
            'min collection': group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].abs().min(),
            'max_missed_payment': max_default_value,
            'Average Number of invoices per year': avg_number_of_invoices,
            'Median Monthly Sales_Total': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().median(),
            'Average Past Due Days Total': round((knock_df[knock_df['Customer'] == customer]['Past Due Days'].mean()), 0),
            'Last 12 months avg sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().tail(12).mean(),
            'Average Transaction Size': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby('S_Billing Document')['S_Total Sales'].sum().mean(),
            'Std_Credit_Limit': std_credit_limit
            

            # 'Recency': int(str(cutoff_date - last_sale_activity).split(" days")[0].strip()) if not pd.isna(last_sale_activity) else 0,
            
            # 'Profit Ratio': round((profit_df[profit_df['Customer'] == customer]['Gross Margin'].sum()/total_sales), 2) if total_sales != 0 else 0,
            
            # 'Median Past Due Days Total': round((knock_df[knock_df['Customer'] == customer]['Past Due Days'].median()), 0),
            # 'debtor_days': int(scoring_df[scoring_df['Customer']== customer]['Debtor Days'].values[0]) if not scoring_df[scoring_df['Customer'] == customer].empty else 0,
            
            
            
            # 'Last 12 months median sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].sum().tail(12).median(),
            
            # 'avg sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].mean(),
            
            
            # 'Avg Monthly Sales_Total': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / (group_df['Sale Tenure_days']/30).values[0] if group_df['Sale Tenure_days'].iloc[0] > 0 else group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            # 'Collection Efficiency%': collections_efficiency.loc[collections_efficiency['Customer'] == customer, 'Collection_Efficiency%'].iloc[0],
            # 'Last 12 months max collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().max()).tail(12).max(),
            
            
            
            

            # 'State': customer_details.get('CM_State', ''),
            
            
            
            
            
            
            
            
        
            
            
            
            
            
            
            
            
            
            
            

            # 'Customer Name': customer_details.get('CM_Name 1', ''),
            # 
            # 'Tenure': tenure,
            # 
            # 'Sale Tenure Days': group_df['Sale Tenure_days'].iloc[0],
            # 'Total Sales' : group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum(),
            # 'Total Collection' : group_df[group_df['C_Document Type'] == 'DZ']['C_Amount in Balance Transaction Currency'].sum(),
            # 'Zone': group_df['CM_ZONE'].iloc[0],
            # 'Opening Balance': group_df['CM_Opening Balances'].iloc[0],
            # 
            # 'F_old': rfm_details.get('Frequency', ''),
            # 'M_old': rfm_details.get('Monetary Value', ''),
            # 'F': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_total'].values[0],
            # 'M': trans_df[(trans_df['Customer'] == customer) & (trans_df['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum(),
            # 'Frequency_365': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_365'].values[0],
            # 'Monetary Value_365': trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum(),
            # 'days_since_first_activity': pd.to_datetime(cutoff_date) - first_activity,
            # 'cash/credit': "cash" if (knock_df[knock_df['Customer'] == customer]['Payment Term'] == '0001').all() else "credit",
            # 'cash/credit_365': "none" if knock_df_last_year.empty else ("cash" if (knock_df_last_year[knock_df_last_year['Customer'] == customer]['Payment Term'] == '0001').all() else "credit"),
            # 'Most Frequent Payment Term': scoring_df[scoring_df['Customer'] == customer]['max payment term used'].values[0],
            # 'Number of invoices Total': invoices_total,
            
            # 'Number of invoices 365': scoring_df[scoring_df['Customer'] == customer]['Number of invoices_365'].values[0],
            # 'Number of Outstanding Invoices': no_of_outstanding_invoices_total,
            # 'Number_of_defaults_total': total_defaults,
            # 'avg_defaults': int(avg_defaults),
            # 'avg_monthly_defaults': avg_monthly_default,
            # 'Number_of_defaults_365': scoring_df[scoring_df['Customer']== customer]['number_of_defaults_365'].values[0],
            # 'Missed Payment Ratio': round((total_defaults/invoices_total), 2) if invoices_total != 0 else -1,
            
            # 'min_missed_payment': min_default_value,
            # 'Average Missed Payment Value': avg_missed_payment_val,
            # 'Missed Payment Ratio_365': round(((scoring_df[scoring_df['Customer']== customer]['number_of_defaults_365'].values[0])/(scoring_df[scoring_df['Customer']== customer]['Number of invoices_365'].values[0])), 2),
            
            # 
            # 'Profit Ratio last 12 months': round((total_profit_last_12_months/total_sales_last_12_months), 2),
            # 'Total Profit% total': scoring_df[scoring_df['Customer'] == customer]['GM%/total_profit_percent(GM/NetRev)'].values[0],
            # 'Total Profit in 365 days': scoring_df[scoring_df['Customer'] == customer]['GM_365/TotalProfit_365'].values[0],
            # 'Total Profit % 365': scoring_df[scoring_df['Customer'] == customer]['GM%_365/TotalProfit%_365'].values[0],
            # 'Average Past Due Days in 365': knock_df_last_year['Past Due Days'].mean() if not knock_df_last_year.empty else 0,
            
            # 
            # 'Net Sale Amount (365)': scoring_df[scoring_df['Customer'] == customer]['Net Sale Amount (365)'].values[0],
            # 'average_closing_bal in 365 days': scoring_df[scoring_df['Customer']== customer]['average_closing_bal'].values[0],
            # 'Net Revenue in 365 days': scoring_df[scoring_df['Customer']== customer]['net_rev_365'].values[0],
            # 
            
            
            
            
            
            # 'Legal/WOP Consideration': scoring_df[scoring_df['Customer']== customer]['Legal/WOP Consideration'].values[0],
            
            # 'Last 12 months min sales': group_df[group_df['S_Billing Type'] == 'ZINV'].groupby(['Year', 'Month'])['S_Total Sales'].min().tail(12).min(),
            
            
            
            # 'min sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].min(),
            
            # 'median sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].median(),
            
            # 'Average Transaction Size 365': trans_df_last_year[(trans_df_last_year['S_Billing Type'] == 'ZINV')].groupby('S_Billing Document')['S_Total Sales'].sum().mean(),
            # 'Top Material purchased': top_materials[-1] if len(top_materials) >= 1 else '',
            # 'Top 2nd Material purchased': top_materials[-2] if len(top_materials) >= 2 else '',
            # 'Top 3rd Material purchased': top_materials[-3] if len(top_materials) >= 3 else '',
            
            
            # 'Median Sales': group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].median(),
            # 'Avg Monthly Sales_365': (trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum() / max(12, 1)) if group_df['Sale Tenure_days'].iloc[0] >= 365 else (group_df['S_Total Sales'].sum() / max((group_df['Sale Tenure_days'] / 30).values[0], 1)),  
            
            
            # 'Each Month Average Sales_365': average_each_month_last_year['Average Sales per Transaction'].mean() if not average_each_month_last_year.empty else 0,
            # 'Each Month Average Sales': average_each_month['Average Sales per Transaction'].mean() if not average_each_month.empty else 0,
            # 'Avg Daily Sales': (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() / group_df['Sale Tenure_days'].iloc[0])
            # if group_df['Sale Tenure_days'].iloc[0] != 0 else 0,
            
            # 'Collection Efficiency%_365' : (
            #     collections_efficiency_last_year.loc[collections_efficiency_last_year['Customer'] == customer, 'Collection_Efficiency%'].iloc[0]
            #     if not collections_efficiency_last_year.loc[collections_efficiency_last_year['Customer'] == customer, 'Collection_Efficiency%'].empty
            #     else None
            # ),
            
            # 'Last 12 months min collection': group_df[group_df['C_Document Type'] == 'DZ'].groupby(['Year', 'Month'])['C_Amount in Balance Transaction Currency'].apply(lambda x: x.abs().min()).tail(12).min(),
            
            
            
            
            
            
            # 'Total Amount Defaulted': knock_df[(knock_df['Customer'] == customer) & knock_df['Default'] == 1]['Sales Amount'].sum(),
            # 'default-to-non_default': round((total_defaults/not_defaults), 2) if not_defaults != 0 else 0,
            # 'Total Amount Defaulted_365': 0 if knock_df_last_year.empty else knock_df_last_year[(knock_df_last_year['Customer'] == customer) & knock_df_last_year['Default'] == 1]['Sales Amount'].sum(),
            # 'Default Rate%': round((knock_df[(knock_df['Customer'] == customer) & (knock_df['Default'] == 1)]['Sales Amount'].sum() / (group_df[group_df['S_Billing Type'] == 'ZINV']['S_Total Sales'].sum() or 1)) * 100, 2),
            # 'Default Rate%_365': 0 if knock_df_last_year.empty else round((knock_df_last_year[(knock_df_last_year['Customer'] == customer) & (knock_df_last_year['Default'] == 1)]['Sales Amount'].sum() / (trans_df_last_year[(trans_df_last_year['Customer'] == customer) & (trans_df_last_year['S_Billing Type'] == 'ZINV')]['S_Total Sales'].sum() or 1)) * 100, 2),
            
        })

    features = pd.DataFrame(features_list)
    return features
