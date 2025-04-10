import pandas as pd 
import os 
from tqdm import tqdm 
from datetime import datetime
import numpy as np
import uuid


data_folder = r".\Data_for_view"

cutoff_date = pd.to_datetime('31-03-2024')
start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')

def clean_sales(sales_df):
    sales_df.rename(columns={
    'Bill.Qty in SKU': 'Billing Qty in SKU'})
    # converting billing date to datetime
    sales_df['Billing Date'] = pd.to_datetime(sales_df['Billing Date'])
    # removing sales having billing date <= cutoff date
    sales_df = sales_df[sales_df['Billing Date'] <= cutoff_date]
    # removing 00 from the sales_df and customer master (doen before, just to be sure!)
    sales_df['Customer'] = sales_df['Customer'].astype(str).str.lstrip('0')
    # getting only common customers in sales df
    sales_df['Customer'] = sales_df['Customer'].astype(str)
    customer_df['Customer'] = customer_df['Customer'].astype(str)
    sales_df = sales_df[sales_df['Customer'].isin(customer_df['Customer'])]
    sales_df = sales_df.loc[:, ~sales_df.columns.duplicated()]
    sales_df.rename(columns={
        'Bill.Qty in SKU': 'Billing Qty in SKU',
        'Net Value': 'Net Value.1',
        'Tax Amount': 'Tax Amount.1',
        'Payt Terms': 'Payment terms',
        'Billing Doc.': 'Billing Document',
        'Item Descr.': 'Item Description'
    }, inplace = True)
    return sales_df

def clean_collections(collections_df):
    # converting posting date to datetime value
    collections_df['Posting Date'] = pd.to_datetime(collections_df['Posting Date'])
    # taking posting date <= cutoff date in collections_df
    collections_df = collections_df[collections_df['Posting Date'] <= cutoff_date]
    # removing 00 from the prefixes on all the customer ids in collections df
    collections_df['Customer'] = collections_df['Customer'].str.lstrip('0')
    # common customers between collections and customer master 
    common_customers_collections = set(collections_df['Customer']).intersection(set(customer_df['Customer']))
    # taking only common customers in collections_df 
    collections_df = collections_df[collections_df['Customer'].isin(common_customers_collections)]
    collections_df.rename(columns = {
        'Amnt in Bal Tr Crcy': 'Amount in Balance Transaction Currency',
        'Payt terms': 'Payment Terms'
    }, inplace = True)
    return collections_df

Billing_type_df = pd.read_excel(r'.\Data_for_view\Billing Type.xls')
CL_log_df = pd.read_pickle(r'.\Data_for_view\credit_limit_history_log.pkl')
CL_log_df.rename(columns =
    {'Customer': 'Customer Code',
    'Update_Date': 'Date',
    'VALUE_OLD': 'Old value',
    'VALUE_NEW': 'New value'}
, inplace = True)
customer_df = pd.read_pickle(r'.\Data_for_view\customer_master_clean_v2')
cutoff_date = pd.to_datetime('2024-03-31')
customer_df.rename(columns = {
    'CM_Opening_Balance' : 'CM_Opening Balances',
    'CD_Credit Limit': 'CM_Credit Limit',
    'CM_ZName': 'CM_ZHIRE-ZNAME',
    'CM_State': 'CM_State_Code',
    'CM_Region_Name': 'CM_State'    
}, inplace = True)

# sales_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v2\Sales_Statewise_Clean_Pickle\sales_Uttar Pradesh.pkl')
# # Remove 'S_' from column names
# sales_df.columns = [col.replace('S_', '') for col in sales_df.columns]
# collections_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v2\Collections_Statewise_Clean_Pickle\collections_Uttar Pradesh.pkl')
# collections_df.columns = [col.replace('CO_', '') for col in collections_df.columns]
# sales_df = clean_sales(sales_df[sales_df['Customer'] == '21000736'])
# collections_df = clean_collections(collections_df[collections_df['Customer'] == '21000736'])

def find_usefull_columns(df):
    
    unique_columns_columns = []
    columns_to_ignore = []
    columns_with_zero = []
    columns_with_one_unique_value= []
    date_time_columns = df.select_dtypes(include=['datetime64[ns]']).columns
    
    for column in tqdm(df.columns):
        
        if column in date_time_columns:continue
        # Check if the column contains only null values
        if df[column].isnull().all():
            columns_to_ignore.append(column)
            continue
        
        # Check if the column has unique values
        if df[column].nunique() == len(df):
            unique_columns_columns.append(column)
            continue
        
        if all(df[column].unique() == [0]):
            columns_with_zero.append(column)
            continue
        if len(df[column].unique())==1:
            columns_with_one_unique_value.append(column)
            continue
    
    result_dict = {'usefull_columns' : list(set(df.columns) -set(columns_with_one_unique_value) - set(columns_to_ignore) - set(columns_with_zero)),
                    "useless_columns" : columns_to_ignore + columns_with_zero + columns_with_one_unique_value,
                    "Columns_with_one_unique_value":columns_with_one_unique_value,
                    'columns_with_unique':unique_columns_columns,
                    'columns_to_ignore' : columns_to_ignore, 
                    'columns_with_zero' : columns_with_zero}
    
    for key , value in result_dict.items():
        result_dict[key] = sorted(value)
    return result_dict
    

from datetime import timedelta
# Convert 'Time.1' to timedelta
def time_to_timedelta(time_str):
    time_str = str(time_str)
    if time_str == "nan":
        time_str = "00:00:00"
    try:
        h, m, s = map(int, time_str.split(':'))
    except:
        h,m,s = 0,0,0
    return timedelta(hours=h, minutes=m, seconds=s)


# states = ['Andhra Pradesh', 'West Bengal']
states = ['Andhra Pradesh', 'Uttar Pradesh', 'Gujarat', 'Telangana', 'Assam', 'West Bengal', 'Punjab', 'Haryana',
            'Bihar', 'Maharashtra', 'Rajasthan', 'Orissa', 'Madhya Pradesh', 'Himachal Pradesh',
            'Chhattisgarh', 'Karnataka', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand', 'others']
# states = ['Uttar Pradesh', 'Haryana', 'Gujarat', 'Telangana', 'Punjab', 'Assam',
#             'Bihar', 'Maharashtra', 'Rajasthan', 'Orissa', 'Madhya Pradesh', 'Himachal Pradesh', 
#            'Chhattisgarh', 'Karnataka', 'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand']
# states = ['Uttar Pradesh', 'Haryana', 'Gujarat', 'Telangana', 'Assam',
#             'Bihar', 'Maharashtra', 'Rajasthan', 'Orissa', 'Madhya Pradesh', 'Himachal Pradesh', 
#             'Chhattisgarh', 'Karnataka', 'Assam', 'Uttarakhand', 'Tamil Nadu', 'Delhi', 'Jharkhand']
others = ['Puducherry', 'Chandigarh', 'Tripura', 'Dadra & Nagar Haveli', 'Meghalaya', 'Jammu & Kashmir', 'Kerala', 'Goa', 'Manipur']


# def trans_df_creation(Sales_df ,Collection_df,Billing_type_df=Billing_type_df,CL_log_df=CL_log_df):
    
#     # res_dict = find_usefull_columns(Sales_df)
#     # Sales_df = Sales_df[res_dict['usefull_columns']]
#     # Billing Type in the Sales_df 
#     Billing_type_df = Billing_type_df[['Billing Type', 'Description', 'Change']]
#     Sales_df['Billing Description'] = ""
#     Sales_df['Billing D/C'] = ''
#     Sales_df['Billing Description'] = Sales_df['Billing Type'].map(dict(zip(Billing_type_df['Billing Type'],Billing_type_df['Description'])))
#     Sales_df['Billing D/C'] = Sales_df['Billing Type'].map(dict(zip(Billing_type_df['Billing Type'],Billing_type_df['Change'])))
#     Sales_df[Sales_df['Billing Description'].isna()][['Billing D/C','Net Value.1']]
#     Sales_df['Total Sales']=(Sales_df['Net Value.1'] + Sales_df['Tax Amount.1'])*Sales_df['Billing D/C']
#     customer_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v3\customer_master_clean_v2')
#     Sales_df['ZHIRE-ZNAME'] = Sales_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_ZHIRE-ZNAME'])))
#     Sales_df = Sales_df[Sales_df['ZHIRE-ZNAME']!="B2B"]
#     Sales_df.drop(columns='ZHIRE-ZNAME',inplace=True)
#     Sales_df.reset_index(drop=True,inplace=True)
#     Sales_df['Billing Date'] = pd.to_datetime(Sales_df['Billing Date'])
#     # Combine 'Created on.1' and 'Time.1'
#     Sales_df['Time Stamp'] = pd.to_datetime(Sales_df['Billing Date'].map(str))
#     # Sales_df.rename(columns={'Payer':'Customer'},inplace=True)
#     Sales_df.columns = [col if (col == 'Customer' or col == 'Time Stamp') else "S_" + col for col in Sales_df.columns]
#     # Sales_df['S_State'] = Sales_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_State'])))
#     # Collection
#     # res_dict = find_usefull_columns(Collection_df)
#     # Collection_df = Collection_df[res_dict['usefull_columns']]
#     # Collection_df['State'] = Collection_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_State'])))
#     Collection_df['ZHIRE-ZNAME'] = Collection_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_ZHIRE-ZNAME'])))
#     Collection_df = Collection_df[Collection_df['ZHIRE-ZNAME']!='B2B']
#     Collection_df.drop(columns='ZHIRE-ZNAME',inplace=True)
#     Collection_df.reset_index(drop=True,inplace=True)
#     Collection_df['Time Stamp'] = pd.to_datetime(Collection_df['Posting Date'])
#     Collection_df.columns = [col if (col == 'Customer' or col== 'Time Stamp') else "C_" + col for col in Collection_df.columns]
    
#     CL_log_df['Customer Code'] = CL_log_df['Customer Code'].astype(str)
#     CL_log_df['Date'] = pd.to_datetime(CL_log_df['Date'])
#     CL_log_df = CL_log_df.sort_values(by=['Customer Code','Date'])
    

#     Sales_stete_df = Sales_df
#     Collection_state_df = Collection_df
#     tota_len = len(Collection_state_df) + len(Sales_stete_df)
    
#     print(f"Total Len: {tota_len}")
#     common_customer = list(set(Sales_stete_df.Customer.unique()).intersection(set(Collection_state_df.Customer.unique())))
#     print("Common Customer: ",len(common_customer))
    
#     customer_sales_not_in_collection = list(set(Sales_stete_df.Customer.unique()) - (set(Collection_state_df.Customer.unique())))
#     print("customer_sales_not_in_collection",len(customer_sales_not_in_collection))
    
#     customer_collection_not_in_sales = list(set(Collection_state_df.Customer.unique()) - (set(Sales_stete_df.Customer.unique())))
#     print("customer_collection_not_in_sales",len(customer_collection_not_in_sales))
    
#     if common_customer:
#         trans_df = pd.concat([pd.concat([Sales_stete_df[Sales_stete_df['Customer'] == customer],
#                                         Collection_state_df[Collection_state_df['Customer']==customer]],
#                                         ignore_index=False).sort_values(by='Time Stamp').reset_index(drop=True) for customer in common_customer]
#                         ,axis=0)
#     else:
#         trans_df = pd.DataFrame()
#     trans_df = pd.concat([trans_df,Sales_stete_df[Sales_stete_df.Customer.isin(customer_sales_not_in_collection)],Collection_state_df[Collection_state_df.Customer.isin(customer_collection_not_in_sales)]],axis=0).reset_index(drop=True)
#     # trans_df = trans_df.reset_index(drop=True)
#     if len(trans_df) == tota_len:
#         # trans_df = pd.concat([trans_df,Sales_stete_df[Sales_stete_df.Customer.isin(customer_sales_not_in_collection)],Collection_state_df[Collection_state_df.Customer.isin(customer_collection_not_in_sales)]],axis=0).reset_index(drop=True)
#         trans_df['Customer'] = trans_df['Customer'].map(str)
#         trans_df['C_Clearing'] = pd.to_datetime(trans_df['C_Clearing'], format='%Y%m%d', errors='coerce')
#         trans_df['Time Stamp'] = pd.to_datetime(trans_df['Time Stamp'])
#         trans_df = trans_df[trans_df['Time Stamp'] >= start_cutoff]
#         trans_df = pd.merge(trans_df,customer_df,on="Customer",how="inner")
#         print("len trans_df : ",len(trans_df))
        
#         columns = trans_df.columns
#         time_stamp_col = ['Customer','Time Stamp']
#         cm_cols = [col for col in columns if col.startswith('CM_')]
#         s_cols = [col for col in columns if col.startswith('S_')]
#         c_cols = [col for col in columns if col.startswith('C_')]
        
#         # Combine the lists
#         sorted_columns = time_stamp_col + cm_cols + s_cols + c_cols
#         trans_df = trans_df[sorted_columns]
        
#         # Extract year, month, week, and quarter
#         trans_df['Year'] = trans_df['Time Stamp'].dt.year
#         trans_df['Date'] = trans_df['Time Stamp'].dt.date
#         trans_df['Month'] = trans_df['Time Stamp'].dt.strftime('%b')
#         trans_df['Week'] = trans_df['Time Stamp'].dt.isocalendar().week
#         trans_df['Quarter'] = trans_df['Time Stamp'].dt.quarter
#         trans_df['Weekday'] = trans_df['Time Stamp'].dt.strftime('%A')
        
#         # Initialize 'Running Outstanding' column
#         trans_df['Running Outstanding'] = 0
#         trans_df['Running Outstanding'] = pd.to_numeric(trans_df['Running Outstanding'])
#         trans_df['Credit Limit'] = 0
#         trans_df['Credit Limit'] = pd.to_numeric(trans_df['Credit Limit'])
#         # Group by customer
#         grouped = trans_df.groupby('Customer')
#         # return grouped

#         # Iterate over each customer group
#         for customer, group in tqdm(grouped):
#             customer_indices = group.index
#             Opening_Balance = group.iloc[0]['CM_Opening Balances'] if not pd.isna(group.iloc[0]['CM_Opening Balances']) else 0
#             # # Determine Opening Balance
#             # if pd.isna(group.iloc[0]['CM_Opening Balances']):
#             #     Opening_Balance = 0
#             # else:
#             #     Opening_Balance = group.iloc[0]['CM_Opening Balances']
#             # Calculate Running Outstanding
#             running_outstanding = [Opening_Balance]
#             # for i in range(len(group)):
#             #     if i == 1:
#             #         if pd.isna(group.iloc[i]['S_Total Sales']): 
#             #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['C_Amount in Balance Transaction Currency'])
#             #         if pd.isna(group.iloc[i]['C_Amount in Balance Transaction Currency']): 
#             #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['S_Total Sales'])
#             #     else:
#             #         if pd.isna(group.iloc[i]['S_Total Sales']):
#             #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['C_Amount in Balance Transaction Currency'])
#             #         if pd.isna(group.iloc[i]['C_Amount in Balance Transaction Currency']):
#             #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['S_Total Sales'])
#             for i in range(len(group)):
#                 if (group.iloc[i]['S_Description'] in ['Cancelled Invoice', 'Cancelled Cred Memo'] or group.iloc[i]['S_Canceled'] == 'X'):
#                     running_outstanding.append(running_outstanding[-1])
#                 else:
#                     sales = group.iloc[i]['S_Total Sales'] if not pd.isna(group.iloc[i].get('S_Total Sales', 0)) else 0
#                     collections = group.iloc[i]['C_Amount in Balance Transaction Currency'] if not pd.isna(group.iloc[i].get('C_Amount in Balance Transaction Currency', 0)) else 0
#                     running_outstanding.append(running_outstanding[-1] + sales + collections)
#                 # print(f"Row {i} - Sales: {sales}, Collections: {collections}, Running Outstanding: {running_outstanding[-1]}")
#             # Assign calculated values back to the DataFrame
#             trans_df.loc[customer_indices, 'Running Outstanding'] = running_outstanding[1::]
            
#             # Credit Limit History Updation
#             cl_df = CL_log_df[CL_log_df['Customer Code'] == customer][['Customer Code','Date','Old value','New value']].copy(deep=True) # type: ignore
#             cl_df = cl_df.reset_index(drop=True)
#             # Iterate through each timestamp
#             for i in range(len(group)):
#                 timestamp = group.loc[group.index[i], 'Time Stamp']
#                 ts_date = timestamp.date()
                
#                 for idx, row in cl_df.iterrows():
#                     row_date = row['Date'].date()            
#                     if idx<len(cl_df)-1:
#                         next_row_date = cl_df.loc[idx + 1, 'Date'].date()
#                     if idx == len(cl_df)-1:
#                         next_row_date = datetime.now().date()
#                     # If timestamp is less than or equal to the first date
#                     if ts_date <= row_date:
#                         group.at[group.index[i], 'Credit Limit'] = row['Old value']
#                         break
                    
#                     # If timestamp is between current date and next date
#                     if ts_date>row_date and ts_date<=next_row_date:
#                         group.at[group.index[i], 'Credit Limit'] = row['New value']                 
#                         break
#                 trans_df.loc[customer_indices,'Credit Limit'] = group.loc[customer_indices,'Credit Limit']
    
#     return trans_df

def trans_df_creation(Sales_df ,Collection_df,customer_df, Billing_type_df=Billing_type_df,CL_log_df=CL_log_df):
    # Sales_df = clean_sales(Sales_df)
    # Collection_df = clean_collections(Collection_df)
    # res_dict = find_usefull_columns(Sales_df)
    # Sales_df = Sales_df[res_dict['usefull_columns']]
    # Billing Type in the Sales_df 
    Sales_df = Sales_df.loc[:, ~Sales_df.columns.duplicated()]
    Collection_df = Collection_df.loc[:, ~Collection_df.columns.duplicated()]
    customer_df = customer_df.loc[:, ~customer_df.columns.duplicated()]
    Billing_type_df = Billing_type_df[['Billing Type', 'Description', 'Change']]
    Sales_df['Billing Description'] = ""
    Sales_df['Billing D/C'] = ''
    Sales_df['Billing Description'] = Sales_df['Billing Type'].map(dict(zip(Billing_type_df['Billing Type'],Billing_type_df['Description'])))
    Sales_df['Billing D/C'] = Sales_df['Billing Type'].map(dict(zip(Billing_type_df['Billing Type'],Billing_type_df['Change'])))
    Sales_df[Sales_df['Billing Description'].isna()][['Billing D/C','Net Value.1']]
    Sales_df['Total Sales']=(Sales_df['Net Value.1'] + Sales_df['Tax Amount.1'])*Sales_df['Billing D/C']
    # customer_df = pd.read_pickle(os.path.join(data_folder,"pkl_data\Customer_df_CM.pkl"))
    Sales_df['ZHIRE-ZNAME'] = Sales_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_ZHIRE-ZNAME'])))
    Sales_df = Sales_df[Sales_df['ZHIRE-ZNAME']!="B2B"]
    Sales_df.drop(columns='ZHIRE-ZNAME',inplace=True)
    Sales_df.reset_index(drop=True,inplace=True)
    Sales_df['Billing Date'] = pd.to_datetime(Sales_df['Billing Date'])
    # Combine 'Created on.1' and 'Time.1'
    Sales_df['Time Stamp'] = pd.to_datetime(Sales_df['Billing Date'].map(str))
    # Sales_df.rename(columns={'Payer':'Customer'},inplace=True)
    Sales_df.columns = [col if (col == 'Customer' or col == 'Time Stamp') else "S_" + col for col in Sales_df.columns]
    # Sales_df['S_State'] = Sales_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_State'])))
    
    # Collection
    # res_dict = find_usefull_columns(Collection_df)
    # Collection_df = Collection_df[res_dict['usefull_columns']]
    # Collection_df['State'] = Collection_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_State'])))
    Collection_df['ZHIRE-ZNAME'] = Collection_df['Customer'].map(dict(zip(customer_df['Customer'],customer_df['CM_ZHIRE-ZNAME'])))
    Collection_df = Collection_df[Collection_df['ZHIRE-ZNAME']!='B2B']
    Collection_df.drop(columns='ZHIRE-ZNAME',inplace=True)
    Collection_df.reset_index(drop=True,inplace=True)
    Collection_df['Time Stamp'] = pd.to_datetime(Collection_df['Posting Date'])
    Collection_df.columns = [col if (col == 'Customer' or col== 'Time Stamp') else "C_" + col for col in Collection_df.columns]
    
    CL_log_df['Customer Code'] = CL_log_df['Customer Code'].astype(str)
    CL_log_df['Date'] = pd.to_datetime(CL_log_df['Date'])
    CL_log_df = CL_log_df.sort_values(by=['Customer Code','Date'])
    
    Sales_stete_df = Sales_df
    Collection_state_df = Collection_df
    tota_len = len(Collection_state_df) + len(Sales_stete_df)
    
    print(f"Total Len: {tota_len}")
    common_customer = list(set(Sales_stete_df.Customer.unique()).intersection(set(Collection_state_df.Customer.unique())))
    print("Common Customer: ",len(common_customer))
    
    customer_sales_not_in_collection = list(set(Sales_stete_df.Customer.unique()) - (set(Collection_state_df.Customer.unique())))
    print("customer_sales_not_in_collection",len(customer_sales_not_in_collection))
    
    customer_collection_not_in_sales = list(set(Collection_state_df.Customer.unique()) - (set(Sales_stete_df.Customer.unique())))
    print("customer_collection_not_in_sales",len(customer_collection_not_in_sales))
    
    if common_customer:
        trans_df = pd.concat([pd.concat([Sales_stete_df[Sales_stete_df['Customer'] == customer],
                                        Collection_state_df[Collection_state_df['Customer']==customer]],
                                        ignore_index=False).sort_values(by='Time Stamp').reset_index(drop=True) for customer in common_customer]
                        ,axis=0)
    else:
        trans_df = pd.DataFrame()
    trans_df = pd.concat([trans_df,Sales_stete_df[Sales_stete_df.Customer.isin(customer_sales_not_in_collection)],Collection_state_df[Collection_state_df.Customer.isin(customer_collection_not_in_sales)]],axis=0).reset_index(drop=True)
    # trans_df = trans_df.reset_index(drop=True)
    if len(trans_df) == tota_len:
        # trans_df = pd.concat([trans_df,Sales_stete_df[Sales_stete_df.Customer.isin(customer_sales_not_in_collection)],Collection_state_df[Collection_state_df.Customer.isin(customer_collection_not_in_sales)]],axis=0).reset_index(drop=True)
        trans_df['Customer'] = trans_df['Customer'].map(str)
        trans_df['C_Clearing'] = pd.to_datetime(trans_df['C_Clearing'], format='%Y%m%d', errors='coerce')
        trans_df['Time Stamp'] = pd.to_datetime(trans_df['Time Stamp'])
        trans_df = trans_df[trans_df['Time Stamp'] >= start_cutoff]
        trans_df = pd.merge(trans_df,customer_df,on="Customer",how="inner")
        print("len trans_df : ",len(trans_df))
        
        columns = trans_df.columns
        time_stamp_col = ['Customer','Time Stamp']
        cm_cols = [col for col in columns if col.startswith('CM_')]
        s_cols = [col for col in columns if col.startswith('S_')]
        c_cols = [col for col in columns if col.startswith('C_')]
        
        # Combine the lists
        sorted_columns = time_stamp_col + cm_cols + s_cols + c_cols
        trans_df = trans_df[sorted_columns]
        
        # Extract year, month, week, and quarter
        trans_df['Year'] = trans_df['Time Stamp'].dt.year
        trans_df['Date'] = trans_df['Time Stamp'].dt.date
        trans_df['Month'] = trans_df['Time Stamp'].dt.strftime('%b')
        trans_df['Week'] = trans_df['Time Stamp'].dt.isocalendar().week
        trans_df['Quarter'] = trans_df['Time Stamp'].dt.quarter
        trans_df['Weekday'] = trans_df['Time Stamp'].dt.strftime('%A')
        
        # Initialize 'Running Outstanding' column
        trans_df['Running Outstanding'] = 0
        trans_df['Running Outstanding'] = pd.to_numeric(trans_df['Running Outstanding'])
        trans_df['Credit Limit'] = 0
        trans_df['Credit Limit'] = pd.to_numeric(trans_df['Credit Limit'])
        # Group by customer
        grouped = trans_df.groupby('Customer')
        # return grouped

        # Iterate over each customer group
        for customer, group in tqdm(grouped):
            customer_indices = group.index
            Opening_Balance = group.iloc[0]['CM_Opening Balances'] if not pd.isna(group.iloc[0]['CM_Opening Balances']) else 0
            # # Determine Opening Balance
            # if pd.isna(group.iloc[0]['CM_Opening Balances']):
            #     Opening_Balance = 0
            # else:
            #     Opening_Balance = group.iloc[0]['CM_Opening Balances']
            # Calculate Running Outstanding
            running_outstanding = [Opening_Balance]
            # for i in range(len(group)):
            #     if i == 1:
            #         if pd.isna(group.iloc[i]['S_Total Sales']): 
            #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['C_Amount in Balance Transaction Currency'])
            #         if pd.isna(group.iloc[i]['C_Amount in Balance Transaction Currency']): 
            #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['S_Total Sales'])
            #     else:
            #         if pd.isna(group.iloc[i]['S_Total Sales']):
            #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['C_Amount in Balance Transaction Currency'])
            #         if pd.isna(group.iloc[i]['C_Amount in Balance Transaction Currency']):
            #             running_outstanding.append(running_outstanding[-1] + group.iloc[i]['S_Total Sales'])
            for i in range(len(group)):
                if (group.iloc[i]['S_Description'] in ['Cancelled Invoice', 'Cancelled Cred Memo'] or group.iloc[i]['S_Canceled'] == 'X'):
                    running_outstanding.append(running_outstanding[-1])
                else:
                    
                    sales = group.iloc[i]['S_Total Sales'] if not pd.isna(group.iloc[i].get('S_Total Sales', 0)) else 0
                    collections = group.iloc[i]['C_Amount in Balance Transaction Currency'] if not pd.isna(group.iloc[i].get('C_Amount in Balance Transaction Currency', 0)) else 0
                    running_outstanding.append(running_outstanding[-1] + sales + collections)
                # print(f"Row {i} - Sales: {sales}, Collections: {collections}, Running Outstanding: {running_outstanding[-1]}")
            # Assign calculated values back to the DataFrame
            trans_df.loc[customer_indices, 'Running Outstanding'] = running_outstanding[1::]
            
            # Credit Limit History Updation
            cl_df = CL_log_df[CL_log_df['Customer Code'] == customer][['Customer Code','Date','Old value','New value']].copy(deep=True) # type: ignore
            cl_df = cl_df.reset_index(drop=True)
            # Iterate through each timestamp
            for i in range(len(group)):
                timestamp = group.loc[group.index[i], 'Time Stamp']
                ts_date = timestamp.date()
                
                for idx, row in cl_df.iterrows():
                    row_date = row['Date'].date()            
                    if idx<len(cl_df)-1:
                        next_row_date = cl_df.loc[idx + 1, 'Date'].date()
                    if idx == len(cl_df)-1:
                        next_row_date = datetime.now().date()
                    # If timestamp is less than or equal to the first date
                    if ts_date <= row_date:
                        group.at[group.index[i], 'Credit Limit'] = row['Old value']
                        break
                    
                    # If timestamp is between current date and next date
                    if ts_date>row_date and ts_date<=next_row_date:
                        group.at[group.index[i], 'Credit Limit'] = row['New value']                 
                        break
                trans_df.loc[customer_indices,'Credit Limit']= group.loc[customer_indices,'Credit Limit']
    
    return trans_df


def knockoff_df_creation(trans_df):
    trans_df = trans_df[trans_df['S_Canceled'] != 'X']
    trans_df = trans_df[~trans_df['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
    PT_df = pd.read_excel(os.path.join(data_folder,'Payments Term.XLSX'),usecols=['Payment terms','No. of Days'])
    # PT_df = pd.read_excel(r'C:\Users\darkhandelwal.ext\OneDrive - Deloitte (O365D)\Documents\Safex_Sync\datasets\Payments_Term.xlsx',usecols=['Payment terms','No. of Days'])
    
    # Define the date string
    # Cut_off_date = "31-03-2025"
    Cut_off_date = "31-03-2024"
    # Define the format of the date string
    date_format = "%d-%m-%Y"
    Cut_off_date = datetime.strptime(Cut_off_date, date_format)
    # print(Cut_off_date)
    print(f"{Cut_off_date}") 

    #Add final knockoff in that 
    MASTER = pd.DataFrame()
    for idx,Customer in tqdm(enumerate(trans_df['Customer'].unique())):
        data = trans_df[trans_df.Customer==Customer].copy(deep = True)
        data = data[['Customer','Date','Month','Year','S_Billing Document','S_Billing Type','C_Document Number', 'C_Document Type', 'CM_Credit Limit','Credit Limit','CM_Opening Balances','S_Payment terms','S_Total Sales','C_Amount in Balance Transaction Currency','Running Outstanding']].copy(deep=True) # type: ignore
        data[['S_Total Sales','C_Amount in Balance Transaction Currency']] = data[['S_Total Sales','C_Amount in Balance Transaction Currency']].fillna(0)
        data[['S_Billing Document','C_Document Number']] = data[['S_Billing Document','C_Document Number']].fillna(0)
        data[['S_Billing Document','C_Document Number']] = data[['S_Billing Document','C_Document Number']].map(int)
        
        data['S_No_of_Days'] = data['S_Payment terms'].map(dict(zip(PT_df['Payment terms'] , PT_df['No. of Days'])))
        data['S_No_of_Days'] = data['S_No_of_Days'].fillna(0)
        data['S_No_of_Days'] = data['S_No_of_Days'].map(int)

        data['S_Billing Type'] = data['S_Billing Type'].fillna('COLLECT')

        data = data.reset_index(drop=True)
        
        OB_df = pd.DataFrame(columns = ['KEY','Customer','Date','Collection ID','Amount','Completion'])
        Sale_df = pd.DataFrame(columns = ['KEY','Customer','Date','Sales ID','Billing Type','Payment Term','Due Date','Due Days','Posting Date','Past Due Date','Collection ID','Sales Amount','Collection Amount','Completion'])
        Collect_df = pd.DataFrame(columns = ['KEY','Customer','Date','Collection ID', 'Document Type', 'Sales ID','OB ID','Amount','Completion','Complete'])
        
        last_index = lambda df: 0 if len(df) == 0 else len(df)
        def Sale_init_with_OB(row,Sale_df=Sale_df):
            idx = last_index(Sale_df)
            Sale_df.loc[idx,'KEY'] = str(uuid.uuid4())
            Sale_df.loc[idx,'Customer'] = row['Customer']
            Sale_df.loc[idx,'Date'] = row['Date']
            Sale_df.loc[idx,'Sales ID'] = "OB_"+str(row['Customer'])
            Sale_df.loc[idx,'Billing Type'] = row['S_Billing Type']
            Sale_df.loc[idx,'Collection ID'] = np.nan
            Sale_df.loc[idx,'Payment Term'] = row['S_Payment terms']
            Sale_df.loc[idx,'Due Date'] = (row['Date'] + timedelta(days=row['S_No_of_Days']))
            Sale_df.loc[idx,'Due Days'] = row['S_No_of_Days']
            if str(row['CM_Opening Balances']) == 'nan':
                Sale_df.loc[idx,'Sales Amount'] = 0
            else:
                Sale_df.loc[idx,'Sales Amount'] = row['CM_Opening Balances']
            Sale_df.loc[idx,'Collection Amount'] = np.nan
            Sale_df.loc[idx,'Completion'] = np.nan
            Sale_df.loc[idx,'Complete'] = False 
            Sale_df.loc[idx,'Posting Date'] = np.nan
            Sale_df.loc[idx,'Past Due Date'] = np.nan
            return True
        
        def Sale_add(row,Sale_df = Sale_df):
            if any(Sale_df['Sales ID'].isin([row['S_Billing Document']])):
                idx = Sale_df[Sale_df['Sales ID'].isin([row['S_Billing Document']])].index[0]
                Sale_df.loc[idx,'Sales Amount'] = Sale_df.loc[idx,'Sales Amount'] + row['S_Total Sales']
                return True
            else:
                idx = last_index(Sale_df)
                Sale_df.loc[idx,'KEY'] = str(uuid.uuid4())
                Sale_df.loc[idx,'Customer'] = row['Customer']
                Sale_df.loc[idx,'Date'] = row['Date']
                Sale_df.loc[idx,'Sales ID'] = row['S_Billing Document']
                Sale_df.loc[idx,'Billing Type'] = row['S_Billing Type']
                Sale_df.loc[idx,'Collection ID'] = np.nan
                Sale_df.loc[idx,'Payment Term'] = row['S_Payment terms']
                Sale_df.loc[idx,'Due Date'] = (row['Date'] + timedelta(days=row['S_No_of_Days']))
                Sale_df.loc[idx,'Due Days'] = row['S_No_of_Days']
                Sale_df.loc[idx,'Sales Amount'] = row['S_Total Sales']
                Sale_df.loc[idx,'Collection Amount'] = np.nan
                Sale_df.loc[idx,'Completion'] = np.nan
                Sale_df.loc[idx,'Complete'] = False 
                Sale_df.loc[idx,'Posting Date'] = np.nan
                Sale_df.loc[idx,'Past Due Date'] = np.nan
            return True
        
        def Collect_add(row,Collect_df = Collect_df):
            Return_sale = "R_"+str(row['S_Billing Document'])
            if 'RETURN' in list(row.keys()) and any(Collect_df['Collection ID'].isin([Return_sale])):
                    idx = Collect_df[Collect_df['Collection ID'].isin([Return_sale])].index[0]
                    Collect_df.loc[idx,'Amount'] = Collect_df.loc[idx,'Amount'] + row['S_Total Sales']
            else:
                row = dict(row)
                idx = last_index(Collect_df)
                Collect_df.loc[idx,'KEY'] = str(uuid.uuid4())
                Collect_df.loc[idx,'Customer'] = row['Customer']
                Collect_df.loc[idx,'Date'] = row['Date']
                Collect_df.loc[idx,'Collection ID'] = row['C_Document Number']
                Collect_df.loc[idx,'Document Type'] = row['C_Document Type']
                Collect_df.loc[idx,'Amount'] = row['C_Amount in Balance Transaction Currency']
                Collect_df.loc[idx,'Complete'] = False
                if 'RETURN' in list(row.keys()):
                    Collect_df.loc[idx,'Collection ID'] = "R_"+str(row['S_Billing Document'])
                    Collect_df.loc[idx,'Amount'] = row['S_Total Sales']
                if 'OB_ID' in list(row.keys()):
                    Collect_df.loc[idx,'OB ID'] = row['OB_ID']
            return True
        
        for idx,row in data.iterrows():
            # opening balance first row creation
            if idx == 0 and row['CM_Opening Balances']>=0:
                Sale_init_with_OB(row)
            
            # Sales Billing Document 
            if row['S_Billing Document'] != 0:
                if row['S_Total Sales'] > 0:
                    Sale_add(row)
                else:
                    row['RETURN'] = True
                    # Sale_add(row)
                    Collect_add(row)
            
            # Collection Document 
            if row['C_Document Number'] != 0:
                Collect_add(row)

        Sale_df_1 = Sale_df.sort_values(['Sales ID','Due Date','Due Days','Sales Amount'],ascending=True)
        Collect_df_1 = Collect_df.copy(deep=True)
        
        i, j = 0,0
        df = []

        while i < len(Sale_df_1) and j < len(Collect_df):
            # if Sale_df_1.loc[i, 'Sales Amount'] < 0:
            #     Collect_df.loc[j, 'Amount'] = abs(Collect_df.loc[j, 'Amount']) + abs(Sale_df_1.loc[i, 'Sales Amount'])
            #     df.append({
            #         'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
            #         'Billing Date': Sale_df_1.loc[i, 'Date'],
            #         'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
            #         'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
            #         'Balance': "Returns_" + str(abs(Sale_df_1.loc[i, 'Sales Amount']))
            #     })
            #     Sale_df_1 = Sale_df_1.drop(index=i).reset_index(drop=True)
            #     continue
            if Sale_df_1.loc[i, 'Sales Amount'] <= abs(Collect_df.loc[j, 'Amount']):
                df.append({
                    'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
                    'Billing Date': Sale_df_1.loc[i, 'Date'],
                    'Payment Term':Sale_df_1.loc[i,'Payment Term'],
                    'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
                    'Due Days':Sale_df_1.loc[i,'Due Days'],
                    'Due Date':Sale_df_1.loc[i,'Due Date'],
                    'Collection_ID': Collect_df.loc[j, 'Collection ID'],
                    'Document Type': Collect_df.loc[j, 'Document Type'],
                    'Posting Date': Collect_df.loc[j, 'Date'],
                    'Past Due Days':(Collect_df.loc[j, 'Date'] - Sale_df_1.loc[i,'Due Date']).days,
                    'Bill Settlement Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i, 'Date']).days,
                    'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
                    'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
                    'Balance': "Collections_" + str(abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount']),
                    'knockoff': 1
                })
                Collect_df.loc[j, 'Amount'] = abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount']
                Sale_df_1.loc[i, 'Sales Amount'] = 0
                i += 1
            else:
                df.append({
                    'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
                    'Billing Date': Sale_df_1.loc[i, 'Date'],
                    'Billing Date': Sale_df_1.loc[i, 'Date'],
                    'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
                    'Payment Term':Sale_df_1.loc[i,'Payment Term'],
                    'Due Days':Sale_df_1.loc[i,'Due Days'],
                    'Due Date':Sale_df_1.loc[i,'Due Date'],
                    'Collection_ID': Collect_df.loc[j, 'Collection ID'],
                    'Document Type': Collect_df.loc[j, 'Document Type'],
                    'Past Due Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i,'Due Date']).days,
                    'Bill Settlement Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i, 'Date']).days,
                    'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
                    'Posting Date': Collect_df.loc[j, 'Date'],
                    'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
                    'Balance': "Sales_" + str(abs(abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount'])),
                    'knockoff': 0
                })
                Sale_df_1.loc[i, 'Sales Amount'] = Sale_df_1.loc[i, 'Sales Amount'] - abs(Collect_df.loc[j, 'Amount'])
                Collect_df.loc[j, 'Amount'] = 0
                j += 1
        
        try:
            if len(df)>0 and df[-1]['Balance'].startswith("Sales"):
                Sale_df_1.loc[i, 'Sales Amount'] += df[-1]['Sales Amount'] - df[-1]['Collections Amount']

            if len(df)>0 and df[-1]['Balance'].startswith("Collections"):
                Collect_df.loc[j, 'Amount'] = df[-1]['Collections Amount'] - df[-1]['Sales Amount']

            if len(df)>0 and df[-1]['Balance'].startswith("Returns"):
                df.append({
                    'Customer': Customer,
                    'Collections Amount': Collect_df.loc[j, 'Amount'],
                    'Collection_ID': Collect_df.loc[j, 'Collection ID'],
                    'Document Type': Collect_df.loc[j, 'Document Type'],
                    'Payment Term': df[-1]['Payment Term'],
                    'Posting Date': Collect_df.loc[j, 'Date'],
                    'Balance': 'Collections_'+str(Collect_df.loc[j, 'Amount'])
                })
            while(i != len(Sale_df_1)):
                try:
                    df.append({
                        'Customer': Customer,
                        'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
                        'Billing Date': Sale_df_1.loc[i, 'Date'],
                        'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
                        'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
                        'Due Date': Sale_df_1.loc[i,'Due Date'],
                        'Balance': 'Sales_' + str(Sale_df_1.loc[i, 'Sales Amount']),
                        'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
                        'knockoff': 0,
                        'Due Days': Sale_df_1.loc[i, 'Due Days'],
                        'Past Due Days': (Cut_off_date.date() - df[-1]['Due Date']).days
                    })
                except:
                    df.append({
                        'Customer': Customer,
                        'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
                        'Billing Date': Sale_df_1.loc[i, 'Date'],
                        'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
                        'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
                        'Due Date': Sale_df_1.loc[i,'Due Date'],
                        'Balance': 'Sales_' + str(Sale_df_1.loc[i, 'Sales Amount']),
                        'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
                        'knockoff': 0,
                        'Due Days': Sale_df_1.loc[i, 'Due Days'],
                        'Past Due Days': (Cut_off_date.date() - Sale_df_1.loc[i,'Due Date']).days
                    })
                i += 1
            while(j != len(Collect_df)):
                df.append({
                    'Customer': Customer,
                    'Collection_ID': Collect_df.loc[j, 'Collection ID'],
                    'Document Type': Collect_df.loc[j, 'Document Type'],
                    'Posting Date': Collect_df.loc[j, 'Date'],
                    'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
                    'Balance': 'Collections_' + str(abs(Collect_df.loc[j, 'Amount'])),
                    'knockoff': 0
                })
                j += 1
                
        except Exception as Error:
            print(f'{Customer} :{Error}')
            break
        
        final = pd.DataFrame(df)
        final['Customer'] = Customer
        MASTER = pd.concat([MASTER, final], ignore_index=True)
    return MASTER

# def knockoff_df_creation(trans_df):
#     trans_df = trans_df[trans_df['S_Canceled'] != 'X']
#     trans_df = trans_df[~trans_df['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
#     PT_df = pd.read_excel(os.path.join(data_folder,'Payments Term.XLSX'),usecols=['Payment terms','No. of Days'])
#     # PT_df = pd.read_excel(r'C:\Users\darkhandelwal.ext\OneDrive - Deloitte (O365D)\Documents\Safex_Sync\datasets\Payments_Term.xlsx',usecols=['Payment terms','No. of Days'])
#     # Define the date string
#     Cut_off_date = "31-03-2024"
#     # Define the format of the date string
#     date_format = "%d-%m-%Y"
#     Cut_off_date = datetime.strptime(Cut_off_date, date_format)
#     # print(Cut_off_date)
#     print(f"{Cut_off_date}") 
#     #Add final knockoff in that 
#     MASTER = pd.DataFrame()
#     for idx,Customer in tqdm(enumerate(trans_df['Customer'].unique())):
#         data = trans_df[trans_df.Customer==Customer].copy(deep = True)
#         data = data[['Customer','Date','Month','Year','S_Billing Document','S_Billing Type','C_Document Number', 'C_Document Type', 'CM_Credit Limit','Credit Limit','CM_Opening Balances','S_Payment terms','S_Total Sales','C_Amount in Balance Transaction Currency','Running Outstanding']].copy(deep=True) # type: ignore
#         data[['S_Total Sales','C_Amount in Balance Transaction Currency']] = data[['S_Total Sales','C_Amount in Balance Transaction Currency']].fillna(0)
#         data[['S_Billing Document','C_Document Number']] = data[['S_Billing Document','C_Document Number']].fillna(0)
#         data[['S_Billing Document','C_Document Number']] = data[['S_Billing Document','C_Document Number']].map(int)
#         data['S_No_of_Days'] = data['S_Payment terms'].map(dict(zip(PT_df['Payment terms'] , PT_df['No. of Days'])))
#         data['S_No_of_Days'] = data['S_No_of_Days'].fillna(0)
#         data['S_No_of_Days'] = data['S_No_of_Days'].map(int)

#         # data['S_Billing Type'] = data['S_Billing Type'].fillna('COLLECT')

#         data = data.reset_index(drop=True)
        
#         OB_df = pd.DataFrame(columns = ['KEY','Customer','Date','Collection ID','Amount','Completion'])
#         Sale_df = pd.DataFrame(columns = ['KEY','Customer','Date','Sales ID','Billing Type','Payment Term','Due Date','Due Days','Posting Date','Past Due Date','Collection ID','Sales Amount','Collection Amount','Completion'])
#         Collect_df = pd.DataFrame(columns = ['KEY','Customer','Date','Collection ID', 'Document Type', 'Sales ID','OB ID','Amount','Completion','Complete'])
        
#         last_index = lambda df: 0 if len(df) == 0 else len(df)
#         def Sale_init_with_OB(row,Sale_df=Sale_df):
#             idx = last_index(Sale_df)
#             Sale_df.loc[idx,'KEY'] = str(uuid.uuid4())
#             Sale_df.loc[idx,'Customer'] = row['Customer']
#             Sale_df.loc[idx,'Date'] = row['Date']
#             Sale_df.loc[idx,'Sales ID'] = "OB_"+str(row['Customer'])
#             Sale_df.loc[idx,'Billing Type'] = row['S_Billing Type']
#             Sale_df.loc[idx,'Collection ID'] = np.nan
#             Sale_df.loc[idx,'Payment Term'] = row['S_Payment terms'] 
#             Sale_df.loc[idx,'Due Date'] = (row['Date'] + timedelta(days=row['S_No_of_Days']))
#             Sale_df.loc[idx,'Due Days'] = row['S_No_of_Days']
#             if str(row['CM_Opening Balances']) == 'nan':
#                 Sale_df.loc[idx,'Sales Amount'] = 0
#             else:
#                 Sale_df.loc[idx,'Sales Amount'] = row['CM_Opening Balances']
#             Sale_df.loc[idx,'Collection Amount'] = np.nan
#             Sale_df.loc[idx,'Completion'] = np.nan
#             Sale_df.loc[idx,'Complete'] = False 
#             Sale_df.loc[idx,'Posting Date'] = np.nan
#             Sale_df.loc[idx,'Past Due Date'] = np.nan
#             return True
        
#         def Sale_add(row,Sale_df = Sale_df):
#             if any(Sale_df['Sales ID'].isin([row['S_Billing Document']])):
#                 idx = Sale_df[Sale_df['Sales ID'].isin([row['S_Billing Document']])].index[0]
#                 Sale_df.loc[idx,'Sales Amount'] = Sale_df.loc[idx,'Sales Amount'] + row['S_Total Sales']
#                 return True
#             else:
#                 idx = last_index(Sale_df)
#                 Sale_df.loc[idx,'KEY'] = str(uuid.uuid4())
#                 Sale_df.loc[idx,'Customer'] = row['Customer']
#                 Sale_df.loc[idx,'Date'] = row['Date']
#                 Sale_df.loc[idx,'Sales ID'] = row['S_Billing Document']
#                 Sale_df.loc[idx,'Billing Type'] = row['S_Billing Type']
#                 Sale_df.loc[idx,'Collection ID'] = np.nan
#                 Sale_df.loc[idx,'Payment Term'] = row['S_Payment terms']
#                 Sale_df.loc[idx,'Due Date'] = (row['Date'] + timedelta(days=row['S_No_of_Days']))
#                 Sale_df.loc[idx,'Due Days'] = row['S_No_of_Days']
#                 Sale_df.loc[idx,'Sales Amount'] = row['S_Total Sales']
#                 Sale_df.loc[idx,'Collection Amount'] = np.nan
#                 Sale_df.loc[idx,'Completion'] = np.nan
#                 Sale_df.loc[idx,'Complete'] = False 
#                 Sale_df.loc[idx,'Posting Date'] = np.nan
#                 Sale_df.loc[idx,'Past Due Date'] = np.nan
#             return True
        
#         def Collect_add(row,Collect_df = Collect_df):
#             Return_sale = "R_"+str(row['S_Billing Document'])
#             if 'RETURN' in list(row.keys()) and any(Collect_df['Collection ID'].isin([Return_sale])):
#                     idx = Collect_df[Collect_df['Collection ID'].isin([Return_sale])].index[0]
#                     Collect_df.loc[idx,'Amount'] = Collect_df.loc[idx,'Amount'] + row['S_Total Sales']
#             else:
#                 row = dict(row)
#                 idx = last_index(Collect_df)
#                 Collect_df.loc[idx,'KEY'] = str(uuid.uuid4())
#                 Collect_df.loc[idx,'Customer'] = row['Customer']
#                 Collect_df.loc[idx,'Date'] = row['Date']
#                 Collect_df.loc[idx,'Collection ID'] = row['C_Document Number']
#                 Collect_df.loc[idx,'Document Type'] = row['C_Document Type']
#                 Collect_df.loc[idx,'Amount'] = row['C_Amount in Balance Transaction Currency']
#                 Collect_df.loc[idx,'Complete'] = False
#                 if 'RETURN' in list(row.keys()):
#                     Collect_df.loc[idx,'Collection ID'] = "R_"+str(row['S_Billing Document'])
#                     Collect_df.loc[idx,'Amount'] = row['S_Total Sales']
#                 if 'OB_ID' in list(row.keys()):
#                     Collect_df.loc[idx,'OB ID'] = row['OB_ID']
#             return True
        
#         for idx,row in data.iterrows():
#             # opening balance first row creation
#             if idx == 0 and row['CM_Opening Balances']>=0:
#                 Sale_init_with_OB(row)
            
#             # Sales Billing Document 
#             if row['S_Billing Document'] != 0:
#                 if row['S_Total Sales'] > 0:
#                     Sale_add(row)
#                 else:
#                     row['RETURN'] = True
#                     # Sale_add(row)
#                     Collect_add(row)
            
#             # Collection Document 
#             if row['C_Document Number'] != 0:
#                 Collect_add(row)

#         Sale_df_1 = Sale_df.sort_values(['Sales ID','Due Date','Due Days','Sales Amount'],ascending=True)
#         Collect_df_1 = Collect_df.copy(deep=True)
        
#         i, j = 0,0
#         df = []

#         while i < len(Sale_df_1) and j < len(Collect_df):
#             # if Sale_df_1.loc[i, 'Sales Amount'] < 0:
#             #     Collect_df.loc[j, 'Amount'] = abs(Collect_df.loc[j, 'Amount']) + abs(Sale_df_1.loc[i, 'Sales Amount'])
#             #     df.append({
#             #         'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
#             #         'Billing Date': Sale_df_1.loc[i, 'Date'],
#             #         'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
#             #         'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
#             #         'Balance': "Returns_" + str(abs(Sale_df_1.loc[i, 'Sales Amount']))
#             #     })
#             #     Sale_df_1 = Sale_df_1.drop(index=i).reset_index(drop=True)
#             #     continue
#             if Sale_df_1.loc[i, 'Sales Amount'] <= abs(Collect_df.loc[j, 'Amount']):
#                 df.append({
#                     'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
#                     'Billing Date': Sale_df_1.loc[i, 'Date'],
#                     'Payment Term':Sale_df_1.loc[i,'Payment Term'],
#                     'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
#                     'Due Days':Sale_df_1.loc[i,'Due Days'],
#                     'Due Date':Sale_df_1.loc[i,'Due Date'],
#                     'Collection_ID': Collect_df.loc[j, 'Collection ID'],
#                     'Document Type': Collect_df.loc[j, 'Document Type'],
#                     'Posting Date': Collect_df.loc[j, 'Date'],
#                     'Past Due Days':(Collect_df.loc[j, 'Date'] - Sale_df_1.loc[i,'Due Date']).days,
#                     'Bill Settlement Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i, 'Date']).days,
#                     'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
#                     'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
#                     'Balance': "Collections_" + str(abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount']),
#                     'knockoff': 1
#                 })
#                 Collect_df.loc[j, 'Amount'] = abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount']
#                 Sale_df_1.loc[i, 'Sales Amount'] = 0
#                 i += 1
#             else:
#                 df.append({
#                     'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
#                     'Billing Date': Sale_df_1.loc[i, 'Date'],
#                     'Billing Date': Sale_df_1.loc[i, 'Date'],
#                     'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
#                     'Payment Term':Sale_df_1.loc[i,'Payment Term'],
#                     'Due Days':Sale_df_1.loc[i,'Due Days'],
#                     'Due Date':Sale_df_1.loc[i,'Due Date'],
#                     'Collection_ID': Collect_df.loc[j, 'Collection ID'],
#                     'Document Type': Collect_df.loc[j, 'Document Type'],
#                     'Past Due Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i,'Due Date']).days,
#                     'Bill Settlement Days':(Collect_df.loc[j, 'Date']- Sale_df_1.loc[i, 'Date']).days,
#                     'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
#                     'Posting Date': Collect_df.loc[j, 'Date'],
#                     'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
#                     'Balance': "Sales_" + str(abs(abs(Collect_df.loc[j, 'Amount']) - Sale_df_1.loc[i, 'Sales Amount'])),
#                     'knockoff': 0
#                 })
#                 Sale_df_1.loc[i, 'Sales Amount'] = Sale_df_1.loc[i, 'Sales Amount'] - abs(Collect_df.loc[j, 'Amount'])
#                 Collect_df.loc[j, 'Amount'] = 0
#                 j += 1
        
#         try:
#             if len(df)>0 and df[-1]['Balance'].startswith("Sales"):
#                 Sale_df_1.loc[i, 'Sales Amount'] += df[-1]['Sales Amount'] - df[-1]['Collections Amount']

#             if len(df)>0 and df[-1]['Balance'].startswith("Collections"):
#                 Collect_df.loc[j, 'Amount'] = df[-1]['Collections Amount'] - df[-1]['Sales Amount']

#             if len(df)>0 and df[-1]['Balance'].startswith("Returns"):
#                 df.append({
#                     'Customer': Customer,
#                     'Collections Amount': Collect_df.loc[j, 'Amount'],
#                     'Collection_ID': Collect_df.loc[j, 'Collection ID'],
#                     'Document Type': Collect_df.loc[j, 'Document Type'],
#                     'Payment Term': df[-1]['Payment Term'],
#                     'Posting Date': Collect_df.loc[j, 'Date'],
#                     'Balance': 'Collections_'+str(Collect_df.loc[j, 'Amount'])
#                 })
#             while(i != len(Sale_df_1)):
#                 try:
#                     df.append({
#                         'Customer': Customer,
#                         'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
#                         'Billing Date': Sale_df_1.loc[i, 'Date'],
#                         'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
#                         'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
#                         'Due Date': Sale_df_1.loc[i,'Due Date'],
#                         'Balance': 'Sales_' + str(Sale_df_1.loc[i, 'Sales Amount']),
#                         'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
#                         'knockoff': 0,
#                         'Due Days': Sale_df_1.loc[i, 'Due Days'],
#                         'Past Due Days': (Cut_off_date.date() - df[-1]['Due Date']).days
#                     })
#                 except:
#                     df.append({
#                         'Customer': Customer,
#                         'Sale_ID': Sale_df_1.loc[i, 'Sales ID'],
#                         'Billing Date': Sale_df_1.loc[i, 'Date'],
#                         'Billing Type': Sale_df_1.loc[i, 'Billing Type'],
#                         'Sales Amount': Sale_df_1.loc[i, 'Sales Amount'],
#                         'Due Date': Sale_df_1.loc[i,'Due Date'],
#                         'Balance': 'Sales_' + str(Sale_df_1.loc[i, 'Sales Amount']),
#                         'Payment Term': Sale_df_1.loc[i, 'Payment Term'],
#                         'knockoff': 0,
#                         'Due Days': Sale_df_1.loc[i, 'Due Days'],
#                         'Past Due Days': (Cut_off_date.date() - Sale_df_1.loc[i,'Due Date']).days
#                     })
#                 i += 1
#             while(j != len(Collect_df)):
#                 df.append({
#                     'Customer': Customer,
#                     'Collection_ID': Collect_df.loc[j, 'Collection ID'],
#                     'Document Type': Collect_df.loc[j, 'Document Type'],
#                     'Posting Date': Collect_df.loc[j, 'Date'],
#                     'Collections Amount': abs(Collect_df.loc[j, 'Amount']),
#                     'Balance': 'Collections_' + str(abs(Collect_df.loc[j, 'Amount'])),
#                     'knockoff': 0
#                 })
#                 j += 1
                
#         except Exception as Error:
#             print(f'{Customer} :{Error}')
#             break
        
#         final = pd.DataFrame(df)
#         final['Customer'] = Customer
#         MASTER = pd.concat([MASTER, final], ignore_index=True)
#     return MASTER

def generate_bins(max_days, interval=30):
    bins = []
    start = 0
    while start < max_days:
        end = start + interval
        if end > max_days:
            end = max_days
        if start ==0 :
            bins.append((start, end))
        else:
            bins.append((start+1, end))
        start += interval
    return bins


def convert_bins_to_strings(bins,interval):
    bin_strings = []
    for idx,(start, end) in enumerate(bins,start=1):
        if end-start < interval-2:
            bin_strings.append(f'{end}+ Days')
        else:
            bin_strings.append(f'{start}-{end} Days')
    return bin_strings

def parse_bin_list(bin_list):
    bins = []
    for bin_str in bin_list:
        if '+' in bin_str:
            start = int(bin_str.split('+')[0])
            bins.append((start, float('inf')))
        else:
            start, end = map(int, bin_str.split(' Days')[0].split('-'))
            bins.append((start, end))
    return bins

def get_day_bin(days, bins):
    for start, end in bins:
        if start <= days <= end:
            return f'{start}-{int(end)} Days' if end != float('inf') else f'{start}+ Days'

def aging_df_creation(Knock_df):
    Knock_df.rename(columns = {'Posting Date': 'Clearing Date'}, inplace = True)
    # Parameters
    max_days = 365
    interval = 30
    Ageing_col_list = [
        'Customer', 'Billing Document', 'Clearing Document', 'Payment Term', 'Due Days', 
        'Due Date', 'Billing Date', 'Clearing Date', 'No Due', 'Total Over Due'
    ]
    
    # Generate bins and column labels
    bins = generate_bins(max_days, interval)
    column_list = convert_bins_to_strings(bins, interval)
    Ageing_col_list.extend(column_list)
    
    # Initialize ageing DataFrame
    ageing_df = pd.DataFrame(columns=Ageing_col_list)
    
    # Parse bins
    parsed_bins = parse_bin_list(column_list)
    
    # Fill missing collection amounts with 0
    Knock_df['Collections Amount'] = Knock_df['Collections Amount'].fillna(0)
    
    # Store rows as a list of dictionaries
    rows = []
    
    # Iterate through unique customers
    for customer in tqdm(Knock_df['Customer'].unique()):
        customer_data = Knock_df[Knock_df['Customer'] == customer]
        
        # Process each row for the customer
        for _, row in customer_data.iterrows():
            ageing_data = {
                'Customer': row['Customer'],
                'Billing Document': row['Sale_ID'],
                'Clearing Document': row['Collection_ID'],
                'Payment Term': row['Payment Term'],
                'Due Days': row['Due Days'],
                'Due Date': row['Due Date'],
                'Billing Date': row['Billing Date'],
                'Clearing Date': row['Clearing Date'],
                'No Due': 0,
                'Total Over Due': 0
            }
            
            past_due_days = row['Past Due Days']
            if past_due_days < 0:
                ageing_data['No Due'] = max(0, row['Sales Amount'] - row['Collections Amount'])
            else:
                bin_label = get_day_bin(past_due_days, parsed_bins)
                ageing_data[bin_label] = max(0, row['Sales Amount'] - row['Collections Amount'])
            
            ageing_data['Total Over Due'] = sum(ageing_data.get(col, 0) for col in column_list)
            
            # Add the row to the list
            rows.append(ageing_data)
    
    # Concatenate rows into a DataFrame
    ageing_df = pd.concat([ageing_df, pd.DataFrame(rows)], ignore_index=True)
    ageing_df.rename(columns = {'Clearing Date': 'Posting Date'}, inplace = True)
    return ageing_df

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

def get_season_of_crop(month, state):
    # Mapping month abbreviation to month number
    month_mapping = {
        'Jan': 1, 'Feb': 2, 'Mar': 3, 'Apr': 4, 'May': 5, 'Jun': 6,
        'Jul': 7, 'Aug': 8, 'Sep': 9, 'Oct': 10, 'Nov': 11, 'Dec': 12
    }
    
    # Convert the month abbreviation to a number
    month_num = month_mapping.get(month)
    
    # Define seasons for Punjab
    if state == 'Punjab':
        if month_num in [3, 4, 5]:  # March, April, May
            return 'Rabi'
        elif month_num in [6, 7, 8, 9]:  # June, July, August, September
            return 'Kharif'
        elif month_num in [10, 11,12,1,2]:  # October, November
            return 'Rabi'
        else:
            return 'Off-season'
    
    # Define seasons for Rajasthan
    elif state == 'Rajasthan':
        if month_num in [2, 3,4, 5, 6]:  # April, May, June
            return 'Rabi'
        elif month_num in [7, 8, 9, 10]:  # July, August, September, October
            return 'Kharif'
        elif month_num in [11, 12, 1]:  # November, December, January
            return 'Rabi'
        else:
            return 'Off-season'
    
    # Define seasons for Uttar Pradesh
    elif state == 'Uttar Pradesh':
        if month_num in [3, 4, 5, 6]:  # March, April, May, June
            return 'Rabi'
        elif month_num in [7, 8, 9, 10]:  # July, August, September, October
            return 'Kharif'
        elif month_num in [11, 12, 1, 2]:  # November, December, January, February
            return 'Rabi'
        else:
            return 'Off-season'
    
    # Define seasons for Haryana
    elif state == 'Haryana':
        if month_num in [3, 4, 5, 6]:  # March to June
            return 'Rabi'
        elif month_num in [7, 8, 9, 10]:  # July to October
            return 'Kharif'
        elif month_num in [11, 12, 1, 2]:  # November to February
            return 'Rabi'
        else:
            return 'Off-season'
    
    # Define seasons for Uttarakhand
    elif state == 'Uttarakhand':
        if month_num in [3, 4, 5]:  # March to May
            return 'Rabi'
        elif month_num in [6, 7, 8, 9]:  # June to September
            return 'Kharif'
        elif month_num in [10, 11, 12, 1, 2]:  # October to February
            return 'Rabi'
        else:
            return 'Off-season'
    
    # If state is not one of the specified states
    else:
        return 'Off-season'

def profitability(Trans_df):
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
    
    Trans_df = Trans_df[Trans_df['S_Canceled'] != 'X']
    Trans_df = Trans_df[~Trans_df['S_Description'].isin(['Cancelled Invoice', 'Cancelled Cred Memo'])]
    Trans_df = Trans_df[Trans_df['CM_ZHIRE-ZNAME'] != "B2B"]
    # print(Trans_df[Trans_df['S_Item Description'].str.startswith("Rate", na = False)]['S_Billing Type'].value_counts()/len(Trans_df[Trans_df['S_Item Description'].str.startswith("Rate", na = False)]))
    #  Addin Financial Year
    Trans_df['Fiscal Year']= Trans_df['S_Billing Date'].apply(get_fiscal_year)
    Trans_df['Fiscal Year_I'] = Trans_df['S_Billing Date'].apply(get_financial_year)
    Trans_df['Material_P_Key'] = Trans_df['Fiscal Year'].astype(str) + " + " + Trans_df['S_Material'].astype(str)
    
    # Adding Costing Sheet
    # Loading and processing the Costing DataFrame
    costing_df = pd.read_excel(os.path.join(data_folder,'Costing Sheet.XLSX'))
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
    
    Rate_Diff_Sales = Trans_df[Trans_df['S_Billing Type'].isin(['ZICR', 'ZIDR', 'ZWDR', 'ZWCR'])][['Customer', 'S_Billing Type', 'Fiscal Year', 'S_Net Value.1']]
    expected_billing_types = ['ZICR', 'ZIDR', 'ZWDR', 'ZWCR']
    rate_diff_pivot_table = Rate_Diff_Sales.pivot_table(
        values='S_Net Value.1', 
        index=['Customer', 'Fiscal Year'], 
        columns='S_Billing Type', 
        aggfunc='sum', 
        fill_value=0
    )
    for col in expected_billing_types:
        if col not in rate_diff_pivot_table.columns:
            rate_diff_pivot_table[col] = 0
    rate_diff_pivot_table = rate_diff_pivot_table[expected_billing_types]
    
    rate_diff_pivot_table['Rate Diff'] = (
        rate_diff_pivot_table.get('ZICR', pd.Series(0, index=rate_diff_pivot_table.index)) +
        rate_diff_pivot_table.get('ZIDR', pd.Series(0, index=rate_diff_pivot_table.index)) +
        rate_diff_pivot_table.get('ZWCR', pd.Series(0, index=rate_diff_pivot_table.index)) + 
        rate_diff_pivot_table.get('ZWDR', pd.Series(0, index=rate_diff_pivot_table.index))
    )
    rate_diff_pivot_table = rate_diff_pivot_table.reset_index()
    # Rate_Diff_Sales = Trans_df[(Trans_df['S_Item Description'].str.startswith("Rate", na = False)) | (Trans_df['S_Billing Type'] == 'ZICR')][['Customer', 'Fiscal Year', 'S_Net Value.1']]
    
    pivot_df = pivot_df.reset_index()
    pivot_df.rename(columns={
        'ZIRE': 'Sale_ZIRE',
        'ZIDR': 'Sale_ZIDR',
        'ZWCR': 'Sale_ZWCR',
        'ZWDR': 'Sale_ZWDR',
        'ZINV': 'Sale_ZINV',
        'ZICR': 'Sale_ZICR'
    }, inplace = True)
    Result_df = pivot_df.merge(rate_diff_pivot_table ,how='left',on=['Customer','Fiscal Year'])
    # Result_df['Fiscal Year'] = Result_df['Fiscal Year'].astype(str)
    Result_df = Result_df.merge(cogs_pivot_df,how='left',on=['Customer','Fiscal Year'])
    # Result_df['Net Sale Amount After RD'] = Result_df['Net Sale Amount'] - Result_df['Rate Diff']
    # Result_df['Gross Margin'] =  Result_df['Net Sale Amount'] -  Result_df['COGS']    # Use a lambda function to handle the division and replace inf with 0
    
    # Result_df['Gross Margin%'] = Result_df.apply(
    #     lambda row: row['Gross Margin'] / abs(row['Net Sale Amount']) if row['Net Sale Amount'] != 0 else 1,
    #     axis=1
    # )
    Result_df_list.append(Result_df)
    Final_result_df = pd.concat(Result_df_list,axis=0)
    #using Financial year as 'Fiscal Year'
    Final_result_df['Net Revenue'] = Final_result_df['Net Sale Amount'] - Final_result_df['Rate Diff']
    if 'COGS' not in Final_result_df.columns:
        Final_result_df['COGS'] = 0  # Default to zero or appropriate value
    Final_result_df['Gross Margin'] = Final_result_df['Net Revenue'] - Final_result_df['COGS']
    Final_result_df['Gross Margin%'] = Final_result_df.apply(
        lambda row: (row['Gross Margin'] / abs(row['Net Revenue']) * 100) if row['Net Revenue'] != 0 else 1,
        axis=1
    )
    return Final_result_df



# optimize this function:
def standardization_CL(Trans_df,knock_off_data,profitability_df):
    
    month_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
    }
    
    Multiplier = 1.10

    MIN_CRITERIA = 80
    MAX_CRITERIA = 90

    Analytical_df_list = []

    customer_standardised_CL_df = pd.DataFrame(columns=['Customer','State','Created Date','First Activity','Last Activity','ASM Code','Tenure','No. of Days For average calculation','Avg Utilization %','Avg Daily Sales','Avg Yearly Sales', 'last 12 months max sales', 'last 12 months min sales', 'Average Yearly collections','last 12 months max collection','last 12 months max credit note','Most Frequent Payment Term used','Customer Profitability','Top 5 Materials purchased','Previous Credit Limit','Current Credit Limit','Max utilization %','Average Bill Settlement Days','Standardised Credit Limit','Condition Flag'])
    quarterly_utilization = []
    monthly_utilization = []
    cutoff_date = pd.to_datetime('31-03-2024')
    start_cutoff = pd.to_datetime('01-04-2022', format='%d-%m-%Y')
    
    # for state in list(Trans_df['CM_State'].unique()):
    # print(state)
    knock_off_data['Customer'] = knock_off_data['Customer'].map(str)
    Trans_df = Trans_df[Trans_df['CM_ZHIRE-ZNAME'] != "B2B"]
    today = cutoff_date
    Trans_df = Trans_df[Trans_df['Time Stamp'] >= start_cutoff]
    Trans_df = Trans_df[Trans_df['Time Stamp'] <= cutoff_date]
    Trans_df['CM_Start Date on'] = pd.to_datetime(Trans_df[ 'CM_Start Date on'])
    Trans_df['CM_Start Date on'].fillna(today,inplace=True)
    Trans_df['First Activity'] = ''
    Trans_df['First Activity'] = Trans_df['Customer'].map(Trans_df.groupby('Customer')['Time Stamp'].min().to_dict())
    Trans_df['Last Activity'] = ''
    Trans_df['Last Activity'] = Trans_df['Customer'].map(Trans_df.groupby('Customer')['Time Stamp'].max().to_dict())
    
    Trans_df['Sale First Activity'] = ''
    Trans_df['Sale First Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].notna()].groupby('Customer')['Time Stamp'].min().to_dict())
    Trans_df['Sale Last Activity'] = ''
    Trans_df['Sale Last Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].notna()].groupby('Customer')['Time Stamp'].max().to_dict())
    
    Trans_df['Collect First Activity'] = ''
    Trans_df['Collect First Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].isna()].groupby('Customer')['Time Stamp'].min().to_dict())
    Trans_df['Collect Last Activity'] = ''
    Trans_df['Collect Last Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].isna()].groupby('Customer')['Time Stamp'].max().to_dict())
    
    # Season of Crop 
    Trans_df['Season_of_crop'] = Trans_df[['Month','CM_State']].apply(lambda row: get_season_of_crop(row['Month'], row['CM_State']), axis=1)
    
    Trans_df['No. of Days For average calculation'] = Trans_df['Last Activity'] - Trans_df['First Activity']
    Trans_df['No. of Days For average calculation'] = Trans_df['No. of Days For average calculation'].apply(lambda x: str(x).split(" days")[0].strip())
    Trans_df['No. of Days For average calculation'] = Trans_df['No. of Days For average calculation'].map(int)
    # Trans_df['First Activity'] = Trans_df[['CM_Start Date on','First Activity']].apply(lambda row: min(row['CM_Start Date on'], row['First Activity']), axis=1)
    Trans_df['Tenure'] = ''
    Trans_df['Tenure'] = Trans_df['Last Activity'] - Trans_df['CM_Start Date on']
    Trans_df['Tenure_days'] = 0
    Trans_df['Tenure_days'] = Trans_df['Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
    Trans_df['Tenure_days'] = Trans_df['Tenure_days'].map(int)
    
    Trans_df['Sale Tenure'] = ''
    Trans_df['Sale First Activity'] = Trans_df['Sale First Activity'].fillna(Trans_df['CM_Start Date on'])
    Trans_df['Sale Last Activity'] = Trans_df['Sale Last Activity'].fillna(Trans_df['Last Activity'])
    Trans_df['Sale Tenure'] = Trans_df['Sale Last Activity'] - Trans_df['Sale First Activity']
    Trans_df['Sale Tenure_days'] = 0
    Trans_df['Sale Tenure_days'] = Trans_df['Sale Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
    Trans_df['Sale Tenure_days'] = Trans_df['Sale Tenure_days'].map(int)
    
    Trans_df['Collect Tenure'] = ''
    Trans_df['Collect First Activity'] = Trans_df['Collect First Activity'].fillna(Trans_df['CM_Start Date on'])
    Trans_df['Collect Last Activity'] = Trans_df['Collect Last Activity'].fillna(Trans_df['Last Activity'])
    Trans_df['Collect Last Activity'] = Trans_df['Collect Last Activity'].fillna(Trans_df['CM_Start Date on'])
    Trans_df['Collect Tenure'] = Trans_df['Collect Last Activity'] - Trans_df['Collect First Activity']
    Trans_df['Collect Tenure_days'] = 0
    Trans_df['Collect Tenure_days'] = Trans_df['Collect Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
    Trans_df['Collect Tenure_days'] = Trans_df['Collect Tenure_days'].map(int)
    Trans_df['Credit Limit'] = pd.to_numeric(Trans_df['Credit Limit'])
    Trans_df['Running Outstanding'] = pd.to_numeric(Trans_df['Running Outstanding'])
    
    for customer, group_df in tqdm(Trans_df.groupby('Customer')):
        group_df['S_Material'] = group_df['S_Material'].map(str)
        group_df['Utilization %'] =  group_df[['Running Outstanding','Credit Limit']].apply(lambda row: utilization_calculate(row['Running Outstanding'],row['Credit Limit']), axis=1)
        indx = len(customer_standardised_CL_df)+1
        customer_standardised_CL_df.loc[indx,'Customer'] = customer
        customer_standardised_CL_df.loc[indx,'State'] = group_df['CM_State'].values[0]
        customer_standardised_CL_df.loc[indx,'Created Date'] = pd.to_datetime(group_df['CM_Start Date on'].values[0]).date()
        customer_standardised_CL_df.loc[indx,'ASM Code'] = group_df['CM_ASM Code'].values[0]
        customer_standardised_CL_df.loc[indx,'First Activity'] = pd.to_datetime(group_df['First Activity'].values[0]).date()
        customer_standardised_CL_df.loc[indx,'Last Activity'] = pd.to_datetime(group_df['Last Activity'].values[0]).date()
        customer_standardised_CL_df.loc[indx,'Tenure'] = group_df['Tenure_days'].values[-1]
        customer_standardised_CL_df.loc[indx,'Avg Utilization %'] = group_df['Utilization %'].mean()
        customer_standardised_CL_df.loc[indx,'Avg Daily Sales'] = (group_df['S_Total Sales'].sum())/group_df['Sale Tenure_days'].values[0] if group_df['Sale Tenure_days'].values[0] != 0 else 0
        if group_df['Sale Tenure_days'].values[0]<365:
            customer_standardised_CL_df.loc[indx,'Avg Yearly Sales'] = group_df['S_Total Sales'].sum()
        else:
            customer_standardised_CL_df.loc[indx,'Avg Yearly Sales'] = (group_df['S_Total Sales'].sum())/(group_df['Sale Tenure_days'].values[0]/365)
        customer_standardised_CL_df.loc[indx,'last 12 months max sales'] = group_df[group_df['S_Billing Type']=='ZINV'].groupby(['Year','Month'])['S_Total Sales'].max().tail(12).max()
        customer_standardised_CL_df.loc[indx,'last 12 months min sales'] = group_df[group_df['S_Billing Type']=='ZINV'].groupby(['Year','Month'])['S_Total Sales'].min().tail(12).min()

        if group_df['Collect Tenure_days'].values[0]<365:
            customer_standardised_CL_df.loc[indx,'Average Yearly collections'] = abs(group_df['C_Amount in Balance Transaction Currency'].sum())
        else:
            customer_standardised_CL_df.loc[indx,'Average Yearly collections'] = abs(group_df['C_Amount in Balance Transaction Currency'].sum())/(group_df['Collect Tenure_days'].values[0]/365)
        customer_standardised_CL_df.loc[indx,'last 12 months max collection'] = group_df[['C_Amount in Balance Transaction Currency','Year','Month']].groupby(['Year','Month'])['C_Amount in Balance Transaction Currency'].max().tail(12).max()
        customer_standardised_CL_df.loc[indx,'last 12 months max credit note'] = group_df[group_df['S_Billing Type'].isin(['ZCCL','ZCSI','ZICL','ZICR','ZIRE','ZSEZ','ZWCR'])].groupby(['Year','Month'])['S_Total Sales'].min().tail(12).min()
        if (group_df['S_Payment terms'].isna()).all():
            customer_standardised_CL_df.loc[indx,'Most Frequent Payment Term used'] = ''
        else:
            customer_standardised_CL_df.loc[indx,'Most Frequent Payment Term used'] =group_df['S_Payment terms'].value_counts().index[0]
        customer_standardised_CL_df.loc[indx,'Customer Profitability'] = profitability_df[profitability_df['Customer']==customer ][['Fiscal Year','Gross Margin%']].groupby('Fiscal Year')['Gross Margin%'].sum().mean()
        customer_standardised_CL_df.loc[indx,'Top 5 Materials purchased'] = ", ".join([mat for mat in group_df.groupby('S_Material')['S_Total Sales'].max().sort_values(ascending=False).index if mat.startswith('F-')][0:5])
        
        
        customer_standardised_CL_df.loc[indx,'Previous Credit Limit'] = group_df['Credit Limit'].values[-1]
        customer_standardised_CL_df.loc[indx,'Current Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
        customer_standardised_CL_df.loc[indx,'No. of Days For average calculation'] = group_df['No. of Days For average calculation'].values[0]
        
        
        Days_State_season = {
                                'Rajasthan': {'Kharif': 123, 'Rabi': 242},
                                'Punjab': {'Kharif': 122, 'Rabi': 243},
                                'Uttar Pradesh': {'Kharif': 123, 'Rabi': 242},
                                'Haryana': {'Kharif': 120, 'Rabi': 245},
                                'Uttarakhand': {'Kharif': 124, 'Rabi': 241}
                            }
        
        # customer_standardised_CL_df.loc[indx,'Average sales during Kharif season'] = group_df[group_df['Season_of_crop'] == 'Kharif'][['Year','S_Total Sales']].groupby(['Year'])['S_Total Sales'].sum().mean()
        # customer_standardised_CL_df.loc[indx,'Average sales during Rabi season'] = group_df[group_df['Season_of_crop'] == 'Rabi'][['Year','S_Total Sales']].groupby(['Year'])['S_Total Sales'].sum().mean()
        group_df['C_Amount in Balance Transaction Currency'] = group_df['C_Amount in Balance Transaction Currency'].abs()
        # customer_standardised_CL_df.loc[indx,'Previous Credit Limit'] = group_df['Credit Limit'].values[-1]
        customer_standardised_CL_df.loc[indx,'Current Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
        customer_standardised_CL_df.loc[indx,'Average Bill Settlement Days'] = knock_off_data[(knock_off_data['Customer']==customer) & (knock_off_data['knockoff']==1)]['Bill Settlement Days'].sum()/len(knock_off_data[knock_off_data['Customer']==customer]['Sale_ID'].unique()) # Average billing settlement days per billing document
        # Quarterly
        g_df = group_df.groupby(['Year','Quarter'])['Utilization %'].max().reset_index()
        if len(g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']]) == 4:
            ug_df = g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']]
        else:
            req_quarter = 4 - len(g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']])
            ug_df = pd.concat([
                g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']],
                g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year'] - 1]
                .tail(req_quarter)
                .sort_values('Quarter', ascending=False)
            ], axis=0)
        ug_df['Customer'] = customer

        # Monthly
        month_g_df = group_df.groupby(['Year','Month'])['Utilization %'].max().reset_index()
        
        month_g_df['Month'] = month_g_df['Month'].map(month_dict)
        
        month_g_df = month_g_df.sort_values(by=['Year','Month'])
        
        # month_g_df = month_g_df.iloc[-1]  # Access the last row directly
        # Get the last year value
        last_year = month_g_df.iloc[-1]['Year']

        if len(month_g_df[month_g_df['Year'] == last_year]) == 12:
            month_ug_df = month_g_df[month_g_df['Year'] == last_year]
        else:
            # Get the required number of months from the previous year
            req_month = 12 - len(month_g_df[month_g_df['Year'] == last_year])
            month_ug_df = pd.concat([month_g_df[month_g_df['Year'] == last_year], 
                                    month_g_df[month_g_df['Year'] == last_year - 1].tail(req_month)], axis=0)
        month_ug_df['Customer'] = customer
        # Filtering out 2023-2024

        # /calculating min and max RO
        month_convert  = lambda x: {v: k for k, v in month_dict.items()}[x]
        MAX_RO = max([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].max() for YR,MNTH in zip(month_ug_df['Year'],month_ug_df['Month'])])
        MIN_RO = min([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].min() for YR,MNTH in zip(month_ug_df['Year'],month_ug_df['Month'])])
        customer_standardised_CL_df.loc[indx, 'MAX RO'] = MAX_RO
        # ug_df['Utilization %'] = pd.to_numeric(ug_df['Utilization %'])
        # # Less than 80%
        # if ug_df['Utilization %'].max() < MIN_CRITERIA:
        #         # Psuedocode:
        #             # If current running utilization <= 20%
        #             # Then
        #             #         1. Calculate x as the max Running outstanding actual value for last one year (from last activity)
        #             #         2. Calculate y as the minimum Running outstanding actual value for last one year (from last activity)
        #             #         3. Check If y <0
        #             #         if yes then
        #             #             Check if Current running outstanding < 0
        #             #            if yes then 
        #             #                   CL1 = (x - y)/2
        #             #                    Curr_Out = Absolute value of current outstanding
        #             #                    Credit_Adj = x/2
        #             #                    CL2 = Curr_Out + Credit_Adj
        #             #                    Standardized CL = min (CL1, CL2)
        #             #            Else
        #             #                    Standardized CL = 1.1 * x
        #             #     Else
        #             #             Standardized CL = 1.1 * x
        #     if ug_df['Utilization %'].max() <= 20:
                    
        #             if MIN_RO<0:
        #                 if group_df['Running Outstanding'].values[-1]<0:
        #                     CL1 = (MAX_RO - MIN_RO)/2
        #                     Curr_Out = abs(group_df['Running Outstanding'].values[-1])
        #                     Credit_Adj = MAX_RO/2
        #                     CL2 = Curr_Out + Credit_Adj
        #                     customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = min(CL1,CL2)
        #                     customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_1_1%'
        #                     customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #                 else:
        #                     customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * MAX_RO
        #                     customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_1_2%'
        #                     customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #             else:
        #                 customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * MAX_RO
        #                 customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_2%'
        #                 customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #     else:
        #             # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * (ug_df['Utilization %'].max()/100) * group_df['CM_Credit Limit'].values[-1]
        #             customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO)
        #             customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #             # customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{MIN_CRITERIA}%'
        #             customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>20% & <{MIN_CRITERIA}%'
        # # greater than 80% less than 90%
        # elif ug_df['Utilization %'].max() >= MIN_CRITERIA and ug_df['Utilization %'].max()<MAX_CRITERIA:
        #     # group_df[f'Above {MIN_CRITERIA} quarterly'] = group_df['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
        #     group_df_last_year = group_df[group_df['Year'] >= last_year-1]
        #     group_df_last_year[f'Above 80 quarterly'] = group_df_last_year['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
        #     group_df_last_year['>80 & <90 flag'] = group_df_last_year['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
        #     # ug_df[f'Above {MIN_CRITERIA} quarterly'] = ug_df['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
        #     # month_ug_df[f'Above {MIN_CRITERIA} quarterly'] = month_ug_df['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
        #     # if all(ug_df[f'Above {MIN_CRITERIA} quarterly'].head(1)) or month_ug_df[f'Above {MIN_CRITERIA} quarterly'].sum()>=3:
        #     if (group_df_last_year[f'>80 & <90 flag'].sum()>=3) or all((group_df_last_year['Above 80 quarterly'].head(1))):
        #         filtered_df = group_df_last_year[(group_df_last_year['>80 & <90 flag']) | (group_df_last_year['Above 80 quarterly'])]
        #         same_cl = filtered_df['Credit Limit'].max()
        #         # customer_standardised_CL_df.loc[indx, 'Check Date'] = group_df_last_year[group_df_last_year[(group_df_last_year['Above 80 quarterly']) | (group_df_last_year['>80 & <90 flag'].sum()>=3)] & (group_df_last_year['Credit Limit'] == same_cl)]['Date']
        #         customer_standardised_CL_df.loc[indx, 'Check Date'] = group_df_last_year.loc[
        #             group_df_last_year['Credit Limit'] == same_cl, 'Date'
        #         ].iloc[0] if not group_df_last_year[
        #             (group_df_last_year['Above 80 quarterly']) |
        #             (group_df_last_year['>80 & <90 flag'].sum() >= 3)
        #         ].empty else None

        #         customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = same_cl
        #         customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #         customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>=80% & <{MAX_CRITERIA}% (3X in 1 yr/1X in 1 qtr) [same CL]'
        #     else:
        #         customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO)
        #         # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * (ug_df['Utilization %'].max())/100) * group_df['CM_Credit Limit'].values[-1]
        #         customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #         customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>=80% & <{MAX_CRITERIA}%'
        # # Greater than 90%
        # elif ug_df['Utilization %'].max() >=MAX_CRITERIA:
        #     ug_df[f'Above {MAX_CRITERIA} quarterly'] = ug_df['Utilization %'].apply(lambda x: x>=MAX_CRITERIA)
        #     month_ug_df[f'Above {MAX_CRITERIA} quarterly'] = month_ug_df['Utilization %'].apply(lambda x: x>=MAX_CRITERIA)
        #     # if all(ug_df[f'Above {MAX_CRITERIA} quarterly'].head(1)) or month_ug_df[f'Above {MAX_CRITERIA} quarterly'].sum()>=3:
        #         # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * (ug_df['Utilization %'].max())/100) * group_df['CM_Credit Limit'].values[-1]
        #     customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO)
        #     customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #     customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>={MAX_CRITERIA}%'
        #     # else:
        #     #     customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
        #     #     customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
        #     #     customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>={MAX_CRITERIA}% [same CL]'
                
        customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = MAX_RO
        # 'Standardised Credit Limit'
        if any(group_df['CM_ASM Code'].isin(['WOP', 'BLOCKED', 'LEGAL', 'DORMANT', 'B2B', 'VACANT_GUN', 'VAC_GHAZIP', 'VACNT_GOLA'])):    # update this
            customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = 0
        
        quarterly_utilization.append(ug_df)
        monthly_utilization.append(month_ug_df)
    # quarterly_utilization_df = pd.concat(quarterly_utilization,axis = 0)
    # monthly_utilization_df = pd.concat(monthly_utilization,axis = 0)
    lookback_date = cutoff_date - timedelta(days=int(1.5 * 365))  # Approximation for 1.5 years
    # Get customers with activity on or after the lookback date
    active_customers = Trans_df.loc[
        Trans_df['Time Stamp'] >= lookback_date, 'Customer'
    ].unique()
    customer_standardised_CL_df['Impact of Standardisation'] = np.where(
        customer_standardised_CL_df['Current Credit Limit'] < customer_standardised_CL_df['Standardised Credit Limit'],
        'Increased',
        np.where(
            customer_standardised_CL_df['Current Credit Limit'] == customer_standardised_CL_df['Standardised Credit Limit'],
            'No Change',
            'Decreased'
        )
    )
    customer_standardised_CL_df['NoActivityFlag'] = ~customer_standardised_CL_df['Customer'].isin(active_customers)
    # profit_df = pd.read_pickle(r'C:\Users\darkhandelwal\Desktop\AgCare-Git\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Rajasthan.pkl')
    profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_combined.pkl')
    customers_to_be_deleted = list(set(customer_standardised_CL_df['Customer']) - set(profit_df['Customer']))

    # Add the Deletion Flag
    customer_standardised_CL_df['Deletion Flag'] = customer_standardised_CL_df['Customer'].apply(
        lambda customer: 'to be deleted' if customer in customers_to_be_deleted else ''
    )
    return customer_standardised_CL_df

# trans_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_Andhra Pradesh.pkl')
# profit_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Andhra Pradesh.pkl')
# knock_df = pd.read_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_Andhra Pradesh.pkl')
# profit_df = profit_df[profit_df['Customer'] == '11005483']
# knock_df = knock_df[knock_df['Customer'] == '11005483']
# trans_df = trans_df[trans_df['Customer'] == '11005483']

# Apply the standardization function and collect the result
# standardized_df = standardization_CL(trans_df, knock_df, profit_df)
# standardized_df.to_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\cl_std_v2\cl_std_Andhra Pradesh')
# states = ['Meghalaya']
# sales_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\Sales_Statewise_Clean_Pickle\sales_Meghalaya.pkl')
# collections_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\Collections_Statewise_Clean_Pickle\collections_Meghalaya.pkl')
# trans_df = trans_df_creation(sales_df, collections_df, Billing_type_df, CL_log_df)
# knock_df = knockoff_df_creation(trans_df)
# profit_df = profitability(trans_df)
# cl_standardization_df = standardization_CL(trans_df, knock_df, profit_df)
# trans_df.to_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_Meghalaya.pkl')
# knock_df.to_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_Meghalaya.pkl')
# profit_df.to_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Meghalaya.pkl')
# cl_standardization_df.to_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\Collections_Statewise_Clean_Pickle\collections_Meghalaya.pkl')

# states = ['Meghalaya']
# df_list = []
# for state in tqdm(states):
#     # Replace file paths with placeholders or dynamically load based on state
#     trans_df = pd.read_pickle(
#         rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_{state}.pkl'
#     )
#     profit_df = pd.read_pickle(
#         rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_{state}.pkl'
#     )
#     knock_df = pd.read_pickle(
#         rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_{state}.pkl'
#     )
#     # Apply the standardization function and collect the result
#     standardized_df = standardization_CL(trans_df, knock_df, profit_df)
#     standardized_df.to_pickle(rf'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\cl_std_v2\cl_std_{state}.pkl')


    # df_list.append(standardized_df)



# trans_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\transactions_clean_statewise\transactions_Rajasthan.pkl')
# profit_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\profit_df_statewise\profit_df_Rajasthan.pkl')
# knock_df = pd.read_pickle(r'C:\Users\Deloite\Desktop\Agcare-GIT\Data\PKL_Data_FINAL_v4\knock_df_statewise\knock_df_Rajasthan.pkl')
# df = standardization_CL(trans_df, knock_df, profit_df)
# df.to_clipboard(index = False)


# CONSIDERING 2 YEARS
def standardization_CL_2_YEARS(Trans_df,knock_off_data,profitability_df):
    month_dict = {
    'Jan': 1,
    'Feb': 2,
    'Mar': 3,
    'Apr': 4,
    'May': 5,
    'Jun': 6,
    'Jul': 7,
    'Aug': 8,
    'Sep': 9,
    'Oct': 10,
    'Nov': 11,
    'Dec': 12
    }
    
    Multiplier = 1.10

    MIN_CRITERIA = 80
    MAX_CRITERIA = 90

    Analytical_df_list = []

    customer_standardised_CL_df = pd.DataFrame(columns=['Customer','State','Created Date','First Activity','Last Activity','ASM Code','Tenure','No. of Days For average calculation','Avg Utilization %','Avg Daily Sales','Avg Yearly Sales', 'last 12 months max sales', 'last 12 months min sales', 'Average Yearly collections','last 12 months max collection','last 12 months max credit note','Most Frequent Payment Term used','Customer Profitability','Top 5 Materials purchased','Previous Credit Limit','Current Credit Limit','Max utilization %','Average Bill Settlement Days','Standardised Credit Limit','Condition Flag'])
    quarterly_utilization = []
    monthly_utilization = []
    cutoff_date = pd.to_datetime('31-03-2024')
    for state in list(Trans_df['CM_State'].unique()):
        print(state)
        knock_off_data['Customer'] = knock_off_data['Customer'].map(str)
        Trans_df = Trans_df[Trans_df['CM_ZHIRE-ZNAME'] != "B2B"]
        today = datetime.now()
        Trans_df['CM_Start Date on'] = pd.to_datetime(Trans_df[ 'CM_Start Date on'])
        Trans_df['CM_Start Date on'].fillna(today,inplace=True)
        Trans_df['First Activity'] = ''
        Trans_df['First Activity'] = Trans_df['Customer'].map(Trans_df.groupby('Customer')['Time Stamp'].min().to_dict())
        Trans_df['Last Activity'] = ''
        Trans_df['Last Activity'] = Trans_df['Customer'].map(Trans_df.groupby('Customer')['Time Stamp'].max().to_dict())
        
        Trans_df['Sale First Activity'] = ''
        Trans_df['Sale First Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].notna()].groupby('Customer')['Time Stamp'].min().to_dict())
        Trans_df['Sale Last Activity'] = ''
        Trans_df['Sale Last Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].notna()].groupby('Customer')['Time Stamp'].max().to_dict())
        
        Trans_df['Collect First Activity'] = ''
        Trans_df['Collect First Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].isna()].groupby('Customer')['Time Stamp'].min().to_dict())
        Trans_df['Collect Last Activity'] = ''
        Trans_df['Collect Last Activity'] = Trans_df['Customer'].map(Trans_df[Trans_df['S_Total Sales'].isna()].groupby('Customer')['Time Stamp'].max().to_dict())
        
        # Season of Crop
        Trans_df['Season_of_crop'] = Trans_df[['Month','CM_State']].apply(lambda row: get_season_of_crop(row['Month'], row['CM_State']), axis=1)
        Trans_df['No. of Days For average calculation'] = Trans_df['Last Activity'] - Trans_df['First Activity']
        Trans_df['No. of Days For average calculation'] = Trans_df['No. of Days For average calculation'].apply(lambda x: str(x).split(" days")[0].strip())
        Trans_df['No. of Days For average calculation'] = Trans_df['No. of Days For average calculation'].map(int)
        # Trans_df['First Activity'] = Trans_df[['CM_Start Date on','First Activity']].apply(lambda row: min(row['CM_Start Date on'], row['First Activity']), axis=1)
        Trans_df['Tenure'] = ''
        Trans_df['Tenure'] = Trans_df['Last Activity'] - Trans_df['CM_Start Date on']
        Trans_df['Tenure_days'] = 0
        Trans_df['Tenure_days'] = Trans_df['Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
        Trans_df['Tenure_days'] = Trans_df['Tenure_days'].map(int)
        
        Trans_df['Sale Tenure'] = ''
        Trans_df['Sale First Activity'] = Trans_df['Sale First Activity'].fillna(Trans_df['CM_Start Date on'])
        Trans_df['Sale Last Activity'] = Trans_df['Sale Last Activity'].fillna(Trans_df['Last Activity'])
        Trans_df['Sale Tenure'] = Trans_df['Sale Last Activity'] - Trans_df['Sale First Activity']
        Trans_df['Sale Tenure_days'] = 0
        Trans_df['Sale Tenure_days'] = Trans_df['Sale Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
        Trans_df['Sale Tenure_days'] = Trans_df['Sale Tenure_days'].map(int)
        
        Trans_df['Collect Tenure'] = ''
        Trans_df['Collect First Activity'] = Trans_df['Collect First Activity'].fillna(Trans_df['CM_Start Date on'])
        Trans_df['Collect Last Activity'] = Trans_df['Collect Last Activity'].fillna(Trans_df['Last Activity'])
        Trans_df['Collect Last Activity'] = Trans_df['Collect Last Activity'].fillna(Trans_df['CM_Start Date on'])
        Trans_df['Collect Tenure'] = Trans_df['Collect Last Activity'] - Trans_df['Collect First Activity']
        Trans_df['Collect Tenure_days'] = 0
        Trans_df['Collect Tenure_days'] = Trans_df['Collect Tenure'].apply(lambda x: str(x).split(" days")[0].strip())
        Trans_df['Collect Tenure_days'] = Trans_df['Collect Tenure_days'].map(int)
        Trans_df['Credit Limit'] = pd.to_numeric(Trans_df['Credit Limit'])
        Trans_df['Running Outstanding'] = pd.to_numeric(Trans_df['Running Outstanding'])
        
        for customer, group_df in tqdm(Trans_df.groupby('Customer')):
            group_df['S_Material'] = group_df['S_Material'].map(str)
            group_df['Utilization %'] =  group_df[['Running Outstanding','Credit Limit']].apply(lambda row: utilization_calculate(row['Running Outstanding'],row['Credit Limit']), axis=1)
            indx = len(customer_standardised_CL_df)+1
            customer_standardised_CL_df.loc[indx,'Customer'] = customer
            customer_standardised_CL_df.loc[indx,'State'] = state
            customer_standardised_CL_df.loc[indx,'Created Date'] = pd.to_datetime(group_df['CM_Start Date on'].values[0]).date()
            customer_standardised_CL_df.loc[indx,'ASM Code'] = group_df['CM_ASM Code'].values[0]
            customer_standardised_CL_df.loc[indx,'First Activity'] = pd.to_datetime(group_df['First Activity'].values[0]).date()
            customer_standardised_CL_df.loc[indx,'Last Activity'] = pd.to_datetime(group_df['Last Activity'].values[0]).date()
            customer_standardised_CL_df.loc[indx,'Tenure'] = group_df['Tenure_days'].values[-1]
            customer_standardised_CL_df.loc[indx,'Avg Utilization %'] = group_df['Utilization %'].mean()
            customer_standardised_CL_df.loc[indx,'Avg Daily Sales'] = (group_df['S_Total Sales'].sum())/group_df['Sale Tenure_days'].values[0] if group_df['Sale Tenure_days'].values[0] != 0 else 0
            if group_df['Sale Tenure_days'].values[0]<365:
                customer_standardised_CL_df.loc[indx,'Avg Yearly Sales'] = group_df['S_Total Sales'].sum()
            else:
                customer_standardised_CL_df.loc[indx,'Avg Yearly Sales'] = (group_df['S_Total Sales'].sum())/(group_df['Sale Tenure_days'].values[0]/365)
            customer_standardised_CL_df.loc[indx,'last 12 months max sales'] = group_df[group_df['S_Billing Type']=='ZINV'].groupby(['Year','Month'])['S_Total Sales'].max().tail(12).max()
            customer_standardised_CL_df.loc[indx,'last 12 months min sales'] = group_df[group_df['S_Billing Type']=='ZINV'].groupby(['Year','Month'])['S_Total Sales'].min().tail(12).min()

            if group_df['Collect Tenure_days'].values[0]<365:
                customer_standardised_CL_df.loc[indx,'Average Yearly collections'] = abs(group_df['C_Amount in Balance Transaction Currency'].sum())
            else:
                customer_standardised_CL_df.loc[indx,'Average Yearly collections'] = abs(group_df['C_Amount in Balance Transaction Currency'].sum())/(group_df['Collect Tenure_days'].values[0]/365)
            customer_standardised_CL_df.loc[indx,'last 12 months max collection'] = group_df[['C_Amount in Balance Transaction Currency','Year','Month']].groupby(['Year','Month'])['C_Amount in Balance Transaction Currency'].max().tail(12).max()
            customer_standardised_CL_df.loc[indx,'last 12 months max credit note'] = group_df[group_df['S_Billing Type'].isin(['ZCCL','ZCSI','ZICL','ZICR','ZIRE','ZSEZ','ZWCR'])].groupby(['Year','Month'])['S_Total Sales'].min().tail(12).min()
            if (group_df['S_Payment terms'].isna()).all():
                customer_standardised_CL_df.loc[indx,'Most Frequent Payment Term used'] = ''
            else:
                customer_standardised_CL_df.loc[indx,'Most Frequent Payment Term used'] =group_df['S_Payment terms'].value_counts().index[0]
            customer_standardised_CL_df.loc[indx,'Customer Profitability'] = profitability_df[profitability_df['Customer']==customer ][['Fiscal Year','Gross Margin%']].groupby('Fiscal Year')['Gross Margin%'].sum().mean()
            customer_standardised_CL_df.loc[indx,'Top 5 Materials purchased'] = ", ".join([mat for mat in group_df.groupby('S_Material')['S_Total Sales'].max().sort_values(ascending=False).index if mat.startswith('F-')][0:5])

            customer_standardised_CL_df.loc[indx,'Previous Credit Limit'] = group_df['Credit Limit'].values[-1]
            customer_standardised_CL_df.loc[indx,'Current Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
            customer_standardised_CL_df.loc[indx,'No. of Days For average calculation'] = group_df['No. of Days For average calculation'].values[0]

            Days_State_season = {
                                    'Rajasthan': {'Kharif': 123, 'Rabi': 242},
                                    'Punjab': {'Kharif': 122, 'Rabi': 243},
                                    'Uttar Pradesh': {'Kharif': 123, 'Rabi': 242},
                                    'Haryana': {'Kharif': 120, 'Rabi': 245},
                                    'Uttarakhand': {'Kharif': 124, 'Rabi': 241}
                                }
            
            # customer_standardised_CL_df.loc[indx,'Average sales during Kharif season'] = group_df[group_df['Season_of_crop'] == 'Kharif'][['Year','S_Total Sales']].groupby(['Year'])['S_Total Sales'].sum().mean()
            # customer_standardised_CL_df.loc[indx,'Average sales during Rabi season'] = group_df[group_df['Season_of_crop'] == 'Rabi'][['Year','S_Total Sales']].groupby(['Year'])['S_Total Sales'].sum().mean()
            group_df['C_Amount in Balance Transaction Currency'] = group_df['C_Amount in Balance Transaction Currency'].abs()
            customer_standardised_CL_df.loc[indx,'Previous Credit Limit'] = group_df['Credit Limit'].values[-1]
            customer_standardised_CL_df.loc[indx,'Current Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
            customer_standardised_CL_df.loc[indx,'Average Bill Settlement Days'] = knock_off_data[(knock_off_data['Customer']==customer) & (knock_off_data['knockoff']==1)]['Bill Settlement Days'].sum()/len(knock_off_data[knock_off_data['Customer']==customer]['Sale_ID'].unique()) # Average billing settlement days per billing document
            # Quarterly
            
            g_df = group_df.groupby(['Year','Quarter'])['Utilization %'].max().reset_index()

            # getting the last 2 years
            date_two_years_ago = group_df['Last Activity'].max() - pd.DateOffset(years=2)
            start_year = date_two_years_ago.year
            start_quarter = date_two_years_ago.quarter
            start_month = date_two_years_ago.month
        
            g_df_2_years = g_df[(g_df['Year'] > start_year) | ((g_df['Year'] == start_year) & (g_df['Quarter'] >= start_quarter))]
            g_df_2_years = g_df_2_years.reset_index()

            if len(g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']]) == 4:
                ug_df = g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']]
            else:
                req_quarter = 4 - len(g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']])
                ug_df = pd.concat([
                    g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year']],
                    g_df[g_df.Year == g_df.at[len(g_df) - 1, 'Year'] - 1]
                    .tail(req_quarter)
                    .sort_values('Quarter', ascending=False)
                ], axis=0)
            ug_df['Customer'] = customer

            # Monthly
            month_g_df = group_df.groupby(['Year','Month'])['Utilization %'].max().reset_index()
            month_g_df['Month'] = month_g_df['Month'].map(month_dict)
            
            month_g_df = month_g_df.sort_values(by=['Year','Month'])
            month_g_df_2_years = month_g_df[(month_g_df['Year'] > start_year) | ((month_g_df['Year'] == start_year) & (month_g_df['Month'] >= start_month))]
            # month_g_df = month_g_df.iloc[-1]  # Access the last row directly
            # Get the last year value
            last_year = month_g_df.iloc[-1]['Year']
            if len(month_g_df[month_g_df['Year'] == last_year]) == 12:
                month_ug_df = month_g_df[month_g_df['Year'] == last_year]
            else:
                # Get the required number of months from the previous year
                req_month = 12 - len(month_g_df[month_g_df['Year'] == last_year])
                month_ug_df = pd.concat([month_g_df[month_g_df['Year'] == last_year], 
                                        month_g_df[month_g_df['Year'] == last_year - 1].tail(req_month)], axis=0)
            month_ug_df['Customer'] = customer
            # Filtering out 2023-2024

            # /calculating min and max RO
            month_convert  = lambda x: {v: k for k, v in month_dict.items()}[x]
            MAX_RO_last_2_years = max([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].max() for YR,MNTH in zip(month_g_df_2_years['Year'],month_g_df_2_years['Month'])])
            MIN_RO_last_2_years = min([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].min() for YR,MNTH in zip(month_g_df_2_years['Year'],month_g_df_2_years['Month'])])
            ug_df['Utilization %'] = pd.to_numeric(ug_df['Utilization %'])
            # Less than 80%
            if ug_df['Utilization %'].max() < MIN_CRITERIA:
                    # Psuedocode:
                        # If current running utilization <= 20%
                        # Then
                        #         1. Calculate x as the max Running outstanding actual value for last one year (from last activity)
                        #         2. Calculate y as the minimum Running outstanding actual value for last one year (from last activity)
                        #         3. Check If y <0
                        #         if yes then
                        #             Check if Current running outstanding < 0
                        #            if yes then 
                        #                   CL1 = (x - y)/2
                        #                    Curr_Out = Absolute value of current outstanding
                        #                    Credit_Adj = x/2
                        #                    CL2 = Curr_Out + Credit_Adj
                        #                    Standardized CL = min (CL1, CL2)
                        #            Else
                        #                    Standardized CL = 1.1 * x
                        #     Else
                        #             Standardized CL = 1.1 * x
                if ug_df['Utilization %'].max() <= 20:
                        MAX_RO = max([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].max() for YR,MNTH in zip(month_ug_df['Year'],month_ug_df['Month'])])
                        MIN_RO = min([group_df[(group_df['Year'] == YR) & (group_df['Month']==month_convert(MNTH))]['Running Outstanding'].min() for YR,MNTH in zip(month_ug_df['Year'],month_ug_df['Month'])])
                        if MIN_RO<0:
                            if group_df['Running Outstanding'].values[-1]<0:
                                CL1 = (MAX_RO - MIN_RO)/2
                                Curr_Out = abs(group_df['Running Outstanding'].values[-1])
                                Credit_Adj = MAX_RO/2
                                CL2 = Curr_Out + Credit_Adj
                                customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = min(CL1,CL2)
                                customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_1_1%'
                                customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                            else:
                                customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * MAX_RO
                                customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_1_2%'
                                customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                        else:
                            customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * MAX_RO
                            customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'<{20}_2%'
                            customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()            
                else:
                        # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = Multiplier * (ug_df['Utilization %'].max()/100) * group_df['CM_Credit Limit'].values[-1]
                        customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO_last_2_years)
                        customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                        customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>20% & <{MIN_CRITERIA}%'

            # greater than 80% less than 90%
            elif ug_df['Utilization %'].max() >= MIN_CRITERIA and ug_df['Utilization %'].max()<MAX_CRITERIA:
                ug_df[f'Above {MIN_CRITERIA} quarterly'] = ug_df['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
                month_ug_df[f'Above {MIN_CRITERIA} quarterly'] = month_ug_df['Utilization %'].apply(lambda x: (x>=MIN_CRITERIA) & (x < MAX_CRITERIA))
                # 3 times in the last 1 year or 1 time in the recent quarter
                if all(ug_df[f'Above {MIN_CRITERIA} quarterly'].head(1)) or month_ug_df[f'Above {MIN_CRITERIA} quarterly'].sum()>=3:
                    customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
                    customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                    customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>=80% & <{MAX_CRITERIA}% (3X in 1 yr/1X in 1 qtr) [same CL]'
                else:
                    customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO_last_2_years)
                    # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * (ug_df['Utilization %'].max())/100) * group_df['CM_Credit Limit'].values[-1]
                    customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                    customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>=80% & <{MAX_CRITERIA}%'
                # customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>80% & <{MAX_CRITERIA}%'
            
            # Greater than 100%
            elif ug_df['Utilization %'].max() >=MAX_CRITERIA:
                ug_df[f'Above {MAX_CRITERIA} quarterly'] = ug_df['Utilization %'].apply(lambda x: x>=MAX_CRITERIA)
                month_ug_df[f'Above {MAX_CRITERIA} quarterly'] = month_ug_df['Utilization %'].apply(lambda x: x>=MAX_CRITERIA)
                if all(ug_df[f'Above {MAX_CRITERIA} quarterly'].head(1)) or month_ug_df[f'Above {MAX_CRITERIA} quarterly'].sum()>=3:
                    # customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * (ug_df['Utilization %'].max())/100) * group_df['CM_Credit Limit'].values[-1]
                    customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = (Multiplier * MAX_RO_last_2_years)
                    customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                    customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>={MAX_CRITERIA}% (3X in 1 yr/1X in 1 qtr)'
                else:
                    customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = group_df['CM_Credit Limit'].values[-1]
                    customer_standardised_CL_df.loc[indx,'Max utilization %'] = ug_df['Utilization %'].max()
                    customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>={MAX_CRITERIA}% [same CL]'
                # customer_standardised_CL_df.loc[indx,'Condition Flag'] = f'>={MAX_CRITERIA}%'
            
            # 'Standardised Credit Limit'
            if any(group_df['CM_ASM Code'].isin(['WOP', 'BLOCKED', 'LEGAL', 'DORMANT', 'B2B', 'VACANT_GUN', 'VAC_GHAZIP', 'VACNT_GOLA'])):    # update this
                customer_standardised_CL_df.loc[indx,'Standardised Credit Limit'] = 0
            
            quarterly_utilization.append(ug_df)
            monthly_utilization.append(month_ug_df)
    
    quarterly_utilization_df = pd.concat(quarterly_utilization,axis = 0)
    monthly_utilization_df = pd.concat(monthly_utilization,axis = 0)

    lookback_date = cutoff_date - timedelta(days=int(1.5 * 365))  # Approximation for 1.5 years
    # Get customers with activity on or after the lookback date
    active_customers = Trans_df.loc[
        Trans_df['Time Stamp'] >= lookback_date, 'Customer'
    ].unique()
    customer_standardised_CL_df['Impact of Standardisation'] = np.where(
        customer_standardised_CL_df['Current Credit Limit'] > customer_standardised_CL_df['Standardised Credit Limit'],
        'Increased',
        np.where(
            customer_standardised_CL_df['Current Credit Limit'] == customer_standardised_CL_df['Standardised Credit Limit'],
            'No Change',
            'Decreased'
        )
    )
    # customer_standardised_CL_df['NoActivityFlag'] = ~customer_standardised_CL_df['Customer'].isin(active_customers)
    # customers_to_be_deleted = list(set(customer_standardised_CL_df['Customer']) - set(profit_df['Customer']))

    # Add the Deletion Flag
    # customer_standardised_CL_df['Deletion Flag'] = customer_standardised_CL_df['Customer'].apply(
        # lambda customer: 'to be deleted' if customer in customers_to_be_deleted else ''
    # )
    return customer_standardised_CL_df

def generate_trans_df(sales_df, collections_df):
    Trans_df = trans_df_creation(sales_df, collections_df)
    Trans_df = Trans_df[Trans_df['Time Stamp'] <= cutoff_date]
    return Trans_df
def generate_knock_offs(trans_df):
    return knockoff_df_creation(trans_df)
def generate_profit_df(trans_df):
    return profitability(trans_df)
def generate_standardisation(trans_df, knock_df, profit_df):
    return standardization_CL(trans_df, knock_df, profit_df)

# def main_transformation(Sales_df, Collection_df):
    
#     trans_df = trans_df_creation(Sales_df, Collection_df)
#     yield 'Transformation', trans_df
    
#     knock_off_df = knockoff_df_creation(trans_df)
#     yield 'Knock Off', knock_off_df
    
#     ageing_df = aging_df_creation(knock_off_df)
#     yield 'Aging', ageing_df
    
#     profitability_df = profitability(trans_df)
#     yield 'Profitability', profitability_df
    
#     customer_standardised_CL_df = standardization_CL(trans_df, knock_off_df, profitability_df)
#     yield 'Customer Standardised Credit Limit', customer_standardised_CL_df


# def main_transformation(Sales_df, Collection_df, step=None):
#     steps = {
#         'Transformation': lambda: trans_df_creation(Sales_df, Collection_df),
#         'Knock Off': lambda: knockoff_df_creation(trans_df_creation(Sales_df, Collection_df)),
#         'Aging': lambda: aging_df_creation(knockoff_df_creation(trans_df_creation(Sales_df, Collection_df))),
#         'Profitability': lambda: profitability(trans_df_creation(Sales_df, Collection_df)),
#         'Customer Standardised Credit Limit': lambda: standardization_CL(
#             trans_df_creation(Sales_df, Collection_df), 
#             knockoff_df_creation(trans_df_creation(Sales_df, Collection_df)), 
#             profitability(trans_df_creation(Sales_df, Collection_df))
#         )
#     }
#     if step and step in steps:
#         return steps[step]()  # Run only the required function
#     else:
#         for key, func in steps.items():
#             yield key, func()