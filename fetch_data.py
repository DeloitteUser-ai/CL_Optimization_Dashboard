import pyodbc
import pandas as pd
from datetime import datetime
import time

# SQL Server connection details
SERVER = 'WIN-O0UQP979DTI\MSSQLSERVER01'
DATABASE = 'SAFEX'
CONNECTION_STRING = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={SERVER};DATABASE={DATABASE};Trusted_Connection=yes"

def fetch_and_filter_customer_data(customer_ids):
    try:
        # Connect to SQL Server
        with pyodbc.connect(CONNECTION_STRING) as connection:
            # Fetch only relevant customers directly from SQL
            # customer_query = """
            # SELECT * FROM [Customer Master]
            # where CRDAT > '20240331' and CRDAT < '20240531'
            # """
            # query = f"""SELECT * 
            # FROM [Customer Master] 
            # WHERE KUNNR IN('0011001634')"""
            formatted_ids = ', '.join(f"'{id}'" for id in customer_ids)
            query = f"""SELECT * 
                FROM [Customer Master] where KUNNR in
                ({formatted_ids})"""
            # query = f"""SELECT * 
            #     FROM [Customer Master] where KUNNR in
            #     ('0011005431', '0011010820', '0011010820', '0011005376', '0011010422', '0031002586',
            #     '0011007499', '0031002944', '0011002648', '0011007481', '0031004056', '0011006512',
            #     '0011010362', '0011003362', '0021004175', '0011011115', '0011003086', '0011012346',
            #     '0021003507', '0021009018', '0011006930', '0011001234', '0011002735', '0031001616',
            #     '0031004788', '0021007953')"""
                # WHERE KUNNR IN ('0011010362')"""
                # where KUNNR IN ('0011010362', '0011010422', '0011011115', '0011012346', '0021007953', '0021009018')"""
            # # query = f"""SELECT * 
            # FROM [Customer Master] 
            # where KUNNR IN (['0021007953', '0021007954', '0021007955', '0021007956', '0021007957', '0021007958', 
            # '0021007959', '0021007960', '0021007961', '0021007962', '0021007963', '0021007964', 
            # '0021007965', '0021007966', '0021007967', '0021007968', '0021007969', '0021007970', '0021007971',
            # '0021007972', '0021007973', '0021007974', '0021007975', '0021007976', '0021007977', '0021007978'])"""
            # WHERE KUNNR IN ('0011001634', '0021007448', '0021004217', '0031005775', '0031005528')"""
            # WHERE KUNNR IN('0011008856', '0011003597', '0031004203', '0011000203', '0011000127')"""
            customer_master = pd.read_sql(query, connection)

            credit_query = """select * from [Customer Credit Details]"""
            credit_details = pd.read_sql(credit_query, connection)

        # Rename columns
        customer_master.rename(columns={
            'KUNNR': 'Customer',
            'NAME1': 'Name 1',
            'NAME2': 'Name 2',
            'ADRNR': 'Address',
            'STRAS': 'Street',
            'ANRED': 'Title',
            'REGIO': 'State_Code',
            'PSTLZ': 'Postal Code',
            'ORT01': 'City',
            'STCD3': 'Tax Number 3',
            'ZPANNO': 'PAN',
            'CRDAT': 'Start Date on',
            'CRTIM': 'Created at',
            'NAME3': 'Customer Name',
            'ZDIR': 'Director Code',
            'ZZONE': 'ZONE',
            'ZNAME': 'ZHIRE-ZNAME',
            'BUKRS': 'Company Code',
            'BUTXT': 'Company Name',
            'BEZEI': 'State',
            'ZTEAM': 'Team Details',
            'ZRMC': 'Supplier',
            'ZRMN': 'RM Name',
            'ZASMC': 'ASM Code',
            'ZSALESPER': 'Sales Person Code',
            'ZTEAM1': 'Team Details',
            'ZVPA': 'Text',
            'ZRMAIL': 'Text',
            'ZSTATE': 'State_Code',
            'ZHQ': 'HQ',
            'ZAREA': 'c',
            'ASM_NAME': 'ASM Name',
            'SALE_PERSON_NAME': 'Sales Person Name'
        }, inplace=True)

        credit_details.rename(columns={
            'PARTNER': 'BusinessPartner',
            'CREDIT_SGMNT': 'Credit Segment',
            'CREDIT_LIMIT': 'Credit Limit',
            'GJAHR': 'Fiscal Year',
            'UMSAV': 'Bal.Carried Fwd',
            'EXPOSURE_AMOUNT': 'Exposure Amount'
        }, inplace=True)

        # Standardize Customer IDs
        customer_master['Customer'] = customer_master['Customer'].astype(str).str.lstrip('0')
        credit_details['BusinessPartner'] = credit_details['BusinessPartner'].astype(str).str.lstrip('0')

        # Map opening balances from credit details to customer master
        customer_master['Opening_Balances'] = customer_master['Customer'].map(
            dict(zip(credit_details['BusinessPartner'], credit_details['Bal.Carried Fwd']))
        )
        
        # Filter customers with valid credit limit and opening balance
        filtered_customers = customer_master[
            (customer_master['Opening_Balances'].notna()) & 
            (customer_master['Customer'].map(dict(zip(credit_details['BusinessPartner'], credit_details['Credit Limit']))).fillna(0) > 0)
        ]
        filtered_customers.columns = [col if (col == 'Customer' or col == 'Time Stamp') else "CM_" + col for col in filtered_customers.columns]
        filtered_customers['CM_Credit Limit'] = filtered_customers['Customer'].map(dict(zip(credit_details['BusinessPartner'], credit_details['Credit Limit'])))
        # Rename columns for clarity
        filtered_customers.rename(columns={
            'CM_Opening_Balances': 'CM_Opening Balances'
        }, inplace=True)

        return filtered_customers

    except Exception as e:
        print(f"Error fetching or processing customer data: {e}")
        return None


def fetch_sales_data(customer_ids):
    server = 'WIN-O0UQP979DTI\MSSQLSERVER01'
    database = 'SAFEX'
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"
    view_name = 'dbo.[Billing Details]'
    formatted_ids = ', '.join(f"'{id}'" for id in customer_ids)
    try:
        # Connect to SQL Server
        with pyodbc.connect(connection_string) as connection:
            # Fetch only sales data after 31st March 2024
            # query = f"""
            # SELECT * FROM {view_name}
            # where FKDAT >= '20240331' and FKDAT <= '20240531'
            # """
            query = f"""SELECT * 
                FROM [Billing Details] where KUNNR in
                ({formatted_ids})"""
            # query = """SELECT * 
            #     FROM [Billing Details] 
            #     WHERE KUNNR IN
            #     ('0011005431', '0011010820', '0011010820', '0011005376', '0011010422', '0031002586',
            #     '0011007499', '0031002944', '0011002648', '0011007481', '0031004056', '0011006512',
            #     '0011010362', '0011003362', '0021004175', '0011011115', '0011003086', '0011012346',
            #     '0021003507', '0021009018', '0011006930', '0011001234', '0011002735', '0031001616',
            #     '0031004788', '0021007953')"""
            # where KUNNR IN ('0011010362', '0011010422', '0011011115', '0011012346', '0021007953', '0021009018')"""
            # query = f"""SELECT * 
            #     FROM [Billing Details]
            #     where KUNNR IN (['0021007953', '0021007954', '0021007955', '0021007956', '0021007957', '0021007958', '0021007959', '0021007960', '0021007961', '0021007962', '0021007963', '0021007964', '0021007965', '0021007966', '0021007967', '0021007968', '0021007969', '0021007970', '0021007971', '0021007972', '0021007973', '0021007974', '0021007975', '0021007976', '0021007977', '0021007978'])"""
                # AND FKDAT >= '20220401' AND FKDAT <= '20240331'
                # WHERE KUNNR IN ('0011001634', '0021007448', '0021004217', '0031005775', '0031005528')"""
                # WHERE FKDAT >= '20220401' AND FKDAT <= '20240331'"""
                
            # # WHERE KUNNR IN ('0011008856', '0011003597', '0031004203', '0011000203', '0011000127')
            # query = f"""SELECT * 
            # FROM [Billing Details]
            # WHERE KUNNR IN ('0031006972', '0021010218', '0021010229', '0021010309', '0011012965', '0031006663', '0021010647', '0021010694', '0031006648', '0011012831')"""
            # query = f"""SELECT * 
            # FROM [Billing Details]
            # WHERE KUNNR IN ('0011001634')"""
            df = pd.read_sql(query, connection)

        # Rename columns
        
        rename_columns = {
            'VBELN': 'Billing Doc.',
            'POSNR_I': 'Item',
            'FKART': 'Billing Type',
            'VTEXT': 'Description',
            'FKTYP': 'Billing Category',
            'FKDAT': 'Billing Date',
            'GJAHR': 'Fiscal Year',
            'BELNR': 'Document Number',
            'POPER': 'Posting Period',
            'ZTERM': 'Payt Terms',
            'REGIO': 'Region',
            'BUKRS': 'Company Code',
            'TAXK1': 'Tax Cls.1 Cust.',
            'TAXK2': 'Tax Cls.2 Cust.',
            'TAXK3': 'Tax Cls.3 Cust.',
            
            'SFAKN': 'Canceled Bill.Dc',
            'FKSTO': 'Canceled',
            'FKIMG_I': 'Invoiced Qty',
            'VTKME_I': 'Sales Unit',
            'MEINS_I': 'Base Unit',
            'SMENG_I': 'Scale Quantity',
            'FKLMG_I': 'Bill.Qty in SKU',
            'LMENG_I': 'Required Qty',
            'KURSK_I': 'Exchange Rate',
            'NETWR_I': 'Net Value.1',
            'AUBEL_I': 'Sales Document',
            
            'AUREF_I': 'Sales Document Refer',
            'MATNR_I': 'Material',
            'ARKTX_I': 'Item Descr.',
            'ZTECHNICAL': 'Technical',
            'MVGR1': 'Material Group 1',
            
            'CHARG_I': 'Batch',
            'MATKL_I': 'Material Group',
            'WGBEZ60': 'Mat. Grp Desc. 2',
            'WERKS_I': 'Plant',
            'PRSFD_I': 'Pricing',
            'SKTOF_I': 'Cash Discount',
            'KONDM_I': 'Mat. Price Grp',
            'LGORT_I': 'Stor. Loc.',
            'WAVWR_I': 'Cost',
            'KZWI1_I': 'Subtotal 1',
            'KZWI2_I': 'Subtotal 2',
            'KZWI3_I': 'Subtotal 3',
            'KZWI4_I': 'Subtotal 4',
            'KZWI5_I': 'Subtotal 5',
            'KZWI6_I': 'Subtotal 6',
            'MWSBP_I': 'Tax Amount',
            'TOTAL': 'Total Amount',
            'KUNNR': 'Customer'
        }
        df.rename(columns=rename_columns, inplace=True)

        return df

    except Exception as e:
        print(f"Error fetching sales data: {e}")
        return None

def clean_sales_data(sales_df, customer_df, cutoff_date, start_cutoff_date):
    cutoff_date = pd.to_datetime(cutoff_date)
    start_cutoff_date = pd.to_datetime(start_cutoff_date, format='%d-%m-%Y')
    sales_df['Billing Date'] = pd.to_datetime(sales_df['Billing Date'])
    sales_df['Customer'] = sales_df['Customer'].astype(str).str.lstrip('0')
    # getting only common customers in sales df
    sales_df = sales_df[sales_df['Billing Date'] <= cutoff_date]
    sales_df = sales_df[sales_df['Billing Date'] >= start_cutoff_date]
    sales_df['Customer'] = sales_df['Customer'].astype(str)
    customer_df['Customer'] = customer_df['Customer'].astype(str)
    customer_df['Customer'] = customer_df['Customer'].astype(str).str.lstrip('0')
    sales_df = sales_df[sales_df['Customer'].isin(customer_df['Customer'])]
    # sales_df = sales_df.loc[:, ~sales_df.columns.duplicated()]
    sales_df.rename(columns={
        'Bill.Qty in SKU': 'Billing Qty in SKU',
        'Tax Amount': 'Tax Amount.1',
        'Payt Terms': 'Payment terms',
        'Billing Doc.': 'Billing Document',
        'Item Descr.': 'Item Description'
    }, inplace = True)
    return sales_df

def clean_collections(collections_df, customer_df, cutoff_date, start_cutoff_date):
    # today = datetime.today()
    # fiscal_year_end = pd.to_datetime(f"{today.year - 1 if today.month <= 3 else today.year}-03-31")
    # cutoff_date = fiscal_year_end
    # converting posting date to datetime value
    collections_df['Posting Date'] = pd.to_datetime(collections_df['Posting Date'])
    cutoff_date = pd.to_datetime(cutoff_date)
    start_cutoff_date = pd.to_datetime(start_cutoff_date, format='%d-%m-%Y')
    # taking posting date <= cutoff date in collections_df
    collections_df = collections_df[collections_df['Posting Date'] <= cutoff_date]
    collections_df = collections_df[collections_df['Posting Date'] >= start_cutoff_date]
    # removing 00 from the prefixes on all the customer ids in collections df
    # collections_df = collections_df[collections_df['Document Type'] == 'DZ']
    collections_df = collections_df[collections_df['Document Type'].isin(['DZ', 'SA', 'DA', 'DG', 'DR'])]
    collections_df['Customer'] = collections_df['Customer'].str.lstrip('0')
    customer_df['Customer'] = customer_df['Customer'].astype(str).str.lstrip('0')
    collections_df['Customer'] = collections_df['Customer'].astype(str)
    customer_df['Customer'] = customer_df['Customer'].astype(str)
    # common customers between collections and customer master 
    common_customers_collections = set(collections_df['Customer']).intersection(set(customer_df['Customer']))
    # taking only common customers in collections_df 
    collections_df = collections_df[collections_df['Customer'].isin(common_customers_collections)]
    collections_df.rename(columns = {
        'Amnt in Bal Tr Crcy': 'Amount in Balance Transaction Currency',
        'Payt terms': 'Payment Terms'
    }, inplace = True)
    return collections_df

def fetch_collections_data(customer_ids):
    server = 'WIN-O0UQP979DTI\MSSQLSERVER01'
    database = 'SAFEX'
    connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"

    # Create a connection string
    # connection_string = f"DRIVER={{ODBC Driver 17 for SQL Server}};SERVER={server};DATABASE={database};Trusted_Connection=yes"

    # Connect to the database
    connection = pyodbc.connect(connection_string)

    # Define the view name
    view_name = '[Accounting entries]'  # Use brackets if the view name contains spaces
    formatted_ids = ', '.join(f"'{id}'" for id in customer_ids)
    # Query the view and load it into a DataFrame
    # query = """SELECT * FROM dbo.[Accounting entries] WHERE BUDAT > '20240331' and BUDAT <= '20240531'"""
    # query = f"""select * from [Accounting entries] where KUNNR in ('0031006972', '0021010218', '0021010229', '0021010309', '0011012965', '0031006663', '0021010647', '0021010694', '0031006648', '0011012831')"""
    query = f"""SELECT * 
                FROM [Accounting Entries] where KUNNR  in 
                 ({formatted_ids})"""
    # query = f"""SELECT * 
    #             FROM [Accounting Entries] 
    #             WHERE KUNNR IN ('0011005431', '0011010820', '0011010820', '0011005376', '0011010422', '0031002586',
    #             '0011007499', '0031002944', '0011002648', '0011007481', '0031004056', '0011006512',
    #             '0011010362', '0011003362', '0021004175', '0011011115', '0011003086', '0011012346',
    #             '0021003507', '0021009018', '0011006930', '0011001234', '0011002735', '0031001616',
    #             '0031004788', '0021007953')"""
    # query = f"""SELECT * 
    #             FROM [Accounting Entries] 
    #             where KUNNR IN ('0011010362', '0011010422', '0011011115', '0011012346', '0021007953', '0021009018')"""
                # WHERE KUNNR IN ('0011010362')"""
                # ('0031004788', '0021009018', '0011010422', '0021007953', '0011003086', '0011011115','0011012346', '0011010362')"""
    # query = f"""select * from [Accounting entries]
    # where KUNNR IN ('0011001634')"""
    # where KUNNR IN ('0011001634')
    # AND BUDAT >= '20220401' and BUDAT <= '20240331'"""
    # WHERE KUNNR IN ('0011001634', '0021007448', '0021004217', '0031005775', '0031005528')"""
    # where BUDAT >= '20220401' and BUDAT <= '20240331'"""
    
    # where KUNNR  in ('0011008856', '0011003597', '0031004203', '0011000203', '0011000127')"""
    df = pd.read_sql(query, connection)

    # Column renaming dictionary
    rename_columns = {
        'RLDNR': 'Ledger',
        'RBUKRS': 'Company Code',
        'GJAHR': 'Fiscal Year',
        'BELNR': 'Document Number',
        'DOCLN': 'Posting Item',
        'KUNNR': 'Customer',
        'TSL': 'Amnt in Bal Tr Crcy',
        'LBKUM': 'Inv. Quantity',
        'DRCRK': 'Debit/Credit',
        'PERIV': 'Fi.Year Variant',
        'POPER': 'Posting period',
        'BUDAT': 'Posting Date',
        'BLART': 'Document Type',
        'LINETYPE': 'Item Category',
        'SGTXT': 'Text',
        'AUGDT': 'Clearing',
        'AUGBL': 'Clrng doc.',
        'AUGGJ': 'Clrg Fiscal Yr',
        'RACCT': 'Account Number',
        'TXT50': 'Long Text',
        'ZTERM': 'Payt terms',
        'BLDAT': 'Document Date',
        'ZTAG1': 'Days from Baseline D',
        'NET_DUE': 'Net Due Date'
    }

    # Rename columns
    df.rename(columns=rename_columns, inplace=True)

    # Close the connection
    connection.close()

    return df


def net_rev_365(trans_df_last_year, profit_df, profit_df_last_year):
    if trans_df_last_year.empty:
        net_rev_365 = 0 if profit_df.empty else profit_df['Net Revenue'].values[0]
    else:
        df1 = profit_df_last_year
        df1.fillna(0, inplace = True)
        net_rev_365 = df1['Net Revenue'].sum()
    return net_rev_365

def avg_invoices_per_year(trans_df):
    invoices = trans_df[trans_df['S_Billing Type'] == 'ZINV']
    invoices_per_year = invoices.groupby('Year')['S_Billing Document'].nunique().reset_index()
    avg_invoices = invoices_per_year['S_Billing Document'].mean()
    avg_invoices = int(avg_invoices) if not pd.isna(avg_invoices) else 0
    return avg_invoices


def calculate_debtor_days(avg_bal, net_rev_365, tenure, activity):
    if net_rev_365 == 0:
        debtor_days = None
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
        except OverflowError:
            debtor_days = 0  # Handle overflow gracefully
        except Exception as e:
            print(f"An error occurred: {e}")
            debtor_days = 0
    return debtor_days


def avg_bal(trans_df, activity, tenure, threshold):
    trans_df['Time Stamp'] = pd.to_datetime(trans_df['Time Stamp'])
    invoices = trans_df[trans_df['S_Billing Type'] == 'ZINV']
    returns = trans_df[trans_df['S_Billing Type'].isin(['ZIRE', 'ZWCR', 'ZCSI'])]
    prev_activity = None
    last_sale_activity = invoices['S_Billing Date'].max()
    threshold_yearmonth = threshold.to_period('M') if pd.notna(threshold) else None
    if pd.notna(threshold):
        if threshold_yearmonth not in trans_df['Time Stamp'].dt.to_period('M').unique():
            prev_rows = trans_df[trans_df['Time Stamp'] <= threshold]
            if not prev_rows.empty:
                prev_activity = pd.to_datetime(prev_rows.iloc[-1]['Time Stamp'])
            else:
                prev_activity = threshold
    else:
        prev_activity = None
    # Convert dates to string or handle None
    threshold = threshold.strftime('%d-%m-%Y') if pd.notna(threshold) else None
    prev_activity = prev_activity.strftime('%d-%m-%Y') if prev_activity else None
    # Calculate running outstanding and sales
    if prev_activity:
        
        running_df = trans_df[(trans_df['Time Stamp'] >= prev_activity) & (trans_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
        Total_Sales = invoices[(invoices['Time Stamp'] >= prev_activity) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
        Net_Returns = returns[(returns['Time Stamp'] >= prev_activity) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
        running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
        grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
    else:
        if threshold is not None and last_sale_activity is not None:
            running_df = trans_df[(trans_df['Time Stamp'] >= threshold) & (trans_df['Time Stamp'] <= pd.to_datetime(last_sale_activity))]
            # Total_Sales = invoices[(invoices['Time Stamp'] >= threshold) & (invoices['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            # Net_Returns = returns[(returns['Time Stamp'] >= threshold) & (returns['Time Stamp'] <= pd.to_datetime(last_sale_activity))]['S_Net Value.1'].sum()
            running_df['YearMonth'] = running_df['Time Stamp'].dt.to_period('M')
            grouped_running = running_df.groupby('YearMonth')['Running Outstanding'].last().reset_index()
        else:
            running_df = trans_df
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
        avg_bal = round((last_12_months['Running Outstanding'].sum() / 12), 2)  if tenure > 1 else round((merged_running['Running Outstanding'].sum() / merged_running.shape[0]), 2)# Use the correct denominator
    else:
        avg_bal = 0
    return avg_bal

