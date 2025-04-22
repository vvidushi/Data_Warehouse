import boto3
import pandas as pd
import numpy as np
import io
from datetime import datetime
import os

S3_BUCKET = <S3_BUCKET>
STAGE_PREFIX = 'stage/'
PROD_PREFIX = 'prod/'

DATASET_MAPPING = {
    'HC_Coverage': 'healthcare_coverage',
    'HealthGFCF': 'health_capital_formation',
    'HEF_Main': 'health_expenditure_financing',
    'HP_cost': 'health_provider_costs',
    'HP_revenue' : 'health_revenue'
}

def read_csv_from_s3_folder(s3_client, dataset_folder):
    try:

        file_key = f"{STAGE_PREFIX}{dataset_folder}/data.csv"
        response = s3_client.get_object(Bucket=S3_BUCKET, Key=file_key)
        df = pd.read_csv(io.BytesIO(response['Body'].read()))
        if 'dataset_name' not in df.columns and dataset_folder in DATASET_MAPPING:
            df['dataset_name'] = DATASET_MAPPING[dataset_folder]
            
        print(f"Successfully loaded {dataset_folder}/data.csv with {df.shape[0]} rows")
        return df
    except Exception as e:
        print(f" Error reading {dataset_folder}/data.csv: {str(e)}")
        return None

def write_df_to_s3(s3_client, df, table_name):
    """Write DataFrame to S3 as CSV"""
    try:
        file_key = f"{PROD_PREFIX}{table_name}.csv"
        csv_buffer = io.StringIO()
        df.to_csv(csv_buffer, index=False)
        
        s3_client.put_object(
            Bucket=S3_BUCKET,
            Key=file_key,
            Body=csv_buffer.getvalue()
        )
        print(f"Successfully wrote {table_name} ({df.shape[0]} rows) to s3://{S3_BUCKET}/{file_key}")
        return True
    except Exception as e:
        print(f" Error writing {table_name}: {str(e)}")
        return False

def standardize_columns(df):
    column_map = {
        'country_code': ['REF_AREA', 'country_code', 'ref_area'],
        'country_name': ['Reference area', 'country_name'],
        'year': ['TIME_PERIOD', 'year', 'time_period'],
        'value': ['OBS_VALUE', 'value', 'obs_value'],
        'insurance_type': ['insurance_type', 'INSURANCE_TYPE'],
        'financing_scheme': ['financing_scheme', 'FINANCING_SCHEME'],
        'provider': ['provider', 'PROVIDER', 'health_care_provider']
    }
    df.columns = [col.lower() for col in df.columns]
    for standard_name, possible_names in column_map.items():
        for col_name in possible_names:
            if col_name.lower() in df.columns:
                df = df.rename(columns={col_name.lower(): standard_name})
                break
                
    return df

def create_healthcare_star_schema(dataframes):
    """Create star schema dimension and fact tables from input dataframes"""
    std_dataframes = {}
    for name, df in dataframes.items():
        std_dataframes[name] = standardize_columns(df)
    
    dataframes = std_dataframes
    star_schema = {}
    
    # --------- DIMENSION TABLES ---------
    
    # 1. Geography Dimension (Dim_Geography)
    geography_frames = []
    for df in dataframes.values():
        if 'country_code' in df.columns:
            geo_df = df[['country_code']].copy()
            if 'country_name' in df.columns:
                geo_df['country_name'] = df['country_name']
            geography_frames.append(geo_df)
    
    if geography_frames:
        dim_geography = pd.concat(geography_frames).drop_duplicates().reset_index(drop=True)
        
        # Handle missing country names
        if 'country_name' not in dim_geography.columns:
            dim_geography['country_name'] = dim_geography['country_code']
            
        # Add region and healthcare system attributes
        dim_geography['region'] = dim_geography['country_code'].apply(
            lambda x: 'Europe' if x in ['AUT', 'DEU', 'FRA', 'GBR', 'ITA', 'ESP', 'NLD', 'LTU'] 
            else ('North America' if x in ['USA', 'CAN'] else 'Other'))
        
        dim_geography['healthcare_system_type'] = dim_geography['country_code'].apply(
            lambda x: 'Mixed' if x == 'USA' else 'Universal')
        
        # Rename to match star schema
        dim_geography = dim_geography.rename(columns={
            'country_code': 'GEOGRAPHY_KEY',
            'country_name': 'Country_Name',
            'region': 'Region',
            'healthcare_system_type': 'Healthcare_System_Type'
        })
        
        star_schema['Dim_Geography'] = dim_geography
    
    # 2. Time Dimension (Dim_Time)
    time_frames = []
    for df in dataframes.values():
        if 'year' in df.columns:
            time_df = df[['year']].copy()
            time_frames.append(time_df)
    
    if time_frames:
        dim_time = pd.concat(time_frames).drop_duplicates().reset_index(drop=True)
        
        # Convert to numeric and sort
        dim_time['year'] = pd.to_numeric(dim_time['year'], errors='coerce')
        dim_time = dim_time.dropna().sort_values('year').reset_index(drop=True)
        dim_time['year'] = dim_time['year'].astype(int)
        
        # Add time attributes
        dim_time['time_period'] = dim_time['year']
        dim_time['quarter'] = 'Q4'  # Annual data default
        dim_time['month'] = 12      # December for annual data
        dim_time['freq'] = 'A'      # Annual frequency
        
        # Rename to match star schema
        dim_time = dim_time.rename(columns={
            'year': 'TIME_KEY',
            'time_period': 'TIME_PERIOD',
            'quarter': 'Quarter',
            'month': 'Month',
            'freq': 'FREQ'
        })
        
        # Add Year column separately after renaming
        dim_time['Year'] = dim_time['TIME_KEY']
        
        star_schema['Dim_Time'] = dim_time
    
    # 3. Insurance Type Dimension (Dim_InsuranceType)
    insurance_types = set()
    for df in dataframes.values():
        if 'insurance_type' in df.columns:
            types = df['insurance_type'].dropna().unique()
            insurance_types.update(types)
    
    # Create Insurance Type dimension
    if insurance_types:
        types_list = list(insurance_types)
    else:
        types_list = ['PHI', 'SHI', 'OOP']
    
    dim_insurance = pd.DataFrame({
        'INSURANCE_TYPE_KEY': types_list,
        'INSURANCE_TYPE': [str(t).replace('_', ' ') for t in types_list],
        'Coverage_Level': np.random.choice(['Premium', 'Standard', 'Basic'], size=len(types_list)),
        'Deductible_Avg': np.random.choice([0, 500, 1000, 2000], size=len(types_list)),
        'Copay_Percent': np.random.randint(0, 30, size=len(types_list))
    })
    star_schema['Dim_InsuranceType'] = dim_insurance
    
    # 4. Demographics Dimension (synthetic)
    dim_demographics = pd.DataFrame({
        'DEMOGRAPHICS_KEY': range(1, 6),
        'Age_Group': ['18-24', '25-34', '35-44', '45-54', '55+'],
        'Income_Level': ['Low', 'Medium-Low', 'Medium', 'Medium-High', 'High'],
        'Employment_Status': ['Student', 'Employed', 'Employed', 'Employed', 'Retired']
    })
    star_schema['Dim_Demographics'] = dim_demographics
    
    # 5. Provider Dimension
    provider_frames = []
    for df in dataframes.values():
        if 'provider' in df.columns:
            provider_df = df[['provider']].drop_duplicates()
            if 'health_care_provider' in df.columns:
                provider_df['provider_name'] = df['health_care_provider']
            provider_frames.append(provider_df)
    
    if provider_frames:
        dim_provider = pd.concat(provider_frames).drop_duplicates().reset_index(drop=True)
        
        # Add provider attributes
        dim_provider['provider_type'] = np.random.choice(
            ['Hospital', 'Clinic', 'Pharmacy', 'Other'], 
            size=len(dim_provider)
        )
        dim_provider['ownership'] = np.random.choice(
            ['Public', 'Private', 'Non-Profit'], 
            size=len(dim_provider)
        )
        
        # Handle missing provider names
        if 'provider_name' not in dim_provider.columns:
            dim_provider['provider_name'] = dim_provider['provider']
        
        # Rename columns
        dim_provider = dim_provider.rename(columns={
            'provider': 'PROVIDER_KEY',
            'provider_name': 'PROVIDER',
            'provider_type': 'Provider_Type',
            'ownership': 'Ownership'
        })
        
        star_schema['Dim_Provider'] = dim_provider
    else:
        # Create default provider dimension
        star_schema['Dim_Provider'] = pd.DataFrame({
            'PROVIDER_KEY': ['HP1', 'HP2', 'HP3', 'HP4'],
            'PROVIDER': ['Hospitals', 'Ambulatory care', 'Pharmacies', 'Other'],
            'Provider_Type': ['Hospital', 'Clinic', 'Pharmacy', 'Other'],
            'Ownership': ['Public', 'Private', 'Private', 'Mixed']
        })
    
    # 6. Financing Scheme Dimension
    financing_frames = []
    for df in dataframes.values():
        if 'financing_scheme' in df.columns:
            financing_df = df[['financing_scheme']].drop_duplicates()
            financing_frames.append(financing_df)
    
    if financing_frames:
        dim_financing = pd.concat(financing_frames).drop_duplicates().reset_index(drop=True)
        
        # Add revenue source
        dim_financing['revenue_source'] = np.random.choice(
            ['Tax', 'Premium', 'Direct Payment', 'Mixed'], 
            size=len(dim_financing)
        )
        
        # Rename columns
        dim_financing = dim_financing.rename(columns={
            'financing_scheme': 'FINANCING_SCHEME_KEY',
            'revenue_source': 'Revenue_Source'
        })
        
        # Add FINANCING_SCHEME column if missing
        dim_financing['FINANCING_SCHEME'] = dim_financing['FINANCING_SCHEME_KEY']
        
        star_schema['Dim_FinancingScheme'] = dim_financing
    else:
        # Create default financing scheme dimension
        star_schema['Dim_FinancingScheme'] = pd.DataFrame({
            'FINANCING_SCHEME_KEY': ['FS1', 'FS3', 'FS4'],
            'FINANCING_SCHEME': ['Government schemes', 'Household out-of-pocket payment', 'Voluntary health insurance schemes'],
            'Revenue_Source': ['Tax', 'Direct Payment', 'Premium']
        })
        
    # 1. Insurance Coverage Outcomes Fact Table
    coverage_dfs = []
    
    # Find healthcare coverage datasets
    for name, df in dataframes.items():
        if 'HC_Coverage' in name:
            if all(col in df.columns for col in ['country_code', 'year', 'value']):
                coverage_dfs.append(df)
    
    # Create fact table
    if coverage_dfs:
        fact_coverage = pd.concat(coverage_dfs).reset_index(drop=True)
        
        # Create ID column
        fact_coverage['coverage_id'] = fact_coverage.index + 1
        
        # Create foreign keys
        fact_coverage['time_key'] = fact_coverage['year']
        fact_coverage['geography_key'] = fact_coverage['country_code']
        
        # Handle insurance type
        if 'insurance_type' in fact_coverage.columns:
            fact_coverage['insurance_type_key'] = fact_coverage['insurance_type']
        else:
            # Default to Social Health Insurance
            fact_coverage['insurance_type_key'] = 'SHI'
        
        # Assign demographics randomly
        fact_coverage['demographics_key'] = np.random.randint(1, 6, size=len(fact_coverage))
        
        # Add synthetic metrics
        fact_coverage['insurance_enrollment_count'] = np.random.randint(10000, 1000000, size=len(fact_coverage))
        fact_coverage['patient_satisfaction_score'] = np.random.uniform(3.0, 4.9, size=len(fact_coverage)).round(1)
        
        # Rename observation value
        fact_coverage['obs_value'] = fact_coverage['value']
        
        # Select and rename final columns
        fact_coverage = fact_coverage.rename(columns={
            'coverage_id': 'Coverage_ID',
            'time_key': 'TIME_KEY',
            'geography_key': 'GEOGRAPHY_KEY',
            'insurance_type_key': 'INSURANCE_TYPE_KEY',
            'demographics_key': 'DEMOGRAPHICS_KEY',
            'obs_value': 'OBS_VALUE',
            'insurance_enrollment_count': 'Insurance_Enrollment_Count',
            'patient_satisfaction_score': 'Patient_Satisfaction_Score'
        })
        
        fact_coverage = fact_coverage[[
            'Coverage_ID', 'TIME_KEY', 'GEOGRAPHY_KEY', 'INSURANCE_TYPE_KEY', 
            'DEMOGRAPHICS_KEY', 'OBS_VALUE', 'Insurance_Enrollment_Count', 
            'Patient_Satisfaction_Score'
        ]]
        
        star_schema['Fact_Insurance_Coverage_Outcomes'] = fact_coverage
    else:
        # Create mock fact table with valid keys from dimensions
        geo_keys = star_schema['Dim_Geography']['GEOGRAPHY_KEY'].unique()[:5]
        time_keys = star_schema['Dim_Time']['TIME_KEY'].unique()[:5]
        ins_keys = star_schema['Dim_InsuranceType']['INSURANCE_TYPE_KEY'].unique()
        demo_keys = star_schema['Dim_Demographics']['DEMOGRAPHICS_KEY'].unique()
        
        # Generate combinations
        mock_rows = []
        coverage_id = 1
        
        for geo in geo_keys[:3]:  # Limit to 3 countries
            for year in time_keys[:3]:  # Limit to 3 years
                for ins in ins_keys[:2]:  # Limit to 2 insurance types
                    mock_rows.append({
                        'Coverage_ID': coverage_id,
                        'TIME_KEY': year,
                        'GEOGRAPHY_KEY': geo,
                        'INSURANCE_TYPE_KEY': ins,
                        'DEMOGRAPHICS_KEY': np.random.choice(demo_keys),
                        'OBS_VALUE': np.random.uniform(50, 99).round(1),
                        'Insurance_Enrollment_Count': np.random.randint(10000, 1000000),
                        'Patient_Satisfaction_Score': np.random.uniform(3.0, 4.9).round(1)
                    })
                    coverage_id += 1
        
        star_schema['Fact_Insurance_Coverage_Outcomes'] = pd.DataFrame(mock_rows)
    
    # 2. Healthcare Financing Fact Table
    financing_dfs = []
    
    # Find healthcare financing datasets
    for name, df in dataframes.items():
        if any(pattern in name for pattern in ['HealthGFCF', 'HEF_Main', 'HP_cost']):
            if all(col in df.columns for col in ['country_code', 'year', 'value']):
                financing_dfs.append(df)
    
    # Create fact table
    if financing_dfs:
        fact_financing = pd.concat(financing_dfs).reset_index(drop=True)
        
        # Create ID column
        fact_financing['financing_id'] = fact_financing.index + 1
        
        # Create foreign keys
        fact_financing['time_key'] = fact_financing['year']
        fact_financing['geography_key'] = fact_financing['country_code']
        
        # Handle provider
        if 'provider' in fact_financing.columns:
            fact_financing['provider_key'] = fact_financing['provider']
        else:
            # Default to Total
            fact_financing['provider_key'] = 'HP1'
        
        # Handle financing scheme
        if 'financing_scheme' in fact_financing.columns:
            fact_financing['financing_scheme_key'] = fact_financing['financing_scheme']
        else:
            # Default to Government schemes
            fact_financing['financing_scheme_key'] = 'FS1'
        
        # Extract per capita spending from data if available
        if 'unit_measure' in fact_financing.columns:
            per_capita_mask = fact_financing['unit_measure'].astype(str).str.contains('per person|per capita', case=False, na=False)
            fact_financing['per_capita_spending'] = np.nan
            if per_capita_mask.any():
                fact_financing.loc[per_capita_mask, 'per_capita_spending'] = fact_financing.loc[per_capita_mask, 'value']
        
        # Generate synthetic per capita spending if needed
        if 'per_capita_spending' not in fact_financing.columns or fact_financing['per_capita_spending'].isna().all():
            fact_financing['per_capita_spending'] = np.random.randint(500, 10000, size=len(fact_financing))
        
        # Extract GDP percentage from data if available
        if 'unit_measure' in fact_financing.columns:
            gdp_mask = fact_financing['unit_measure'].astype(str).str.contains('GDP', case=False, na=False)
            fact_financing['percent_gdp'] = np.nan
            if gdp_mask.any():
                fact_financing.loc[gdp_mask, 'percent_gdp'] = fact_financing.loc[gdp_mask, 'value']
        
        # Generate synthetic GDP percentage if needed
        if 'percent_gdp' not in fact_financing.columns or fact_financing['percent_gdp'].isna().all():
            fact_financing['percent_gdp'] = np.random.uniform(3.0, 18.0, size=len(fact_financing)).round(1)
        
        # Rename observation value
        fact_financing['obs_value'] = fact_financing['value']
        
        # Select and rename final columns
        fact_financing = fact_financing.rename(columns={
            'financing_id': 'Financing_ID',
            'time_key': 'TIME_KEY',
            'geography_key': 'GEOGRAPHY_KEY',
            'provider_key': 'PROVIDER_KEY',
            'financing_scheme_key': 'FINANCING_SCHEME_KEY',
            'obs_value': 'OBS_VALUE',
            'per_capita_spending': 'Per_Capita_Spending',
            'percent_gdp': 'Percent_GDP'
        })
        
        fact_financing = fact_financing[[
            'Financing_ID', 'TIME_KEY', 'GEOGRAPHY_KEY', 'PROVIDER_KEY', 
            'FINANCING_SCHEME_KEY', 'OBS_VALUE', 'Per_Capita_Spending', 'Percent_GDP'
        ]]
        
        star_schema['Fact_Healthcare_Financing'] = fact_financing
    else:
        # Create mock fact table with valid keys from dimensions
        geo_keys = star_schema['Dim_Geography']['GEOGRAPHY_KEY'].unique()[:5]
        time_keys = star_schema['Dim_Time']['TIME_KEY'].unique()[:5]
        prov_keys = star_schema['Dim_Provider']['PROVIDER_KEY'].unique()
        fin_keys = star_schema['Dim_FinancingScheme']['FINANCING_SCHEME_KEY'].unique()
        
        # Generate combinations
        mock_rows = []
        financing_id = 1
        
        for geo in geo_keys[:3]:  # Limit to 3 countries
            for year in time_keys[:3]:  # Limit to 3 years
                for prov in prov_keys[:2]:  # Limit to 2 providers
                    for fin in fin_keys[:2]:  # Limit to 2 financing schemes
                        mock_rows.append({
                            'Financing_ID': financing_id,
                            'TIME_KEY': year,
                            'GEOGRAPHY_KEY': geo,
                            'PROVIDER_KEY': prov,
                            'FINANCING_SCHEME_KEY': fin,
                            'OBS_VALUE': np.random.uniform(100, 10000).round(1),
                            'Per_Capita_Spending': np.random.randint(500, 15000),
                            'Percent_GDP': np.random.uniform(3.0, 18.0).round(1)
                        })
                        financing_id += 1
        
        star_schema['Fact_Healthcare_Financing'] = pd.DataFrame(mock_rows)
    
    return star_schema

def main():
    """Main ETL function to read from stage folders and create star schema"""
    try:
        # Initialize S3 client
        s3_client = boto3.client('s3')
        
        # Define dataset folders in stage/
        dataset_folders = ['HC_Coverage', 'HealthGFCF', 'HEF_Main', 'HP_cost']
        
        # Read data from stage folders
        print("Reading data from S3 stage folders...")
        dataframes = {}
        
        for folder in dataset_folders:
            df = read_csv_from_s3_folder(s3_client, folder)
            if df is not None:
                dataframes[folder] = df
        
        if not dataframes:
            print(" No data could be loaded from stage folders. Please check S3 paths.")
            return
        
        # Create star schema
        print(" Transforming data into star schema...")
        star_schema = create_healthcare_star_schema(dataframes)
        
        # Create prod folder if needed
        try:
            s3_client.head_object(Bucket=S3_BUCKET, Key=PROD_PREFIX)
        except:
            s3_client.put_object(Bucket=S3_BUCKET, Key=PROD_PREFIX)
            print(f" Created {PROD_PREFIX} folder in S3")
        
        # Write dimension and fact tables to S3
        print("Writing star schema tables to S3...")
        for table_name, df in star_schema.items():
            write_df_to_s3(s3_client, df, table_name)
        
        # Output summary
        dims = [t for t in star_schema.keys() if t.startswith('Dim_')]
        facts = [t for t in star_schema.keys() if t.startswith('Fact_')]
        
        print(f"ETL Complete: Created {len(dims)} dimension tables and {len(facts)} fact tables")
        print(f"Dimensions: {', '.join(dims)}")
        print(f"Facts: {', '.join(facts)}")
            
    except Exception as e:
        print(f" Error in ETL process: {str(e)}")
        raise

if __name__ == "__main__":
    main()
