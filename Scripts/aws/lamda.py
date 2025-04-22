import base64
import json
import boto3
import csv
from io import StringIO

s3 = boto3.client('s3')
S3_BUCKET = <S3_BUCKET>

def lambda_handler(event, context):
    print(f"Received {len(event['Records'])} records")
    
    # Dictionary to hold CSV data for each dataset
    dataset_files = {
        'HealthGFCF': {'headers': None, 'rows': []},
        'HC_Coverage': {'headers': None, 'rows': []},
        'HEF_Main': {'headers': None, 'rows': []},
        'HP_cost': {'headers': None, 'rows': []}
    }

    for record in event['Records']:
        try:
            # Decode payload
            payload = json.loads(base64.b64decode(record['kinesis']['data']))
            dataset = record['kinesis']['partitionKey']
            
            # Skip unknown datasets
            if dataset not in dataset_files:
                print(f"Unknown dataset: {dataset}")
                continue
                
            # Initialize headers if first record
            if not dataset_files[dataset]['headers']:
                dataset_files[dataset]['headers'] = list(payload.keys())
                
            # Add row data
            dataset_files[dataset]['rows'].append(
                [str(payload.get(h, '')) for h in dataset_files[dataset]['headers']]
            )
            
        except Exception as e:
            print(f"Error processing record: {str(e)}")

    # Process each dataset
    for dataset, data in dataset_files.items():
        if not data['rows']:
            continue
            
        s3_key = f"{dataset}/data.csv"
        csv_buffer = StringIO()
        writer = csv.writer(csv_buffer)
        
        existing_rows = []
        file_exists = False
        try:
            # Try to get existing CSV
            existing_csv = s3.get_object(Bucket=S3_BUCKET, Key=s3_key)['Body'].read().decode('utf-8')
            reader = csv.reader(StringIO(existing_csv))
            existing_rows = list(reader)
            file_exists = True
        except s3.exceptions.NoSuchKey:
            pass
        except Exception as e:
            print(f"Error reading existing CSV for {dataset}: {str(e)}")
            continue

        # Always write the header first
        writer.writerow(data['headers'])
        # If file existed, write all existing rows except the header
        if file_exists and existing_rows:
            existing_header = existing_rows[0]
            if existing_header != data['headers']:
                print(f"Header mismatch for {dataset}, skipping...")
                continue
            writer.writerows(existing_rows[1:])
        # Write new rows
        writer.writerows(data['rows'])
        
        # Upload to S3
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=s3_key,
            Body=csv_buffer.getvalue()
        )
        print(f"Updated {dataset} CSV with {len(data['rows'])} new records")

    return {
        'statusCode': 200,
        'body': json.dumps('CSV files updated successfully')
    }
