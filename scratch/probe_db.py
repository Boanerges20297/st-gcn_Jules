import pymysql
import os
import json
import logging
import decimal
from dotenv import load_dotenv

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')
load_dotenv()

def check_and_download_mysql_data():
    json_path = 'data/raw/dados_status.json'
    test_json_path = 'scratch/dados_status_test.json'
    max_local_id = 0
    local_data_exists = False
    
    if os.path.exists(json_path):
        try:
            with open(json_path, 'r', encoding='utf-8') as f:
                content = json.load(f)
            # Traverse to find the table data
            for item in content:
                if isinstance(item, dict) and item.get('type') == 'table' and 'data' in item:
                    data_rows = item['data']
                    if data_rows:
                        ids = []
                        for row in data_rows:
                            val = row.get('id')
                            if val is not None:
                                try:
                                    ids.append(int(val))
                                except ValueError:
                                    pass
                        if ids:
                            max_local_id = max(ids)
                            local_data_exists = True
                            logging.info(f"Local dados_status.json max ID: {max_local_id}")
        except Exception as e:
            logging.warning(f"Could not parse local dados_status.json: {e}")
            
    # Connect to MySQL database
    host = os.getenv('MYSQL_HOST', '').replace('"', '')
    port = int(os.getenv('MYSQL_PORT', '3306').replace('"', ''))
    user = os.getenv('MYSQL_USER', '').replace('"', '')
    password = os.getenv('MYSQL_PASSWORD', '').replace('"', '')
    database = os.getenv('MYSQL_DATABASE', '').replace('"', '')
    
    if not host or not user or not database:
        logging.warning("MySQL credentials not fully configured in .env. Skipping database extraction.")
        return
        
    try:
        conn = pymysql.connect(
            host=host,
            port=port,
            user=user,
            password=password,
            database=database,
            charset='utf8mb4',
            cursorclass=pymysql.cursors.DictCursor
        )
        try:
            with conn.cursor() as cursor:
                # Find maximum ID in DB
                cursor.execute("SELECT MAX(id) as max_id FROM dados_status")
                res = cursor.fetchone()
                max_db_id = res['max_id'] if res and res['max_id'] is not None else 0
                
                logging.info(f"Database max ID: {max_db_id}")
                
                # Check if there are new records (using a test condition or actual check)
                if max_db_id > max_local_id or not local_data_exists:
                    logging.info(f"New data detected! Fetching all rows from dados_status...")
                    # Get all rows
                    cursor.execute("SELECT * FROM dados_status ORDER BY id DESC")
                    rows = cursor.fetchall()
                    
                    # Convert types to match standard json format (e.g. decimals, dates, times, none to null)
                    formatted_rows = []
                    for row in rows:
                        formatted_row = {}
                        for k, v in row.items():
                            if v is None:
                                formatted_row[k] = None
                            elif isinstance(v, (int, float, decimal.Decimal)):
                                formatted_row[k] = str(v)
                            elif hasattr(v, 'strftime'):  # dates, times, datetimes
                                if hasattr(v, 'hour'): # time or datetime
                                    # For timedelta objects (which pymysql can return for TIME columns)
                                    import datetime
                                    if isinstance(v, datetime.timedelta):
                                        # format timedelta as HH:MM:SS
                                        total_seconds = int(v.total_seconds())
                                        hours = total_seconds // 3600
                                        minutes = (total_seconds % 3600) // 60
                                        seconds = total_seconds % 60
                                        formatted_row[k] = f"{hours:02d}:{minutes:02d}:{seconds:02d}"
                                    else:
                                        formatted_row[k] = v.strftime('%H:%M:%S')
                                else: # date
                                    formatted_row[k] = v.strftime('%Y-%m-%d')
                            elif isinstance(v, bytes):
                                formatted_row[k] = v.decode('utf-8', errors='ignore')
                            else:
                                formatted_row[k] = str(v)
                        formatted_rows.append(formatted_row)
                    
                    # Build phpMyAdmin style JSON
                    json_data = [
                        {"type": "header", "version": "5.1.3", "comment": "Export to JSON plugin for PHPMyAdmin"},
                        {"type": "database", "name": database},
                        {
                            "type": "table",
                            "name": "dados_status",
                            "database": database,
                            "data": formatted_rows
                        }
                    ]
                    
                    # Write to test_json_path first
                    os.makedirs(os.path.dirname(test_json_path), exist_ok=True)
                    with open(test_json_path, 'w', encoding='utf-8') as f:
                        json.dump(json_data, f, ensure_ascii=False, indent=4)
                    logging.info(f"Successfully downloaded {len(formatted_rows)} rows to {test_json_path}")
                else:
                    logging.info("No new data in MySQL database. Skipping download.")
        finally:
            conn.close()
    except Exception as e:
        logging.error(f"Error querying MySQL database: {e}")

check_and_download_mysql_data()
