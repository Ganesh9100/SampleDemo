# import logging
import base64
import boto3
import os
import json
from urllib.parse import unquote_plus # space issue in s3 file name resolver 
from parser import (
    extract_text,
    map_word_id,
    extract_table_info,
    get_key_map,
    get_value_map,
    get_kv_map,
)

s3_client = boto3.client('s3')
textract = boto3.client("textract",region_name="us-east-1")
# s3://ganesh-poc/input-files/
def lambda_handler(event, context):
    BUCKET_NAME = "vthtdata"
    textract_response = {}

    
            
    def call_textract_bytes(bucket,key):
        try:
            response = textract.analyze_document(
                    Document={'S3Object': {'Bucket': bucket, "Name": key}},
                    FeatureTypes=["FORMS","TABLES"]
                )
            print("Post Response")
            # print("response",response)
            raw_txt_word = extract_text(response)
            raw_txt_line = extract_text(response,extract_by="LINE")
            word_map = map_word_id(response)
            table,table_name = extract_table_info(response, word_map)
            key_map = get_key_map(response, word_map)
            value_map = get_value_map(response, word_map)
            final_map = get_kv_map(key_map, value_map)
            
          
            res = {"table_name" : table_name, "table" : table}
            forms_detected = {"form" : final_map}
            # blank = {"Divider":"*************************************************"}
            raw = {"raw_text_line" :raw_txt_line,"raw_text_word":raw_txt_word} 
            key_value_table = {**forms_detected,**res,**raw}
            print("key_value_table is ",key_value_table)
            return key_value_table
            
        except Exception as Ee:
            return f"error with textract {Ee}"
        
    try:
        key = "updatedAdhar.jpg"
        bucket = 'vthtdata'
        final_response= call_textract_bytes(bucket,key)
    
        return final_response
    except Exception as e:
        raise IOError(e)
    