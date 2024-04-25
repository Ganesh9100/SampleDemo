import json
import urllib.parse
import boto3
client = boto3.client('bedrock-runtime')

def lambda_handler(event, context):
    #extracting text 
    textract_client = boto3.client('textract')
    responseText = textract_client.detect_document_text(
    Document={
        'S3Object': {
            'Bucket': 'vthtdata',
            'Name': 'img5.jpg'
        }
    }
)   
    text = ''
    for item in responseText["Blocks"]:
        if item["BlockType"] == "LINE":
            text =text + item["Text"]+ ' '
   
    
    
    # using bedrock 
    
    print(event)
    user_prompt = "Extract all the key details as Name, DOB, Gender, Aadhar Number return as JSON" + text
    instruction = f"<s>[INST] {user_prompt} [/INST]"

    model_id = "mistral.mistral-7b-instruct-v0:2"

    body = {
        "prompt": instruction,
        "max_tokens": 200,
        "temperature": 0.1,
        }
    response = client.invoke_model(
        accept='application/json',
        body= json.dumps(body),
        contentType='application/json',
        modelId='mistral.mixtral-8x7b-instruct-v0:1') # mistral.mixtral-8x7b-instruct-v0:1, mistral.mistral-7b-instruct-v0:2'
    
    response_body = json.loads(response["body"].read())
    outputs = response_body.get("outputs")

    completions = [output["text"] for output in outputs]
    print(completions)
    st = json.loads(str(completions[0]))
    
    
    # return completions

    # response_byte = response['body'].read()
    # response_string = json.loads(response_byte)

    # t = response_string['completions'][0]['data']['text']
    return {
        'statusCode': 200,
        'body': st
    }
