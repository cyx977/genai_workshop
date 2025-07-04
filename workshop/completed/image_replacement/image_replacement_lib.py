import boto3
import json
import base64
from io import BytesIO
from random import randint


#get a BytesIO object from file bytes
def get_bytesio_from_bytes(image_bytes):
    image_io = BytesIO(image_bytes)
    return image_io


#get a base64-encoded string from file bytes
def get_base64_from_bytes(image_bytes):
    resized_io = get_bytesio_from_bytes(image_bytes)
    img_str = base64.b64encode(resized_io.getvalue()).decode("utf-8")
    return img_str


#load the bytes from a file on disk
def get_bytes_from_file(file_path):
    with open(file_path, "rb") as image_file:
        file_bytes = image_file.read()
    return file_bytes


#get the stringified request body for the InvokeModel API call
def get_image_inpainting_request_body(prompt, image_bytes=None, mask_prompt=None, negative_prompt=None):
    input_image_base64 = get_base64_from_bytes(image_bytes)
    
    body = { #create the JSON payload to pass to the InvokeModel API
        "taskType": "INPAINTING",
        "inPaintingParams": {
            "image": input_image_base64,
            "maskPrompt": mask_prompt,
        },
        "imageGenerationConfig": {
            "numberOfImages": 1,  # Number of variations to generate
            "quality": "premium",  # Allowed values are "standard" and "premium"
            "height": 512,
            "width": 512,
            "cfgScale": 8.0,
            "seed": randint(0, 100000),  # Use a random seed
        },
    }
    
    if prompt:  #Indicate what we want to insert where the masked item(s) were (blank to remove an item)
        body['inPaintingParams']['text'] = prompt #if prompt is missing, we will just remove the item(s) indicated in the mask prompt
    
    return json.dumps(body)


#get a BytesIO object from the Nova Canvas response
def get_response_image(response):

    response = json.loads(response.get('body').read())
    
    images = response.get('images')
    
    image_data = base64.b64decode(images[0])

    return BytesIO(image_data)


#generate an image using Amazon Nova Canvas
def get_image_from_model(prompt_content, image_bytes, mask_prompt=None):
    session = boto3.Session()

    bedrock = session.client(service_name='bedrock-runtime', region_name='us-east-1') #creates a Bedrock client
    
    body = get_image_inpainting_request_body(prompt_content, image_bytes, mask_prompt=mask_prompt)
    
    response = bedrock.invoke_model(body=body, modelId="amazon.nova-canvas-v1:0", contentType="application/json", accept="application/json")
    
    output = get_response_image(response)
    
    return output

