import boto3, json

# print("\n----A basic call to the Converse API----\n")

# session = boto3.Session(profile_name='patrick')
session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

message_list = []

# initial_message = {
#     "role": "user",
#     "content": [
#         { "text": "What is the capital of kathmandu ?" } 
#     ],
# }

# m1 = {
#     "role": "assistant",
#     "content": [
#         { "text": "Kathmandu" } 
#     ],
# }

# m2 = {
#     "role": "user",
#     "content": [
#         { "text": "is it good place?" } 
#     ],
# }

# message_list.append(initial_message)
# message_list.append(m1)
# message_list.append(m2)

# response = bedrock.converse(
#     modelId="us.amazon.nova-lite-v1:0",
#     # modelId="us.huggingface-asr-whisper-large-v3-turbo",
#     messages=message_list,
#     inferenceConfig={
#         "maxTokens": 100,
#         "temperature": 0.5
#     },
# )

# response_message = response['output']['message']
# print(json.dumps(response_message, indent=4))

print("\n----Including an image in a message----\n")

with open("image.webp", "rb") as image_file:
    image_bytes = image_file.read()

image_message = {
    "role": "user",
    "content": [
        { "text": "Image 1:" },
        {
            "image": {
                "format": "webp",
                "source": {
                    "bytes": image_bytes #no base64 encoding required!
                }
            }
        },
        { "text": "Please describe the image." }
    ],
}

message_list.append(image_message)

response = bedrock.converse(
    modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=message_list,
    system= [
        {
            'text': "Talk tike you're talking to the customer support with greetings from Max International"
        }
    ],
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0,
    },
)

response_message = response['output']['message']
print(json.dumps(response_message, indent=4))

message_list.append(response_message)

print("\n----Getting response metadata and token counts----\n")

print("Stop Reason:", response['stopReason'])
print("Usage:", json.dumps(response['usage'], indent=4))



