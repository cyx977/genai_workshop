import boto3, json, math

print("\n----Defining a tool and sending a message that will make Claude ask for tool use----\n")

session = boto3.Session()
bedrock = session.client(service_name='bedrock-runtime')

tool_list = [
    {
        "toolSpec": {
            "name": "cosine",
            "description": "Calculate the cosine of x.",
            "inputSchema": {
                "json": {
                    "type": "object",
                    "properties": {
                        "x": {
                            "type": "number",
                            "description": "The number to pass to the function."
                        }
                    },
                    "required": ["x"]
                }
            }
        }
    }
]

message_list = []

initial_message = {
    "role": "user",
    "content": [
        { "text": "What is the cosine of 7?" } 
    ],
}

message_list.append(initial_message)

response = bedrock.converse(
    modelId="us.anthropic.claude-3-7-sonnet-20250219-v1:0",
    messages=message_list,
    inferenceConfig={
        "maxTokens": 2000,
        "temperature": 0
    },
    toolConfig={
        "tools": tool_list
    },
    system=[{"text":"You must only do math by using a tool."}]
)

response_message = response['output']['message']
print(json.dumps(response_message, indent=4))
message_list.append(response_message)

print("\n----Calling a function based on the toolUse content block.----\n")

response_content_blocks = response_message['content']

for content_block in response_content_blocks:
    if 'toolUse' in content_block:
        tool_use_block = content_block['toolUse']
        tool_use_name = tool_use_block['name']
        
        print(f"Using tool {tool_use_name}")
        
        if tool_use_name == 'cosine':
            tool_result_value = math.cos(tool_use_block['input']['x'])
            print(tool_result_value)
            
    elif 'text' in content_block:
        print(content_block['text'])

