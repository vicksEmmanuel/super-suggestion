import json
from google.protobuf.json_format import MessageToDict
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api_service.controllers.CodeGeneratorController import CodeGeneratorController
from api_service.grpc_config.generated.code_generator_service_pb2 import GenerateCodeRequest

router = APIRouter()



@router.post("/stream")
async def test_api(request: Request):
    body = await request.json()
    prompt = body.get('prompt', "")
    suffix_prompt = body.get('suffix_prompt', "")
    req = GenerateCodeRequest(
        prompt=prompt,
        suffix_prompt=suffix_prompt
    )

    async def generate_response():
        response = CodeGeneratorController().generate_code(request=req)
        response_dict = MessageToDict(response, preserving_proto_field_name=True)
        yield f"data: {json.dumps(response_dict)}\n\n"

    # /// On Client
    # const eventSource = new EventSource('/test');

    # eventSource.onmessage = function(event) {
    #     const data = JSON.parse(event.data);
    #     console.log('Received data:', data);
    #     // Process the received data as needed
    # };

    # eventSource.onerror = function(error) {
    #     console.error('Error:', error);
    # };

    return StreamingResponse(generate_response(), media_type='text/event-stream')

@router.post("/test")
async def test_api(request: Request):
    body = await request.json()
    prompt = body.get('prompt', "")
    suffix_prompt = body.get('suffix_prompt', "")

    req = GenerateCodeRequest(
        prompt=prompt,
        suffix_prompt=suffix_prompt
    )
    response = CodeGeneratorController().generate_code(request=req)
    response_dict = MessageToDict(response, preserving_proto_field_name=True)
    return {"response": response_dict}

