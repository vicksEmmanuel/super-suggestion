import json
from google.protobuf.json_format import MessageToDict
import asyncio
from fastapi import APIRouter, Request
from fastapi.responses import StreamingResponse

from api_service.controllers.CodeGeneratorController import CodeGeneratorController
from api_service.controllers.EmbeddingsController import EmbeddingsController
from api_service.controllers.FileController import FileController
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

@router.get("/searching")
async def testing(
    request: Request
):
    body = await request.json()
    search = body.get("search","")
    key = body.get("key","super-suggestion")
    emb = EmbeddingsController()
    emb.create_index_for_collection()
    emb.wait_for_index_building_complete()
    result = emb.embedding_retrieval(f'''
        {search}
        ''', 
        key
    )

    return {"response": result}

@router.get("/create_embeddings")
async def create_embeddings(
    request: Request
):
    
    body = await request.json()
    file_location = body.get('file_location', "./file_manager.py")
    key = body.get('key', "super-suggestion")

    emb = EmbeddingsController()
    file_controls = FileController()
    emb.create_collection_in_milvus()

    # Ensure index creation and wait for it to build
    emb.create_index_for_collection()
    emb.wait_for_index_building_complete()

    emb.download_workspace_and_create_embeddings(file_location, key)
    return {"response","Done"}

@router.post("/check_if_key_exist")
async def check_if_embedding_exists(request: Request):
    body = await request.json()
    key = body.get("key","super-suggestion")
    emb = EmbeddingsController()
    emb.create_collection_in_milvus()
    emb.create_index_for_collection()
    emb.wait_for_index_building_complete()
    return {"response": emb.check_if_embedding_exists(key)}