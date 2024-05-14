from controllers.EmbeddingsController import EmbeddingsController
import json
from fastapi import APIRouter, Request, Form, UploadFile
from api_service.controllers.FileController import FileController

router = APIRouter()

@router.post("/upload_file_to_workspace")
async def upload_code_file_to_workspace(
    request: Request,
    file: UploadFile = Form(...),
    file_path: str = Form(...),
    key: str = Form(...)
):
    fileControls = FileController()
    file_content = await file.read()
    destination_key = fileControls.upload_code_file_to_workspace_folder(file_content, file.filename, file_path, key)
    return {"response": "Success", "destination_key": destination_key}

@router.post("/delete_file_from_workspace")
async def delete_code_file_from_workspace(
    request: Request,
    file_name: str = Form(...),
    file_path: str = Form(...),
    key: str = Form(...)
):
    fileControls = FileController()
    fileControls.delete_code_file_from_workspace_folder(file_name, file_path, key)
    return {"response": "Success"}

@router.post("/download_workspace")
async def download_code_workspace(
    request: Request,
    workspace_folder: str = Form(...),
    key: str = Form(...)
):
    fileControls = FileController()
    fileControls.download_code_workspace_folder(workspace_folder, key)
    return {"response": "Success"}


@router.get("/testing")
async def testing(
    request: Request
):
    emb = EmbeddingsController()
    collection = emb.create_collection_in_milvus()
    # Ensure index creation and wait for it to build
    emb.create_index_for_collection(collection)
    emb.wait_for_index_building_complete(collection.name)

    # Download workspace and create embeddings
    emb.download_workspace_and_create_embeddings(".", "super-suggestion")