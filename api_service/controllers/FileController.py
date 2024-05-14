import boto3
import os

class FileController:
    def __init__(self):
        session = boto3.session.Session()
        self.s3_client = session.client(
            service_name='s3',
            aws_access_key_id=os.getenv("S3_BUCKET_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("S3_BUCKET_SECRET_KEY"),
        )
        self.s3_bucket_name = os.getenv("S3_BUCKET_NAME")

    def check_if_code_workspace_folder_exists(self, key: str):
        objects = self.s3_client.list_objects(Bucket=self.s3_bucket_name, Prefix=key)
        print(f"Objects: {objects}")
        return len(objects.get('Contents', [])) > 0

    def upload_code_file_to_workspace_folder(self, file_content: bytes, file_name: str, file_path: str, key: str):
        if not file_path or file_path == ".":
            destination_key = os.path.join(key, file_name)
        else:
            destination_key = os.path.join(key, file_path, file_name)
        self.s3_client.put_object(Body=file_content, Bucket=self.s3_bucket_name, Key=destination_key)
        return destination_key

    def delete_code_file_from_workspace_folder(self, file_name: str, file_path: str, key: str):
        if not file_path or file_path == ".":
            destination_key = os.path.join(key, file_name)
        else:
            destination_key = os.path.join(key, file_path, file_name)
        self.s3_client.delete_object(Bucket=self.s3_bucket_name, Key=destination_key)

    def download_code_workspace_folder(self, workspace_folder: str, key: str):
        objects = self.s3_client.list_objects(Bucket=self.s3_bucket_name, Prefix=key)
        for obj in objects.get('Contents', []):
            file_path = os.path.relpath(obj['Key'], key)
            local_file_path = os.path.join(workspace_folder, file_path)
            os.makedirs(os.path.dirname(local_file_path), exist_ok=True)
            self.s3_client.download_file(self.s3_bucket_name, obj['Key'], local_file_path)
        
        # return download path
        return workspace_folder