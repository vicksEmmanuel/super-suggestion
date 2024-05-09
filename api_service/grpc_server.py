from concurrent import futures
import logging
import sys
import os
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(current_dir)
sys.path.append(project_root)

from grpc_config.services.code_generator import CodeGeneratorService
from grpc_config.interceptors.logging_interceptor import LoggingInterceptor
from auth.auth_interceptor import AuthInterceptor
from grpc_reflection.v1alpha import reflection
from grpc_config.generated.code_generator_service_pb2_grpc import add_CodeGeneratorServiceServicer_to_server
from grpc_config.generated.code_generator_service_pb2 import GenerateCodeRequest, GenerateCodeResponse
from grpc_config.generated import  code_generator_service_pb2
from util.logging import logger
import grpc




def start_grpc_server():
    isProduction = os.getenv("ENVIRONMENT") == 'production'
    auth_interceptor = AuthInterceptor(
        [] if isProduction else ['/CodeGeneratorService/GenerateCode']
    )


    logging.basicConfig(level=logging.INFO)
    logging_interceptor = LoggingInterceptor()
    # Create a gRPC server
    server = grpc.server(futures.ThreadPoolExecutor(max_workers=10), interceptors=[ auth_interceptor, logging_interceptor ])
    # Add the services to the server
    add_CodeGeneratorServiceServicer_to_server(CodeGeneratorService(), server)
    SERVICE_NAMES = (
        code_generator_service_pb2.DESCRIPTOR.services_by_name['CodeGeneratorService'].full_name,
        reflection.SERVICE_NAME,
    )

    grpc_url = os.getenv('HOST_URL_GRPC')
    grpc_port = os.getenv('HOST_PORT_GRPC')
    server.add_insecure_port(f'{grpc_url}:{grpc_port}')
    reflection.enable_server_reflection(SERVICE_NAMES, server)

    logger.info(f'Starting gRPC server on {grpc_url}:{grpc_port}')

    server.start()
    server.wait_for_termination()
