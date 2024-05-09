
from api_service.controllers.CodeGeneratorController import CodeGeneratorController
from api_service.grpc.generated.code_generator_service_pb2_grpc import CodeGeneratorServiceServicer


class CodeGeneratorService(CodeGeneratorServiceServicer):
    def __init__(self):
        self.controller = CodeGeneratorController()

    def GenerateCode(self, request, context):
        return self.controller.generate_code(request)
    