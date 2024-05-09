
from grpc_config.generated.code_generator_service_pb2_grpc import CodeGeneratorServiceServicer
from controllers.CodeGeneratorController import CodeGeneratorController


class CodeGeneratorService(CodeGeneratorServiceServicer):
    def __init__(self):
        self.controller = CodeGeneratorController()

    def GenerateCode(self, request, context):
        return self.controller.generate_code(request)
    