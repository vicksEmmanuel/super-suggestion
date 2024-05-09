
from grpc_config.generated.code_generator_service_pb2 import ErrorCode, ErrorResponse, GenerateCodeRequest, GenerateCodeResponse
from model_service.model.using_langchain_memory import GenerateCodeModel

class CodeGeneratorController:
    def generate_code(self, request: GenerateCodeRequest) -> GenerateCodeResponse:
        try:
            generated_code = GenerateCodeModel(
                prompt_file=request.prompt,
                suffix_prompt_file=request.suffix_prompt
            ).generate_code()

            response = GenerateCodeResponse(
                code=generated_code["code"],
                prefix=generated_code["prefix"],
                infill=generated_code["infill"],
                suffix=generated_code["suffix"]
            )
        except Exception as e:
            error_response = ErrorResponse(
                error_code=ErrorCode.MODEL_ERROR,
                error_message=str(e)
            )
            response = GenerateCodeResponse(error=error_response)

        return response
