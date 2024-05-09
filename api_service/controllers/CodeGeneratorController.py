
from grpc_config.generated.code_generator_service_pb2 import ErrorCode, ErrorResponse, GenerateCodeRequest, GenerateCodeResponse, GeneratedCode
from model_service.model.using_langchain_memory import GenerateCodeModel

class CodeGeneratorController:
    def generate_code(self, request: GenerateCodeRequest) -> GenerateCodeResponse:
        try:            
            # Check if prompt is provided
            if not request.prompt:
                error_response = ErrorResponse(
                    error_code=ErrorCode.INVALID_REQUEST,
                    error_message="Prompt is required"
                )
                response = GenerateCodeResponse(error=error_response)
                return response
            
            # Check if suffix_prompt is provided, use empty string if not
            suffix_prompt = request.suffix_prompt if request.suffix_prompt else ""
            
            generated_code = GenerateCodeModel(
                prompt=request.prompt,
                suffix_prompt=suffix_prompt
            ).generate_code()

            generated_code = GeneratedCode(
                code=generated_code["code"],
                prefix=generated_code["prefix"],
                infill=generated_code["infill"],
                suffix=generated_code["suffix"]
            )

            response = GenerateCodeResponse(generated_code=generated_code)

        except Exception as e:
            error_response = ErrorResponse(
                error_code=ErrorCode.MODEL_ERROR,
                error_message=str(e)
            )
            response = GenerateCodeResponse(error=error_response)

        return response
