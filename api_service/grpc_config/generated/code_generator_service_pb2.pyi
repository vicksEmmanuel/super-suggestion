from google.protobuf.internal import enum_type_wrapper as _enum_type_wrapper
from google.protobuf import descriptor as _descriptor
from google.protobuf import message as _message
from typing import ClassVar as _ClassVar, Mapping as _Mapping, Optional as _Optional, Union as _Union

DESCRIPTOR: _descriptor.FileDescriptor

class ErrorCode(int, metaclass=_enum_type_wrapper.EnumTypeWrapper):
    __slots__ = ()
    UNKNOWN_ERROR: _ClassVar[ErrorCode]
    INVALID_REQUEST: _ClassVar[ErrorCode]
    MODEL_ERROR: _ClassVar[ErrorCode]
UNKNOWN_ERROR: ErrorCode
INVALID_REQUEST: ErrorCode
MODEL_ERROR: ErrorCode

class GenerateCodeRequest(_message.Message):
    __slots__ = ("prompt", "suffix_prompt")
    PROMPT_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_PROMPT_FIELD_NUMBER: _ClassVar[int]
    prompt: str
    suffix_prompt: str
    def __init__(self, prompt: _Optional[str] = ..., suffix_prompt: _Optional[str] = ...) -> None: ...

class GenerateCodeResponse(_message.Message):
    __slots__ = ("generated_code", "error")
    GENERATED_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_FIELD_NUMBER: _ClassVar[int]
    generated_code: GeneratedCode
    error: ErrorResponse
    def __init__(self, generated_code: _Optional[_Union[GeneratedCode, _Mapping]] = ..., error: _Optional[_Union[ErrorResponse, _Mapping]] = ...) -> None: ...

class GeneratedCode(_message.Message):
    __slots__ = ("code", "prefix", "infill", "suffix")
    CODE_FIELD_NUMBER: _ClassVar[int]
    PREFIX_FIELD_NUMBER: _ClassVar[int]
    INFILL_FIELD_NUMBER: _ClassVar[int]
    SUFFIX_FIELD_NUMBER: _ClassVar[int]
    code: str
    prefix: str
    infill: str
    suffix: str
    def __init__(self, code: _Optional[str] = ..., prefix: _Optional[str] = ..., infill: _Optional[str] = ..., suffix: _Optional[str] = ...) -> None: ...

class ErrorResponse(_message.Message):
    __slots__ = ("error_code", "error_message")
    ERROR_CODE_FIELD_NUMBER: _ClassVar[int]
    ERROR_MESSAGE_FIELD_NUMBER: _ClassVar[int]
    error_code: ErrorCode
    error_message: str
    def __init__(self, error_code: _Optional[_Union[ErrorCode, str]] = ..., error_message: _Optional[str] = ...) -> None: ...
