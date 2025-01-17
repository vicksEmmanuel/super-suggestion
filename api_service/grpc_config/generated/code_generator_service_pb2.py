# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: code_generator_service.proto
# Protobuf Python Version: 5.26.1
"""Generated protocol buffer code."""
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
from google.protobuf.internal import builder as _builder
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()




DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n\x1c\x63ode_generator_service.proto\x12\x0e\x63ode_generator\"<\n\x13GenerateCodeRequest\x12\x0e\n\x06prompt\x18\x01 \x01(\t\x12\x15\n\rsuffix_prompt\x18\x02 \x01(\t\"\x89\x01\n\x14GenerateCodeResponse\x12\x37\n\x0egenerated_code\x18\x01 \x01(\x0b\x32\x1d.code_generator.GeneratedCodeH\x00\x12.\n\x05\x65rror\x18\x02 \x01(\x0b\x32\x1d.code_generator.ErrorResponseH\x00\x42\x08\n\x06result\"M\n\rGeneratedCode\x12\x0c\n\x04\x63ode\x18\x01 \x01(\t\x12\x0e\n\x06prefix\x18\x02 \x01(\t\x12\x0e\n\x06infill\x18\x03 \x01(\t\x12\x0e\n\x06suffix\x18\x04 \x01(\t\"U\n\rErrorResponse\x12-\n\nerror_code\x18\x01 \x01(\x0e\x32\x19.code_generator.ErrorCode\x12\x15\n\rerror_message\x18\x02 \x01(\t*D\n\tErrorCode\x12\x11\n\rUNKNOWN_ERROR\x10\x00\x12\x13\n\x0fINVALID_REQUEST\x10\x01\x12\x0f\n\x0bMODEL_ERROR\x10\x02\x32s\n\x14\x43odeGeneratorService\x12[\n\x0cGenerateCode\x12#.code_generator.GenerateCodeRequest\x1a$.code_generator.GenerateCodeResponse\"\x00\x62\x06proto3')

_globals = globals()
_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, _globals)
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'code_generator_service_pb2', _globals)
if not _descriptor._USE_C_DESCRIPTORS:
  DESCRIPTOR._loaded_options = None
  _globals['_ERRORCODE']._serialized_start=416
  _globals['_ERRORCODE']._serialized_end=484
  _globals['_GENERATECODEREQUEST']._serialized_start=48
  _globals['_GENERATECODEREQUEST']._serialized_end=108
  _globals['_GENERATECODERESPONSE']._serialized_start=111
  _globals['_GENERATECODERESPONSE']._serialized_end=248
  _globals['_GENERATEDCODE']._serialized_start=250
  _globals['_GENERATEDCODE']._serialized_end=327
  _globals['_ERRORRESPONSE']._serialized_start=329
  _globals['_ERRORRESPONSE']._serialized_end=414
  _globals['_CODEGENERATORSERVICE']._serialized_start=486
  _globals['_CODEGENERATORSERVICE']._serialized_end=601
# @@protoc_insertion_point(module_scope)
