# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data_processing/protos/annotated_text.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from data_processing.protos import interaction_pb2 as data__processing_dot_protos_dot_interaction__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n+data_processing/protos/annotated_text.proto\x12\x0elanguage.tapas\x1a(data_processing/protos/interaction.proto\"\xf2\x01\n\rAnnotatedText\x12/\n\x0b\x61nnotations\x18\x01 \x03(\x0b\x32\x1a.language.tapas.Annotation2[\n\x16\x61nnotated_question_ext\x12\x18.language.tapas.Question\x18\xd6\xd7\x84\x91\x01 \x01(\x0b\x32\x1d.language.tapas.AnnotatedText2S\n\x12\x61nnotated_cell_ext\x12\x14.language.tapas.Cell\x18\xd6\xd7\x84\x91\x01 \x01(\x0b\x32\x1d.language.tapas.AnnotatedText\"\x88\x02\n\x15\x41nnotationDescription\x12M\n\x0c\x64\x65scriptions\x18\x01 \x03(\x0b\x32\x37.language.tapas.AnnotationDescription.DescriptionsEntry\x1a\x33\n\x11\x44\x65scriptionsEntry\x12\x0b\n\x03key\x18\x01 \x01(\t\x12\r\n\x05value\x18\x02 \x01(\t:\x02\x38\x01\x32k\n\x1b\x61nnotation_descriptions_ext\x12\x1b.language.tapas.Interaction\x18\xc3\xd4\xa6\x98\x01 \x01(\x0b\x32%.language.tapas.AnnotationDescription\"R\n\nAnnotation\x12\x18\n\x10\x62\x65gin_byte_index\x18\x01 \x01(\x03\x12\x16\n\x0e\x65nd_byte_index\x18\x02 \x01(\x03\x12\x12\n\nidentifier\x18\x03 \x01(\t')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'data_processing.protos.annotated_text_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  data__processing_dot_protos_dot_interaction__pb2.Question.RegisterExtension(_ANNOTATEDTEXT.extensions_by_name['annotated_question_ext'])
  data__processing_dot_protos_dot_interaction__pb2.Cell.RegisterExtension(_ANNOTATEDTEXT.extensions_by_name['annotated_cell_ext'])
  data__processing_dot_protos_dot_interaction__pb2.Interaction.RegisterExtension(_ANNOTATIONDESCRIPTION.extensions_by_name['annotation_descriptions_ext'])

  DESCRIPTOR._options = None
  _ANNOTATIONDESCRIPTION_DESCRIPTIONSENTRY._options = None
  _ANNOTATIONDESCRIPTION_DESCRIPTIONSENTRY._serialized_options = b'8\001'
  _ANNOTATEDTEXT._serialized_start=106
  _ANNOTATEDTEXT._serialized_end=348
  _ANNOTATIONDESCRIPTION._serialized_start=351
  _ANNOTATIONDESCRIPTION._serialized_end=615
  _ANNOTATIONDESCRIPTION_DESCRIPTIONSENTRY._serialized_start=455
  _ANNOTATIONDESCRIPTION_DESCRIPTIONSENTRY._serialized_end=506
  _ANNOTATION._serialized_start=617
  _ANNOTATION._serialized_end=699
# @@protoc_insertion_point(module_scope)
