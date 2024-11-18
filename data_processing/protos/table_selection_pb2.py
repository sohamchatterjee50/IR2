# -*- coding: utf-8 -*-
# Generated by the protocol buffer compiler.  DO NOT EDIT!
# source: data_processing/protos/table_selection.proto
"""Generated protocol buffer code."""
from google.protobuf.internal import builder as _builder
from google.protobuf import descriptor as _descriptor
from google.protobuf import descriptor_pool as _descriptor_pool
from google.protobuf import symbol_database as _symbol_database
# @@protoc_insertion_point(imports)

_sym_db = _symbol_database.Default()


from data_processing.protos import interaction_pb2 as data__processing_dot_protos_dot_interaction__pb2


DESCRIPTOR = _descriptor_pool.Default().AddSerializedFile(b'\n,data_processing/protos/table_selection.proto\x12\x0elanguage.tapas\x1a(data_processing/protos/interaction.proto\"\xe1\x07\n\x0eTableSelection\x12H\n\x0fselected_tokens\x18\x03 \x03(\x0b\x32/.language.tapas.TableSelection.TokenCoordinates\x12S\n\x16model_prediction_stats\x18\x02 \x01(\x0b\x32\x33.language.tapas.TableSelection.ModelPredictionStats\x12\x37\n\x05\x64\x65\x62ug\x18\x64 \x01(\x0b\x32(.language.tapas.TableSelection.DebugInfo\x1aP\n\x10TokenCoordinates\x12\x11\n\trow_index\x18\x01 \x01(\x05\x12\x14\n\x0c\x63olumn_index\x18\x02 \x01(\x05\x12\x13\n\x0btoken_index\x18\x03 \x01(\x05\x1a\x44\n\x1cModelPredictionStatsPerModel\x12\x10\n\x08model_id\x18\x01 \x01(\t\x12\x12\n\nis_correct\x18\x02 \x01(\x08\x1a\x8c\x01\n\x1dModelPredictionStatsPerColumn\x12\x0e\n\x06\x63olumn\x18\x03 \x01(\x05\x12[\n\x16model_prediction_stats\x18\x02 \x03(\x0b\x32;.language.tapas.TableSelection.ModelPredictionStatsPerModel\x1a\xd2\x01\n\x14ModelPredictionStats\x12]\n\x17\x63olumn_prediction_stats\x18\x01 \x03(\x0b\x32<.language.tapas.TableSelection.ModelPredictionStatsPerColumn\x12[\n\x16model_prediction_stats\x18\x02 \x03(\x0b\x32;.language.tapas.TableSelection.ModelPredictionStatsPerModel\x1a\x9f\x01\n\tDebugInfo\x12@\n\x07\x63olumns\x18\x01 \x03(\x0b\x32/.language.tapas.TableSelection.DebugInfo.Column\x1aP\n\x06\x43olumn\x12\r\n\x05index\x18\x01 \x01(\x05\x12\r\n\x05score\x18\x02 \x01(\x01\x12\x13\n\x0bis_selected\x18\x03 \x01(\x08\x12\x13\n\x0bis_required\x18\x04 \x01(\x08\x32Y\n\x13table_selection_ext\x12\x18.language.tapas.Question\x18\xcf\xab\xe0\x89\x01 \x01(\x0b\x32\x1e.language.tapas.TableSelection')

_builder.BuildMessageAndEnumDescriptors(DESCRIPTOR, globals())
_builder.BuildTopDescriptorsAndMessages(DESCRIPTOR, 'data_processing.protos.table_selection_pb2', globals())
if _descriptor._USE_C_DESCRIPTORS == False:
  data__processing_dot_protos_dot_interaction__pb2.Question.RegisterExtension(_TABLESELECTION.extensions_by_name['table_selection_ext'])

  DESCRIPTOR._options = None
  _TABLESELECTION._serialized_start=107
  _TABLESELECTION._serialized_end=1100
  _TABLESELECTION_TOKENCOORDINATES._serialized_start=341
  _TABLESELECTION_TOKENCOORDINATES._serialized_end=421
  _TABLESELECTION_MODELPREDICTIONSTATSPERMODEL._serialized_start=423
  _TABLESELECTION_MODELPREDICTIONSTATSPERMODEL._serialized_end=491
  _TABLESELECTION_MODELPREDICTIONSTATSPERCOLUMN._serialized_start=494
  _TABLESELECTION_MODELPREDICTIONSTATSPERCOLUMN._serialized_end=634
  _TABLESELECTION_MODELPREDICTIONSTATS._serialized_start=637
  _TABLESELECTION_MODELPREDICTIONSTATS._serialized_end=847
  _TABLESELECTION_DEBUGINFO._serialized_start=850
  _TABLESELECTION_DEBUGINFO._serialized_end=1009
  _TABLESELECTION_DEBUGINFO_COLUMN._serialized_start=929
  _TABLESELECTION_DEBUGINFO_COLUMN._serialized_end=1009
# @@protoc_insertion_point(module_scope)
