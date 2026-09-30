prag_add_module(engine DEPENDS prag_core prag_exec prag_graph prag_parse prag_embed prag_index prag_kde prag_retrieve prag_context prag_edit prag_llm prag_tune)

# While the tune module has no implementation (header only), the engine tests
# link a stand-in pipeline (tests/tune_stub.hpp). It switches itself off as
# soon as prag_tune is a real library.
get_target_property(prag_engine_tune_type prag_tune TYPE)
if(PRAG_BUILD_TESTS AND prag_engine_tune_type STREQUAL "INTERFACE_LIBRARY")
  file(GLOB prag_engine_tests CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/tests/test_*.cpp")
  foreach(test_source IN LISTS prag_engine_tests)
    get_filename_component(test_name "${test_source}" NAME_WE)
    target_compile_definitions(${test_name} PRIVATE PRAG_ENGINE_TUNE_STUB=1)
  endforeach()
endif()
