# Sources directly in this directory are built in every configuration. The
# llama.cpp implementation lives in llama/ and the stubs in stub/; exactly one
# of the two is added, so neither build contains an empty translation unit.
if(PRAG_WITH_LLAMA)
  prag_add_module(llm DEPENDS prag_core llama)
  target_compile_definitions(prag_llm PUBLIC PRAG_HAVE_LLAMA=1)
  target_compile_definitions(prag_llm PRIVATE PRAG_LLAMA_TAG="${PRAG_LLAMA_TAG}")
  file(GLOB prag_llm_backend CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/llama/*.cpp")
else()
  prag_add_module(llm DEPENDS prag_core)
  file(GLOB prag_llm_backend CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/stub/*.cpp")
endif()

target_sources(prag_llm PRIVATE ${prag_llm_backend})
target_include_directories(prag_llm PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")

# The slow tests write models of up to 2 GiB; they go into the build tree.
# Without PRAG_SLOW_TESTS=1 in the environment every slow case skips.
if(TARGET test_llm_slow)
  target_compile_definitions(test_llm_slow PRIVATE PRAG_LLM_TEST_DIR="${CMAKE_CURRENT_BINARY_DIR}/work")
  set_tests_properties(test_llm_slow PROPERTIES TIMEOUT 3600)
endif()
