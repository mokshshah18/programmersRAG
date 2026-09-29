prag_add_module(parse DEPENDS prag_core prag_graph prag_tree_sitter)

# The corpus test parses files through the ordered executor to show that the
# parser is re-entrant. The library itself does not depend on exec.
if(TARGET test_parse_corpus AND TARGET prag_exec)
  target_link_libraries(test_parse_corpus PRIVATE prag_exec)
  target_compile_definitions(test_parse_corpus PRIVATE PRAG_PARSE_TEST_WITH_EXEC=1)
endif()
