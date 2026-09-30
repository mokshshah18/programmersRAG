prag_add_module(server DEPENDS prag_engine httplib::httplib)

# The server test also runs the command line program when it is built.
if(PRAG_BUILD_TESTS AND TARGET test_server AND EXISTS "${CMAKE_SOURCE_DIR}/src/cli/CMakeLists.txt")
  target_compile_definitions(test_server PRIVATE PRAG_CLI_EXE="$<TARGET_FILE:prag>")
  add_dependencies(test_server prag)
endif()
