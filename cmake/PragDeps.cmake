# Third-party sources are cloned once, at a pinned tag, into <source>/.deps/<name>
# and shared by every build tree. Build products stay inside each build tree.
#
#   prag_fetch(<name> <git-url> <tag>)   -> sets PRAG_DEP_<name>_DIR in the caller's scope

find_package(Git REQUIRED)

set(PRAG_DEPS_DIR "${CMAKE_SOURCE_DIR}/.deps" CACHE PATH "Directory holding third-party source checkouts")

function(prag_fetch name url tag)
  set(dir "${PRAG_DEPS_DIR}/${name}")
  set(stamp "${dir}/.prag-tag")
  set(current "")
  if(EXISTS "${stamp}")
    file(READ "${stamp}" current)
    string(STRIP "${current}" current)
  endif()

  if(NOT current STREQUAL tag)
    if(EXISTS "${dir}")
      file(REMOVE_RECURSE "${dir}")
    endif()
    message(STATUS "Fetching ${name} ${tag}")
    execute_process(
      COMMAND "${GIT_EXECUTABLE}" -c advice.detachedHead=false -c core.longpaths=true
              clone --depth 1 --branch "${tag}" --single-branch "${url}" "${dir}"
      RESULT_VARIABLE result
      OUTPUT_QUIET
      ERROR_VARIABLE error)
    if(NOT result EQUAL 0)
      file(REMOVE_RECURSE "${dir}")
      message(FATAL_ERROR "Could not fetch ${name} ${tag} from ${url}:\n${error}")
    endif()
    file(WRITE "${stamp}" "${tag}\n")
  endif()

  set(PRAG_DEP_${name}_DIR "${dir}" PARENT_SCOPE)
endfunction()

# ---- nlohmann/json (header only) --------------------------------------------
prag_fetch(json https://github.com/nlohmann/json.git v3.12.0)
if(NOT TARGET nlohmann_json::nlohmann_json)
  add_library(nlohmann_json INTERFACE)
  add_library(nlohmann_json::nlohmann_json ALIAS nlohmann_json)
  target_include_directories(nlohmann_json SYSTEM INTERFACE "${PRAG_DEP_json_DIR}/single_include")
endif()

# ---- cpp-httplib (header only) ----------------------------------------------
prag_fetch(httplib https://github.com/yhirose/cpp-httplib.git v0.58.0)
if(NOT TARGET httplib::httplib)
  add_library(prag_httplib INTERFACE)
  add_library(httplib::httplib ALIAS prag_httplib)
  target_include_directories(prag_httplib SYSTEM INTERFACE "${PRAG_DEP_httplib_DIR}")
  if(WIN32)
    target_link_libraries(prag_httplib INTERFACE ws2_32 crypt32)
  endif()
endif()

# ---- tree-sitter runtime and grammars ---------------------------------------
prag_fetch(tree_sitter https://github.com/tree-sitter/tree-sitter.git v0.26.13)
prag_fetch(tree_sitter_c https://github.com/tree-sitter/tree-sitter-c.git v0.24.2)
prag_fetch(tree_sitter_cpp https://github.com/tree-sitter/tree-sitter-cpp.git v0.23.4)
prag_fetch(tree_sitter_python https://github.com/tree-sitter/tree-sitter-python.git v0.25.0)
if(NOT TARGET prag_tree_sitter)
  set(ts_sources
    "${PRAG_DEP_tree_sitter_DIR}/lib/src/lib.c"
    "${PRAG_DEP_tree_sitter_c_DIR}/src/parser.c"
    "${PRAG_DEP_tree_sitter_cpp_DIR}/src/parser.c"
    "${PRAG_DEP_tree_sitter_cpp_DIR}/src/scanner.c"
    "${PRAG_DEP_tree_sitter_python_DIR}/src/parser.c"
    "${PRAG_DEP_tree_sitter_python_DIR}/src/scanner.c")
  add_library(prag_tree_sitter STATIC EXCLUDE_FROM_ALL ${ts_sources})
  target_include_directories(prag_tree_sitter SYSTEM PUBLIC "${PRAG_DEP_tree_sitter_DIR}/lib/include")
  target_include_directories(prag_tree_sitter PRIVATE "${PRAG_DEP_tree_sitter_DIR}/lib/src")
  set_target_properties(prag_tree_sitter PROPERTIES C_STANDARD 11 C_STANDARD_REQUIRED ON)
  if(MSVC)
    target_compile_options(prag_tree_sitter PRIVATE /w /utf-8)
  else()
    target_compile_options(prag_tree_sitter PRIVATE -w)
  endif()
endif()

# ---- llama.cpp ---------------------------------------------------------------
if(PRAG_WITH_LLAMA)
  prag_fetch(llama https://github.com/ggml-org/llama.cpp.git ${PRAG_LLAMA_TAG})
  set(LLAMA_BUILD_TESTS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_SERVER OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_TOOLS OFF CACHE BOOL "" FORCE)
  set(LLAMA_BUILD_COMMON OFF CACHE BOOL "" FORCE)
  set(LLAMA_CURL OFF CACHE BOOL "" FORCE)
  set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
  set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
  # llama.cpp is written for C++17; C++20 turns its u8 literals into char8_t.
  function(prag_add_llama)
    set(CMAKE_CXX_STANDARD 17)
    set(CMAKE_CXX_STANDARD_REQUIRED ON)
    add_subdirectory("${PRAG_DEP_llama_DIR}" "${CMAKE_BINARY_DIR}/_deps/llama" EXCLUDE_FROM_ALL)
  endfunction()
  prag_add_llama()
endif()
