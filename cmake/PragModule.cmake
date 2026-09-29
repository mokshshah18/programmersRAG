# Helpers shared by every module under src/.
#
#   prag_add_module(<name> DEPENDS <targets...>)
#     Builds src/<name>/*.cpp into the static library prag_<name>, exporting
#     src/<name>/include. Every file under src/<name>/tests/test_*.cpp becomes a
#     CTest executable linked against the module.

function(prag_apply_warnings target)
  if(MSVC)
    target_compile_options(${target} PRIVATE /W4 /wd4324 /permissive- /utf-8 /EHsc)
    target_compile_definitions(${target} PRIVATE _CRT_SECURE_NO_WARNINGS NOMINMAX WIN32_LEAN_AND_MEAN)
  else()
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endfunction()

function(prag_apply_simd target)
  if(NOT PRAG_NATIVE_SIMD)
    return()
  endif()
  if(MSVC)
    target_compile_options(${target} PUBLIC /arch:AVX2)
  elseif(CMAKE_SYSTEM_PROCESSOR MATCHES "x86_64|AMD64|amd64")
    target_compile_options(${target} PUBLIC -mavx2 -mfma)
  endif()
endfunction()

function(prag_add_module name)
  cmake_parse_arguments(ARG "" "" "DEPENDS" ${ARGN})
  set(dir "${CMAKE_CURRENT_SOURCE_DIR}")

  file(GLOB sources CONFIGURE_DEPENDS "${dir}/*.cpp" "${dir}/*.c")
  if(sources)
    add_library(prag_${name} STATIC ${sources})
    target_include_directories(prag_${name} PUBLIC "${dir}/include")
    target_link_libraries(prag_${name} PUBLIC ${ARG_DEPENDS})
    target_compile_features(prag_${name} PUBLIC cxx_std_20)
    prag_apply_warnings(prag_${name})
    prag_apply_simd(prag_${name})
  else()
    add_library(prag_${name} INTERFACE)
    target_include_directories(prag_${name} INTERFACE "${dir}/include")
    target_link_libraries(prag_${name} INTERFACE ${ARG_DEPENDS})
    target_compile_features(prag_${name} INTERFACE cxx_std_20)
  endif()

  if(PRAG_BUILD_TESTS)
    file(GLOB tests CONFIGURE_DEPENDS "${dir}/tests/test_*.cpp")
    foreach(test_source IN LISTS tests)
      get_filename_component(test_name "${test_source}" NAME_WE)
      add_executable(${test_name} "${test_source}")
      target_link_libraries(${test_name} PRIVATE prag_${name})
      target_include_directories(${test_name} PRIVATE "${CMAKE_SOURCE_DIR}/tests")
      target_compile_definitions(${test_name} PRIVATE
        PRAG_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
        PRAG_DEMO_REPO="${PRAG_DEMO_REPO}")
      prag_apply_warnings(${test_name})
      add_test(NAME ${test_name} COMMAND ${test_name})
      set_tests_properties(${test_name} PROPERTIES TIMEOUT 600 LABELS "${name}")
    endforeach()
  endif()
endfunction()
