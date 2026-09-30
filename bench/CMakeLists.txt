# Experiment drivers (docs/EXPERIMENTS.md): one executable, one sub-command per experiment.
file(GLOB prag_bench_sources CONFIGURE_DEPENDS "${CMAKE_CURRENT_SOURCE_DIR}/*.cpp")
add_executable(prag-bench ${prag_bench_sources})
target_link_libraries(prag-bench PRIVATE prag_engine)
target_include_directories(prag-bench PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}")
target_compile_features(prag-bench PRIVATE cxx_std_20)
target_compile_definitions(prag-bench PRIVATE
  PRAG_SOURCE_DIR="${CMAKE_SOURCE_DIR}"
  PRAG_DEMO_REPO="${PRAG_DEMO_REPO}")
prag_apply_warnings(prag-bench)
if(WIN32)
  target_link_libraries(prag-bench PRIVATE psapi)
endif()
