set(bench_name taxi_reduced)
set(EXECUTE_TEST_LIBS gtest ArrowQueryRunner ArrowStorage ${MAPD_LIBRARIES} ${Arrow_LIBRARIES} ${CMAKE_DL_LIBS} ${CUDA_LIBRARIES} ${Boost_LIBRARIES} ${ZLIB_LIBRARIES})
add_executable(${bench_name} taxi_reduced_bench.cpp)
target_link_libraries(${bench_name} ${EXECUTE_TEST_LIBS} benchmark)

add_executable(taxi_full taxi_full_bench.cpp)
target_link_options(taxi_full PUBLIC "LINKER:--no-as-needed")
target_link_libraries(taxi_full ${PROFILER_LIBS} ${EXECUTE_TEST_LIBS} benchmark)
