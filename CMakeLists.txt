cmake_minimum_required(VERSION 3.18)

# scikit-build-core passes the version from pyproject.toml. The fallback keeps
# the CMake project usable on its own, without the Python build frontend.
if(NOT DEFINED SKBUILD_PROJECT_VERSION)
  set(SKBUILD_PROJECT_VERSION "0.1.0")
endif()

project(isodendron VERSION ${SKBUILD_PROJECT_VERSION} LANGUAGES C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_C_EXTENSIONS OFF)

set(ISODENDRON_SOURCES
  csrc/src/isodendron.c
  csrc/src/arena.c
  csrc/src/counts.c
  csrc/src/blob.c
)

add_library(isodendron_objects OBJECT ${ISODENDRON_SOURCES})
set_target_properties(isodendron_objects PROPERTIES POSITION_INDEPENDENT_CODE ON)

target_include_directories(isodendron_objects
  PUBLIC  ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
  PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/csrc/src
)

target_compile_definitions(isodendron_objects
  PRIVATE ISODENDRON_VERSION_STR="${PROJECT_VERSION}"
)

if(WIN32)
  target_compile_definitions(isodendron_objects PRIVATE ISODENDRON_BUILD_DLL)
endif()

if(MSVC)
  target_compile_options(isodendron_objects PRIVATE /W4)
else()
  target_compile_options(isodendron_objects PRIVATE -Wall -Wextra)
endif()

add_library(isodendron SHARED $<TARGET_OBJECTS:isodendron_objects>)
target_include_directories(isodendron
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
)

if(NOT WIN32)
  find_package(Threads REQUIRED)
  target_link_libraries(isodendron PRIVATE Threads::Threads)
endif()

# scikit-build-core sets the install prefix to the wheel's platlib, so this
# lands in site-packages/isodendron/ next to the Python modules.
install(TARGETS isodendron
  RUNTIME DESTINATION isodendron
  LIBRARY DESTINATION isodendron
)

option(ISODENDRON_BUILD_TESTS "Build the C test suite" ON)

if(ISODENDRON_BUILD_TESTS)
  enable_testing()

  add_executable(test_isodendron
    csrc/tests/test_isodendron.c
    $<TARGET_OBJECTS:isodendron_objects>
  )

  target_include_directories(test_isodendron PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/include
    ${CMAKE_CURRENT_SOURCE_DIR}/csrc/src
  )

  if(WIN32)
    target_compile_definitions(test_isodendron PRIVATE ISODENDRON_BUILD_DLL)
  else()
    target_link_libraries(test_isodendron PRIVATE Threads::Threads)
  endif()

  add_test(NAME c_api COMMAND test_isodendron)
endif()
