# The compiled froggeric v22.5 chat renderer lives in its own directory so the whole feature can
# be removed (or rebased past an upstream change) without touching the shared frontend sources.
target_sources(ninfer_engine PRIVATE
  ${CMAKE_CURRENT_SOURCE_DIR}/python.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/renderer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tags.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/think.cpp)
