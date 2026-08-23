# WebUI asset downloading and C++ embedding for ninfer-serve

option(NINFER_ENABLE_UI "Download and embed the WebUI into ninfer-serve" ON)
set(NINFER_UI_RELEASE_TAG "b10587" CACHE STRING "llama.cpp release tag for prebuilt WebUI")

set(NINFER_UI_GEN_DIR "${CMAKE_BINARY_DIR}/generated/ninfer_ui")
set(NINFER_UI_CPP "${NINFER_UI_GEN_DIR}/ui.cpp")
set(NINFER_UI_H "${NINFER_UI_GEN_DIR}/ui.h")

file(MAKE_DIRECTORY "${NINFER_UI_GEN_DIR}")

# Build the host embedder tool
add_executable(ninfer_ui_embed "${PROJECT_SOURCE_DIR}/tools/ui/embed.cpp")
target_compile_features(ninfer_ui_embed PRIVATE cxx_std_17)

set(UI_ASSET_DIR "")

if(NINFER_ENABLE_UI)
  set(LOCAL_SRC_DIST "${PROJECT_SOURCE_DIR}/tools/ui/dist")
  if(EXISTS "${LOCAL_SRC_DIST}/index.html")
    message(STATUS "NInfer UI: using local assets from ${LOCAL_SRC_DIST}")
    set(UI_ASSET_DIR "${LOCAL_SRC_DIST}")
  else()
    set(UI_DOWNLOAD_DIR "${CMAKE_BINARY_DIR}/_deps/ui_dist")
    set(UI_ARCHIVE "${CMAKE_BINARY_DIR}/_deps/llama-${NINFER_UI_RELEASE_TAG}-ui.tar.gz")
    set(UI_STAMP "${UI_DOWNLOAD_DIR}/.extracted_${NINFER_UI_RELEASE_TAG}")

    if(NOT EXISTS "${UI_STAMP}" OR NOT EXISTS "${UI_DOWNLOAD_DIR}/index.html")
      file(MAKE_DIRECTORY "${CMAKE_BINARY_DIR}/_deps")
      set(UI_URL "https://github.com/ggml-org/llama.cpp/releases/download/${NINFER_UI_RELEASE_TAG}/llama-${NINFER_UI_RELEASE_TAG}-ui.tar.gz")
      message(STATUS "NInfer UI: downloading WebUI release ${NINFER_UI_RELEASE_TAG} from ${UI_URL}")
      
      file(DOWNLOAD "${UI_URL}" "${UI_ARCHIVE}"
        STATUS download_status
        TIMEOUT 60
        TLS_VERIFY ON
      )
      list(GET download_status 0 dl_rc)
      if(NOT dl_rc EQUAL 0)
        list(GET download_status 1 dl_err)
        message(WARNING "NInfer UI: download failed (${dl_err}). Building with UI stubs.")
      else()
        file(REMOVE_RECURSE "${UI_DOWNLOAD_DIR}")
        file(MAKE_DIRECTORY "${UI_DOWNLOAD_DIR}")
        message(STATUS "NInfer UI: extracting ${UI_ARCHIVE}")
        file(ARCHIVE_EXTRACT INPUT "${UI_ARCHIVE}" DESTINATION "${UI_DOWNLOAD_DIR}")
        file(WRITE "${UI_STAMP}" "${NINFER_UI_RELEASE_TAG}")
      endif()
    endif()

    if(EXISTS "${UI_DOWNLOAD_DIR}")
      set(UI_ASSET_DIR "${UI_DOWNLOAD_DIR}")
    endif()
  endif()
endif()

# Custom command to generate ui.cpp and ui.h
if(NOT "${UI_ASSET_DIR}" STREQUAL "")
  add_custom_command(
    OUTPUT "${NINFER_UI_CPP}" "${NINFER_UI_H}"
    COMMAND ninfer_ui_embed "${NINFER_UI_CPP}" "${NINFER_UI_H}" "${UI_ASSET_DIR}"
    DEPENDS ninfer_ui_embed
    COMMENT "Generating ninfer-serve WebUI embedded asset source"
    VERBATIM
  )
else()
  add_custom_command(
    OUTPUT "${NINFER_UI_CPP}" "${NINFER_UI_H}"
    COMMAND ninfer_ui_embed "${NINFER_UI_CPP}" "${NINFER_UI_H}"
    DEPENDS ninfer_ui_embed
    COMMENT "Generating ninfer-serve empty WebUI stub source"
    VERBATIM
  )
endif()

add_custom_target(ninfer_ui_assets DEPENDS "${NINFER_UI_CPP}" "${NINFER_UI_H}")

set_source_files_properties("${NINFER_UI_CPP}" "${NINFER_UI_H}" PROPERTIES GENERATED TRUE)
