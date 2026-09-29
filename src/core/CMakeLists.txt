find_package(Threads REQUIRED)
prag_add_module(core DEPENDS nlohmann_json::nlohmann_json Threads::Threads)
