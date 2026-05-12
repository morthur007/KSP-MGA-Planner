#include "ksp_plugin_test/plugin_io.hpp"
#include "serialization/ksp_plugin.pb.h"

#include <cstdint>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr
        << "usage: dump_plugin_message "
        << "<input_serialized_plugin_lines.b64> "
        << "<output_plugin.pb> "
        << "<output_plugin.txt>\n";
    return 2;
  }

  namespace plugin_io = principia::ksp_plugin_test::_plugin_io;

  std::int64_t bytes_read = 0;

  auto plugin = plugin_io::ReadPluginFromFile(
      std::filesystem::path(argv[1]),
      std::string_view("gipfeli"),
      std::string_view("base64"),
      bytes_read);

  std::cerr << "[OK] Plugin deserializado\n";
  std::cerr << "[OK] bytes_read=" << bytes_read << "\n";

  principia::serialization::Plugin message;
  plugin->WriteToMessage(&message);

  std::cerr << "[OK] Plugin::WriteToMessage concluído\n";
  std::cerr << "[INFO] message.ByteSizeLong()=" << message.ByteSizeLong() << "\n";
  std::cerr << "[INFO] message.SpaceUsedLong()=" << message.SpaceUsedLong() << "\n";
  std::cerr << "[INFO] has_ephemeris=" << message.has_ephemeris() << "\n";
  std::cerr << "[INFO] ephemeris.ByteSizeLong()="
            << message.ephemeris().ByteSizeLong() << "\n";

  {
    std::ofstream out(argv[2], std::ios::binary);
    if (!message.SerializeToOstream(&out)) {
      std::cerr << "[FAIL] SerializeToOstream falhou\n";
      return 1;
    }
  }

  {
    std::ofstream out(argv[3]);
    out << message.DebugString();
  }

  std::cerr << "[OK] protobuf binário salvo em " << argv[2] << "\n";
  std::cerr << "[OK] DebugString salvo em " << argv[3] << "\n";

  return 0;
}
