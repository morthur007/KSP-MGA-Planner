#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin_test/plugin_io.hpp"

#include <cstdint>
#include <filesystem>
#include <iostream>
#include <string_view>

// Não precisamos de Instant ou Second aqui na main, 
// pois passaremos apenas o 'double' para o plugin.
int main(int argc, char const* argv[]) {
  if (argc < 3) {
      std::cerr << "Uso: dump_current_snapshot input.b64 output.csv [time_offset_s]\n";
      return 1;
  }

  double time_offset_s = 0.0;
  if (argc > 3) {
      // std::stod converte string para double
      time_offset_s = std::stod(argv[3]);
  }

  namespace plugin_io = principia::ksp_plugin_test::_plugin_io;
  std::int64_t bytes_read = 0;

  // Carrega o plugin
  auto plugin = plugin_io::ReadPluginFromFile(
      std::filesystem::path(argv[1]),
      std::string_view("gipfeli"),
      std::string_view("base64"),
      bytes_read);

  std::cerr << "[OK] Plugin deserializado (Bytes: " << bytes_read << ")\n";

  // CHAMA A FUNÇÃO QUE VOCÊ MODIFICOU:
  // Toda a lógica de Instant, Second e Prolong já está lá no plugin.cpp
  plugin->DumpCurrentEphemerisSnapshotToCsv(argv[2], time_offset_s);

  std::cerr << "[OK] Snapshot (T+" << time_offset_s << "s) escrito em " << argv[2] << "\n";
  return 0;
}
