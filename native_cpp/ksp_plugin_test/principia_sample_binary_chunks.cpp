#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin_test/plugin_io.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 8) {
    std::cerr
        << "usage:\n"
        << "  principia_sample_binary_chunks "
        << "<input_serialized_plugin_lines.b64> "
        << "<output_dir> "
        << "<central_body> "
        << "<start_offset_s> "
        << "<duration_s> "
        << "<step_s> "
        << "<chunk_s>\n\n"
        << "example:\n"
        << "  principia_sample_binary_chunks "
        << "data/jnsq_gate0/principia_serialized_plugin.b64 "
        << "data/jnsq_gate0/ephem_chunks_30y_1h "
        << "Sun 0.2 946728000 3600 31557600\n";
    return 2;
  }

  namespace plugin_io = principia::ksp_plugin_test::_plugin_io;

  std::filesystem::path const input(argv[1]);
  std::string const output_dir(argv[2]);
  std::string const central_body(argv[3]);

  double const start_offset_s = std::strtod(argv[4], nullptr);
  double const duration_s = std::strtod(argv[5], nullptr);
  double const step_s = std::strtod(argv[6], nullptr);
  double const chunk_s = std::strtod(argv[7], nullptr);

  std::int64_t bytes_read = 0;

  auto plugin = plugin_io::ReadPluginFromFile(
      input,
      std::string_view("gipfeli"),
      std::string_view("base64"),
      bytes_read);

  std::cerr << "[OK] Plugin deserializado\n";
  std::cerr << "[OK] bytes_read=" << bytes_read << "\n";
  std::cerr << "[INFO] output_dir=" << output_dir << "\n";
  std::cerr << "[INFO] central_body=" << central_body << "\n";
  std::cerr << "[INFO] start_offset_s=" << start_offset_s << "\n";
  std::cerr << "[INFO] duration_s=" << duration_s << "\n";
  std::cerr << "[INFO] step_s=" << step_s << "\n";
  std::cerr << "[INFO] chunk_s=" << chunk_s << "\n";

  plugin->DumpSampledEphemerisToBinaryChunks(
      output_dir,
      central_body,
      start_offset_s,
      duration_s,
      step_s,
      chunk_s);

  std::cerr << "[OK] binary chunks written to " << output_dir << "\n";
  return 0;
}
