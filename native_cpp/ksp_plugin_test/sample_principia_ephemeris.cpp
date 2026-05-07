#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin_test/plugin_io.hpp"

#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string_view>

int main(int argc, char** argv) {
  if (argc != 7) {
    std::cerr
        << "usage:\n"
        << "  sample_principia_ephemeris "
        << "<input_serialized_plugin_lines.b64> "
        << "<output.csv> "
        << "<central_body> "
        << "<start_offset_s> "
        << "<duration_s> "
        << "<step_s>\n\n"
        << "example:\n"
        << "  sample_principia_ephemeris "
        << "data/jnsq_gate0/principia_serialized_plugin.b64 "
        << "data/jnsq_gate0/principia_ephemeris_30d.csv "
        << "Sun 0 2592000 21600\n";
    return 2;
  }

  namespace plugin_io = principia::ksp_plugin_test::_plugin_io;

  std::filesystem::path const input(argv[1]);
  std::string const output(argv[2]);
  std::string const central_body(argv[3]);

  double const start_offset_s = std::strtod(argv[4], nullptr);
  double const duration_s = std::strtod(argv[5], nullptr);
  double const step_s = std::strtod(argv[6], nullptr);

  std::int64_t bytes_read = 0;

  auto plugin = plugin_io::ReadPluginFromFile(
      input,
      std::string_view("gipfeli"),
      std::string_view("base64"),
      bytes_read);

  std::cerr << "[OK] Plugin deserializado\n";
  std::cerr << "[OK] bytes_read=" << bytes_read << "\n";
  std::cerr << "[INFO] central_body=" << central_body << "\n";
  std::cerr << "[INFO] start_offset_s=" << start_offset_s << "\n";
  std::cerr << "[INFO] duration_s=" << duration_s << "\n";
  std::cerr << "[INFO] step_s=" << step_s << "\n";

  plugin->DumpSampledEphemerisToCsv(
      output,
      central_body,
      start_offset_s,
      duration_s,
      step_s);

  std::cerr << "[OK] ephemeris sampled to " << output << "\n";
  return 0;
}
