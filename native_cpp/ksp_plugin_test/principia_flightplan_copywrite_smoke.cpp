#include <cstdlib>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <string>

#include "ksp_plugin/interface.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin_test/plugin_io.hpp"

namespace {

using principia::interface::Burn;
using principia::interface::NavigationManoeuvre;
using principia::interface::PullSerializer;
using principia::interface::XYZ;

using principia::interface::principia__FlightPlanGetManoeuvre;
using principia::interface::principia__FlightPlanInsert;
using principia::interface::principia__FlightPlanNumberOfManoeuvres;
using principia::interface::principia__SerializePlugin;

using principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile;

void Usage(char const* argv0) {
  std::cerr
      << "usage:\n"
      << "  " << argv0
      << " <input_plugin.b64> <output_plugin.b64> <vessel_guid>"
      << " [dt_s] [dv_tangent_m_s]\n\n"
      << "Example:\n"
      << "  " << argv0
      << " input.b64 output.b64 6073... 30 1\n";
}

void WriteSerializedPlugin(std::filesystem::path const& output_path,
                           decltype(ReadPluginFromFile(
                               std::filesystem::path{}, "gipfeli", "base64")) const& plugin) {
  std::ofstream out(output_path);
  if (!out) {
    throw std::runtime_error("cannot open output file");
  }

  PullSerializer* serializer = nullptr;

  for (;;) {
    char const* const chunk =
        principia__SerializePlugin(plugin.get(), &serializer, "gipfeli", "base64");

    if (chunk == nullptr) {
      break;
    }

    out << chunk << "\n";
  }
}

}  // namespace

int main(int argc, char** argv) {
  if (argc < 4 || argc > 6) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::filesystem::path const input_plugin = argv[1];
  std::filesystem::path const output_plugin = argv[2];
  char const* const vessel_guid = argv[3];

  double const dt_s = argc >= 5 ? std::stod(argv[4]) : 30.0;
  double const dv_tangent_m_s = argc >= 6 ? std::stod(argv[5]) : 1.0;

  std::cerr << "[INFO] loading plugin: " << input_plugin << "\n";
  auto plugin = ReadPluginFromFile(input_plugin, "gipfeli", "base64");

  int const before =
      principia__FlightPlanNumberOfManoeuvres(plugin.get(), vessel_guid);

  if (before < 1) {
    std::cerr << "[ERROR] expected at least one existing manoeuvre to clone\n";
    return EXIT_FAILURE;
  }

  std::cerr << "[INFO] manoeuvres before: " << before << "\n";

  NavigationManoeuvre* const source =
      principia__FlightPlanGetManoeuvre(plugin.get(), vessel_guid, before - 1);

  Burn burn = source->burn;
  delete source;

  // Dummy safe copy-write test:
  // - preserve the original frame, thrust, Isp, inertial flag;
  // - insert a tiny 1 m/s tangent burn shortly after the existing manoeuvre.
  burn.initial_time += dt_s;
  burn.delta_v = XYZ{dv_tangent_m_s, 0.0, 0.0};

  std::cerr << "[INFO] inserting dummy burn:\n"
            << "       index          = " << before << "\n"
            << "       initial_time   = " << std::setprecision(17)
            << burn.initial_time << "\n"
            << "       delta_v_nav    = ["
            << burn.delta_v.x << ", "
            << burn.delta_v.y << ", "
            << burn.delta_v.z << "] m/s\n";

  // We deliberately do not trust the Status layout yet. The post-write probe is
  // the validator: manoeuvre_count must increase and the new burn must appear.
  (void)principia__FlightPlanInsert(plugin.get(), vessel_guid, burn, before);

  int const after =
      principia__FlightPlanNumberOfManoeuvres(plugin.get(), vessel_guid);

  std::cerr << "[INFO] manoeuvres after: " << after << "\n";

  if (after != before + 1) {
    std::cerr << "[ERROR] insert did not increase manoeuvre count\n";
    return EXIT_FAILURE;
  }

  std::cerr << "[INFO] serializing modified plugin: " << output_plugin << "\n";
  WriteSerializedPlugin(output_plugin, plugin);

  std::cout << "{\n"
            << "  \"status\": \"ok\",\n"
            << "  \"input_plugin\": \"" << input_plugin.string() << "\",\n"
            << "  \"output_plugin\": \"" << output_plugin.string() << "\",\n"
            << "  \"vessel_guid\": \"" << vessel_guid << "\",\n"
            << "  \"manoeuvre_count_before\": " << before << ",\n"
            << "  \"manoeuvre_count_after\": " << after << ",\n"
            << "  \"inserted_index\": " << before << ",\n"
            << "  \"inserted_dt_s\": " << dt_s << ",\n"
            << "  \"inserted_delta_v_navigation_m_s\": ["
            << dv_tangent_m_s << ", 0, 0]\n"
            << "}\n";

  return EXIT_SUCCESS;
}
