#include <cmath>
#include <cstdlib>
#include <filesystem>
#include <iomanip>
#include <iostream>
#include <string>

#include "ksp_plugin/interface.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin_test/plugin_io.hpp"

namespace {

using principia::interface::NavigationManoeuvre;
using principia::interface::NavigationManoeuvreFrenetTrihedron;
using principia::interface::XYZ;

using principia::interface::principia__FlightPlanCount;
using principia::interface::principia__FlightPlanExists;
using principia::interface::principia__FlightPlanGetActualFinalTime;
using principia::interface::principia__FlightPlanGetDesiredFinalTime;
using principia::interface::principia__FlightPlanGetInitialTime;
using principia::interface::principia__FlightPlanGetManoeuvre;
using principia::interface::principia__FlightPlanGetManoeuvreFrenetTrihedron;
using principia::interface::principia__FlightPlanGetManoeuvreInitialPlottedVelocity;
using principia::interface::principia__FlightPlanNumberOfAnomalousManoeuvres;
using principia::interface::principia__FlightPlanNumberOfManoeuvres;
using principia::interface::principia__FlightPlanNumberOfSegments;
using principia::interface::principia__FlightPlanSelected;

using principia::interface::principia__VesselBinormal;
using principia::interface::principia__VesselNormal;
using principia::interface::principia__VesselTangent;
using principia::interface::principia__VesselVelocity;

using principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile;

void Usage(char const* argv0) {
  std::cerr
      << "usage:\n"
      << "  " << argv0 << " <principia_serialized_plugin.b64> <vessel_guid>\n";
}

double Norm(XYZ const& v) {
  return std::sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
}

void PrintXYZInline(XYZ const& v) {
  std::cout << "["
            << std::setprecision(17)
            << v.x << ", " << v.y << ", " << v.z << "]";
}

void PrintXYZField(char const* name, XYZ const& v, bool comma = true) {
  std::cout << "    \"" << name << "\": ";
  PrintXYZInline(v);
  if (comma) {
    std::cout << ",";
  }
  std::cout << "\n";
}

void PrintXYZField6(char const* name, XYZ const& v, bool comma = true) {
  std::cout << "      \"" << name << "\": ";
  PrintXYZInline(v);
  if (comma) {
    std::cout << ",";
  }
  std::cout << "\n";
}

}  // namespace

int main(int argc, char** argv) {
  if (argc != 3) {
    Usage(argv[0]);
    return EXIT_FAILURE;
  }

  std::filesystem::path const plugin_b64 = argv[1];
  char const* const vessel_guid = argv[2];

  std::cerr << "[INFO] loading plugin: " << plugin_b64 << "\n";

  auto plugin = ReadPluginFromFile(plugin_b64, "gipfeli", "base64");

  std::cout << std::setprecision(17);
  std::cout << "{\n";
  std::cout << "  \"status\": \"ok\",\n";
  std::cout << "  \"vessel_guid\": \"" << vessel_guid << "\",\n";

  bool const has_flight_plan =
      principia__FlightPlanExists(plugin.get(), vessel_guid);

  std::cout << "  \"has_flight_plan\": "
            << (has_flight_plan ? "true" : "false") << ",\n";

  int flight_plan_count = 0;
  int selected_flight_plan = -1;
  int manoeuvre_count = 0;
  int anomalous_manoeuvre_count = 0;
  int segment_count = 0;

  if (has_flight_plan) {
    flight_plan_count =
        principia__FlightPlanCount(plugin.get(), vessel_guid);
    selected_flight_plan =
        principia__FlightPlanSelected(plugin.get(), vessel_guid);
    manoeuvre_count =
        principia__FlightPlanNumberOfManoeuvres(plugin.get(), vessel_guid);
    anomalous_manoeuvre_count =
        principia__FlightPlanNumberOfAnomalousManoeuvres(plugin.get(), vessel_guid);
    segment_count =
        principia__FlightPlanNumberOfSegments(plugin.get(), vessel_guid);

    std::cout << "  \"flight_plan_count\": " << flight_plan_count << ",\n";
    std::cout << "  \"selected_flight_plan\": " << selected_flight_plan << ",\n";
    std::cout << "  \"flight_plan\": {\n";
    std::cout << "    \"initial_time\": "
              << principia__FlightPlanGetInitialTime(plugin.get(), vessel_guid)
              << ",\n";
    std::cout << "    \"desired_final_time\": "
              << principia__FlightPlanGetDesiredFinalTime(plugin.get(), vessel_guid)
              << ",\n";
    std::cout << "    \"actual_final_time\": "
              << principia__FlightPlanGetActualFinalTime(plugin.get(), vessel_guid)
              << "\n";
    std::cout << "  },\n";
  } else {
    std::cout << "  \"flight_plan_count\": 0,\n";
    std::cout << "  \"selected_flight_plan\": -1,\n";
    std::cout << "  \"flight_plan\": null,\n";
  }

  std::cout << "  \"manoeuvre_count\": " << manoeuvre_count << ",\n";
  std::cout << "  \"anomalous_manoeuvre_count\": "
            << anomalous_manoeuvre_count << ",\n";
  std::cout << "  \"segment_count\": " << segment_count << ",\n";

  std::cout << "  \"manoeuvres\": [\n";
  for (int i = 0; i < manoeuvre_count; ++i) {
    NavigationManoeuvre* const manoeuvre =
        principia__FlightPlanGetManoeuvre(plugin.get(), vessel_guid, i);

    NavigationManoeuvreFrenetTrihedron const trihedron =
        principia__FlightPlanGetManoeuvreFrenetTrihedron(
            plugin.get(), vessel_guid, i);

    XYZ const initial_plotted_velocity =
        principia__FlightPlanGetManoeuvreInitialPlottedVelocity(
            plugin.get(), vessel_guid, i);

    XYZ const delta_v = manoeuvre->burn.delta_v;

    std::cout << "    {\n";
    std::cout << "      \"index\": " << i << ",\n";
    std::cout << "      \"initial_time\": "
              << manoeuvre->burn.initial_time << ",\n";
    std::cout << "      \"final_time\": "
              << manoeuvre->final_time << ",\n";
    std::cout << "      \"duration\": "
              << manoeuvre->duration << ",\n";
    std::cout << "      \"time_of_half_delta_v\": "
              << manoeuvre->time_of_half_delta_v << ",\n";
    std::cout << "      \"time_to_half_delta_v\": "
              << manoeuvre->time_to_half_delta_v << ",\n";

    std::cout << "      \"thrust_kN\": "
              << manoeuvre->burn.thrust_in_kilonewtons << ",\n";
    std::cout << "      \"specific_impulse_s_g0\": "
              << manoeuvre->burn.specific_impulse_in_seconds_g0 << ",\n";
    std::cout << "      \"is_inertially_fixed\": "
              << (manoeuvre->burn.is_inertially_fixed ? "true" : "false")
              << ",\n";

    std::cout << "      \"initial_mass_tonnes\": "
              << manoeuvre->initial_mass_in_tonnes << ",\n";
    std::cout << "      \"final_mass_tonnes\": "
              << manoeuvre->final_mass_in_tonnes << ",\n";
    std::cout << "      \"mass_flow\": "
              << manoeuvre->mass_flow << ",\n";

    std::cout << "      \"delta_v_navigation_m_s\": ";
    PrintXYZInline(delta_v);
    std::cout << ",\n";
    std::cout << "      \"delta_v_norm_m_s\": " << Norm(delta_v) << ",\n";

    PrintXYZField6("initial_plotted_velocity", initial_plotted_velocity);

    std::cout << "      \"frenet_trihedron\": {\n";
    PrintXYZField6("tangent", trihedron.tangent);
    PrintXYZField6("normal", trihedron.normal);
    PrintXYZField6("binormal", trihedron.binormal, false);
    std::cout << "      }\n";

    std::cout << "    }";
    if (i + 1 < manoeuvre_count) {
      std::cout << ",";
    }
    std::cout << "\n";

    delete manoeuvre;
  }
  std::cout << "  ],\n";

  XYZ const velocity = principia__VesselVelocity(plugin.get(), vessel_guid);
  XYZ const tangent = principia__VesselTangent(plugin.get(), vessel_guid);
  XYZ const normal = principia__VesselNormal(plugin.get(), vessel_guid);
  XYZ const binormal = principia__VesselBinormal(plugin.get(), vessel_guid);

  std::cout << "  \"vessel_frame\": {\n";
  PrintXYZField("velocity", velocity);
  PrintXYZField("tangent", tangent);
  PrintXYZField("normal", normal);
  PrintXYZField("binormal", binormal, false);
  std::cout << "  }\n";

  std::cout << "}\n";
  return EXIT_SUCCESS;
}
