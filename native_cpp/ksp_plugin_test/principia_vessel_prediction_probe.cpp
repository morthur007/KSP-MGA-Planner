// ksp_plugin_test/principia_vessel_prediction_probe.cpp
//
// Read-only probe.
// Goal:
//   Compare the canonical Principia vessel state
//   vessel->psychohistory()->back()
// against a Principia-native prediction sampled at t0 + dt.
//
// Usage:
//   ./principia_vessel_prediction_probe \
//     --plugin data/principia/live_probe/principia_serialized_plugin_rocket.b64 \
//     --vessel <VESSEL_GUID> \
//     --dt 600
//
// Notes:
//   - Does not modify or serialize the plugin.
//   - Does not touch FlightPlan.
//   - Uses Vessel::RefreshPrediction(target_time), which follows Principia's
//     own prediction path starting from psychohistory()->back().

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <cmath>

#include "absl/log/check.h"
#include "ksp_plugin/interface.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin/vessel.hpp"
#include "quantities/si.hpp"
#include "serialization/ksp_plugin.pb.h"
#include "ksp_plugin_test/plugin_io.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"

namespace {

using namespace principia::interface;
using namespace principia::ksp_plugin::_plugin;
using namespace principia::ksp_plugin::_vessel;
using namespace principia::quantities::_si;
using namespace principia::ksp_plugin::_frames;
using namespace principia::physics::_discrete_trajectory;
using namespace principia::physics::_ephemeris;

void PrintXYZ(std::ostream& out, XYZ const& xyz) {
  out << "[" << std::setprecision(17)
      << xyz.x << "," << xyz.y << "," << xyz.z << "]";
}

template<typename DegreesOfFreedomLike>
void PrintRelativeToParent(std::ostream& out,
                           Plugin* const plugin,
                           Vessel* const vessel,
                           DegreesOfFreedomLike const& vessel_dof,
                           Instant const& t) {
  auto const parent_body = vessel->parent()->body();
  auto const parent_dof =
      plugin->ephemeris()->trajectory(parent_body)->EvaluateDegreesOfFreedom(t);

  auto const relative_position =
      vessel_dof.position() - parent_dof.position();
  auto const relative_velocity =
      vessel_dof.velocity() - parent_dof.velocity();

  auto const r = relative_position.coordinates();
  auto const v = relative_velocity.coordinates();

  double const rx = r[0] / Metre;
  double const ry = r[1] / Metre;
  double const rz = r[2] / Metre;

  double const vx = v[0] / (Metre / Second);
  double const vy = v[1] / (Metre / Second);
  double const vz = v[2] / (Metre / Second);

  double const distance =
      std::sqrt(rx * rx + ry * ry + rz * rz);
  double const speed =
      std::sqrt(vx * vx + vy * vy + vz * vz);
  double const radial_velocity =
      (rx * vx + ry * vy + rz * vz) / distance;

  out << "{";
  out << "\"r_parent_m\":[" << rx << "," << ry << "," << rz << "],";
  out << "\"v_parent_m_s\":[" << vx << "," << vy << "," << vz << "],";
  out << "\"distance_m\":" << distance << ",";
  out << "\"speed_m_s\":" << speed << ",";
  out << "\"radial_velocity_m_s\":" << radial_velocity;
  out << "}";
}

template<typename PositionLike>
XYZ RawPositionToXYZ(PositionLike const& position) {
  auto const d = position - Barycentric::origin;
  auto const c = d.coordinates();
  return XYZ{
      .x = c[0] / Metre,
      .y = c[1] / Metre,
      .z = c[2] / Metre,
  };
}

template<typename VelocityLike>
XYZ RawVelocityToXYZ(VelocityLike const& velocity) {
  auto const c = velocity.coordinates();
  return XYZ{
      .x = c[0] / (Metre / Second),
      .y = c[1] / (Metre / Second),
      .z = c[2] / (Metre / Second),
  };
}

template<typename DegreesOfFreedomLike>
void PrintDOF(std::ostream& out, DegreesOfFreedomLike const& dof) {
  out << "\"r_raw_m\":";
  PrintXYZ(out, RawPositionToXYZ(dof.position()));
  out << ",\"v_raw_m_s\":";
  PrintXYZ(out, RawVelocityToXYZ(dof.velocity()));
}
struct Args {
  std::string plugin_path;
  std::string vessel_guid;
  double dt_seconds = 0.0;
  std::int64_t max_steps = 10000;
};

Args ParseArgs(int argc, char** argv) {
  Args args;
  for (int i = 1; i < argc; ++i) {
    std::string const key = argv[i];

    auto require_value = [&](char const* name) -> std::string {
      if (i + 1 >= argc) {
        std::cerr << "ERROR: missing value for " << name << "\n";
        std::exit(2);
      }
      return argv[++i];
    };

    if (key == "--plugin") {
      args.plugin_path = require_value("--plugin");
    } else if (key == "--vessel") {
      args.vessel_guid = require_value("--vessel");
    } else if (key == "--dt") {
      args.dt_seconds = std::stod(require_value("--dt"));
    } else if (key == "--max-steps") {
      args.max_steps = std::stoll(require_value("--max-steps"));
    } else if (key == "--help" || key == "-h") {
      std::cout
          << "Usage:\n"
          << "  " << argv[0]
          << " --plugin plugin.b64 --vessel GUID --dt SECONDS"
          << " [--max-steps 10000]\n";
      std::exit(0);
    } else {
      std::cerr << "ERROR: unknown argument: " << key << "\n";
      std::exit(2);
    }
  }

  if (args.plugin_path.empty()) {
    std::cerr << "ERROR: --plugin is required\n";
    std::exit(2);
  }
  if (args.vessel_guid.empty()) {
    std::cerr << "ERROR: --vessel is required\n";
    std::exit(2);
  }
  if (!(args.dt_seconds >= 0.0)) {
    std::cerr << "ERROR: --dt must be non-negative\n";
    std::exit(2);
  }

  return args;
}

}  // namespace

int main(int argc, char** argv) {
  Args const args = ParseArgs(argc, argv);

  std::int64_t bytes_processed = 0;
  auto const plugin_holder =
      principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile(
          args.plugin_path,
          "gipfeli",
          "base64",
          bytes_processed);

  Plugin* const plugin = const_cast<Plugin*>(&*plugin_holder);

  if (plugin == nullptr) {
    std::cerr << "ERROR: ReadPluginFromFile returned nullptr\n";
    return 4;
  }

  if (!plugin->HasVessel(args.vessel_guid)) {
    std::cerr << "ERROR: plugin has no vessel with GUID: "
              << args.vessel_guid << "\n";
    return 5;
  }

  Vessel* const vessel = plugin->GetVessel(args.vessel_guid);

  auto const psychohistory = vessel->psychohistory();
  if (psychohistory == vessel->trajectory().segments().end() ||
      psychohistory->empty()) {
    std::cerr << "ERROR: vessel psychohistory is missing or empty\n";
    return 6;
  }

  auto const& initial_point = psychohistory->back();
  auto const initial_time = initial_point.time;
  auto const target_time = initial_time + args.dt_seconds * Second;
  DiscreteTrajectory<Barycentric> direct_prediction;
  direct_prediction.Append(
      initial_time,
      initial_point.degrees_of_freedom).IgnoreError();

  auto direct_parameters = vessel->prediction_adaptive_step_parameters();
  direct_parameters.set_max_steps(
      std::max(direct_parameters.max_steps(), args.max_steps));

  auto const direct_status =
      plugin->ephemeris()->FlowWithAdaptiveStep(
          &direct_prediction,
          Ephemeris<Barycentric>::NoIntrinsicAcceleration,
          target_time,
          direct_parameters,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

  // Make prediction long enough for the requested dt.  DefaultPredictionParameters
  // uses max_steps=1000; for debugging LKO over 600/1800/3600 s, allow larger.
  auto prediction_parameters = vessel->prediction_adaptive_step_parameters();
  prediction_parameters.set_max_steps(
      std::max(prediction_parameters.max_steps(), args.max_steps));
  vessel->set_prediction_adaptive_step_parameters(prediction_parameters);

  // This calls Principia's native prediction path and then truncates after
  // target_time, using the same origin as RefreshPrediction(): psychohistory back.
  vessel->RefreshPrediction(target_time);

  auto const prediction = vessel->prediction();
  bool const has_prediction =
      prediction != vessel->trajectory().segments().end() &&
      !prediction->empty();

  std::cout << std::setprecision(17);
  std::cout << "{";

  std::cout << "\"status\":\"ok\",";
  std::cout << "\"vessel_guid\":\"" << args.vessel_guid << "\",";
  std::cout << "\"dt_seconds\":" << args.dt_seconds << ",";
  std::cout << "\"max_steps_requested\":" << args.max_steps << ",";

  std::cout << "\"game_epoch_initial_t_seconds\":"
            << (initial_time - plugin->GameEpoch()) / Second << ",";
  std::cout << "\"game_epoch_target_t_seconds\":"
            << (target_time - plugin->GameEpoch()) / Second << ",";

  std::cout << "\"psychohistory_back_from_parent\":";
  PrintRelativeToParent(std::cout,
                        plugin,
                        vessel,
                        initial_point.degrees_of_freedom,
                        initial_time);
  std::cout << ",";
  PrintDOF(std::cout, initial_point.degrees_of_freedom);
  std::cout << ",";

  std::cout << "\"plugin_vessel_velocity_world_m_s\":";
  PrintXYZ(std::cout, ToXYZ(plugin->VesselVelocity(args.vessel_guid)));
  std::cout << ",";

  std::cout << "\"prediction_available\":"
            << (has_prediction ? "true" : "false") << ",";

  if (has_prediction) {
    auto const& prediction_back = prediction->back();

    std::cout << "\"prediction_back_time_seconds\":"
              << (prediction_back.time - plugin->GameEpoch()) / Second << ",";
    std::cout << "\"prediction_back\":{";
    PrintDOF(std::cout, prediction_back.degrees_of_freedom);
    std::cout << "},";

    bool const reached_target = prediction_back.time >= target_time;
    std::cout << "\"prediction_reached_target\":"
              << (reached_target ? "true" : "false") << ",";

    if (reached_target) {
      auto const sampled = prediction->EvaluateDegreesOfFreedom(target_time);
      std::cout << "\"prediction_sample_at_target\":{";
      PrintDOF(std::cout, sampled);
      std::cout << "},";
    }
  }

  std::cout << "\"direct_flow_status\":\""
            << direct_status.ToString() << "\",";

  std::cout << "\"direct_prediction_size\":"
            << direct_prediction.size() << ",";

  std::cout << "\"direct_prediction_back_time_seconds\":"
            << (direct_prediction.back().time - plugin->GameEpoch()) / Second
            << ",";

  std::cout << "\"direct_prediction_back_from_parent\":";
  PrintRelativeToParent(std::cout,
                        plugin,
                        vessel,
                        direct_prediction.back().degrees_of_freedom,
                        direct_prediction.back().time);
  std::cout << ",";
  PrintDOF(std::cout, direct_prediction.back().degrees_of_freedom);
  std::cout << ",";

  bool const direct_reached_target = direct_prediction.back().time >= target_time;
  std::cout << "\"direct_prediction_reached_target\":"
            << (direct_reached_target ? "true" : "false") << ",";

  if (direct_reached_target) {
    auto const direct_sample =
        direct_prediction.EvaluateDegreesOfFreedom(target_time);
    std::cout << "\"direct_prediction_sample_at_target\":{";
    PrintDOF(std::cout, direct_sample);
    std::cout << "},";
  }

  std::cout << "\"note\":\"read_only_vessel_prediction_probe_v1_direct_flow\"";
  std::cout << "}\n";

  return 0;
}
