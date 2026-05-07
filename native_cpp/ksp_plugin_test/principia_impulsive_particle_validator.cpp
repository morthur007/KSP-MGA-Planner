#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

#include "absl/status/status.h"

#include "base/not_null.hpp"
#include "geometry/instant.hpp"
#include "geometry/point.hpp"
#include "geometry/space.hpp"      // CORREÇÃO: space em vez de vector
#include "geometry/grassmann.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin/frames.hpp"   // CORREÇÃO: frames.hpp
#include "ksp_plugin_test/plugin_io.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "quantities/quantities.hpp"

namespace principia_impulsive_particle_validator {
namespace {

using principia::base::_not_null::not_null;
using principia::geometry::_instant::Instant;
using principia::geometry::_space::Position;      // CORREÇÃO: namespace _space
using principia::geometry::_space::Velocity;      // CORREÇÃO: namespace _space
using principia::geometry::_space::Displacement;  // CORREÇÃO: adicionado
using principia::physics::_degrees_of_freedom::DegreesOfFreedom;
using principia::physics::_discrete_trajectory::DiscreteTrajectory;
using principia::physics::_ephemeris::Ephemeris;
using principia::quantities::_si::Metre;
using principia::quantities::_si::Second;

using principia::ksp_plugin::_frames::Barycentric; // CORREÇÃO: namespace _frames
using principia::ksp_plugin::_plugin::Plugin;

struct Candidate {
  std::string id;
  double t0_s;
  double burn_t_s;
  double t1_s;
  double x_m;
  double y_m;
  double z_m;
  double vx_m_s;
  double vy_m_s;
  double vz_m_s;
  double burn_dvx_m_s;
  double burn_dvy_m_s;
  double burn_dvz_m_s;
};

struct Result {
  std::string id;
  std::string status;
  double t0_s = 0.0;
  double burn_t_s = 0.0;
  double t1_s = 0.0;

  double burn_x_m = 0.0;
  double burn_y_m = 0.0;
  double burn_z_m = 0.0;

  double burn_vx_before_m_s = 0.0;
  double burn_vy_before_m_s = 0.0;
  double burn_vz_before_m_s = 0.0;

  double burn_vx_after_m_s = 0.0;
  double burn_vy_after_m_s = 0.0;
  double burn_vz_after_m_s = 0.0;

  double final_x_m = 0.0;
  double final_y_m = 0.0;
  double final_z_m = 0.0;

  double final_vx_m_s = 0.0;
  double final_vy_m_s = 0.0;
  double final_vz_m_s = 0.0;

  std::string message;
};

std::vector<std::string> SplitCsvLine(std::string const& line) {
  std::vector<std::string> fields;
  std::string field;
  std::stringstream ss(line);
  while (std::getline(ss, field, ',')) {
    fields.push_back(field);
  }
  return fields;
}

double ParseDouble(std::string const& s, std::string const& field_name) {
  char* end = nullptr;
  double const value = std::strtod(s.c_str(), &end);
  if (end == s.c_str() || *end != '\0') {
    throw std::runtime_error("bad numeric field " + field_name + ": " + s);
  }
  return value;
}

std::vector<Candidate> ReadCandidates(std::string const& path) {
  std::ifstream file(path);
  if (!file.good()) {
    throw std::runtime_error("cannot open candidate CSV: " + path);
  }

  std::vector<Candidate> candidates;
  std::string line;
  bool first = true;
  while (std::getline(file, line)) {
    if (line.empty()) continue;
    if (first) {
      first = false;
      if (line.find("id,") == 0) continue;
    }

    auto const f = SplitCsvLine(line);
    if (f.size() < 13) {
      throw std::runtime_error("expected 13 fields in input CSV line: " + line);
    }

    Candidate c;
    c.id = f[0];
    c.t0_s = ParseDouble(f[1], "t0_s");
    c.burn_t_s = ParseDouble(f[2], "burn_t_s");
    c.t1_s = ParseDouble(f[3], "t1_s");
    c.x_m = ParseDouble(f[4], "x_m");
    c.y_m = ParseDouble(f[5], "y_m");
    c.z_m = ParseDouble(f[6], "z_m");
    c.vx_m_s = ParseDouble(f[7], "vx_m_s");
    c.vy_m_s = ParseDouble(f[8], "vy_m_s");
    c.vz_m_s = ParseDouble(f[9], "vz_m_s");
    c.burn_dvx_m_s = ParseDouble(f[10], "burn_dvx_m_s");
    c.burn_dvy_m_s = ParseDouble(f[11], "burn_dvy_m_s");
    c.burn_dvz_m_s = ParseDouble(f[12], "burn_dvz_m_s");
    candidates.push_back(c);
  }
  return candidates;
}

void WriteResults(std::string const& path, std::vector<Result> const& results) {
  std::ofstream file(path);
  if (!file.good()) {
    throw std::runtime_error("cannot open output CSV: " + path);
  }

  file << "id,status,t0_s,burn_t_s,t1_s,"
       << "burn_x_m,burn_y_m,burn_z_m,"
       << "burn_vx_before_m_s,burn_vy_before_m_s,burn_vz_before_m_s,"
       << "burn_vx_after_m_s,burn_vy_after_m_s,burn_vz_after_m_s,"
       << "final_x_m,final_y_m,final_z_m,"
       << "final_vx_m_s,final_vy_m_s,final_vz_m_s,"
       << "message\n";

  file << std::setprecision(17);
  for (auto const& r : results) {
    file << r.id << ',' << r.status << ',' << r.t0_s << ',' << r.burn_t_s << ',' << r.t1_s << ','
         << r.burn_x_m << ',' << r.burn_y_m << ',' << r.burn_z_m << ','
         << r.burn_vx_before_m_s << ',' << r.burn_vy_before_m_s << ',' << r.burn_vz_before_m_s << ','
         << r.burn_vx_after_m_s << ',' << r.burn_vy_after_m_s << ',' << r.burn_vz_after_m_s << ','
         << r.final_x_m << ',' << r.final_y_m << ',' << r.final_z_m << ','
         << r.final_vx_m_s << ',' << r.final_vy_m_s << ',' << r.final_vz_m_s << ','
         << '"' << r.message << '"' << '\n';
  }
}

// CORREÇÃO: Assinatura e parâmetros completos
not_null<std::unique_ptr<Plugin const>> ReadPluginOrDie(std::string const& plugin_b64_path) {
  std::int64_t bytes_processed = 0;
  return principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile(
      plugin_b64_path, "gipfeli", "base64", bytes_processed);
}

// CORREÇÃO: Displacement + Origem
Position<Barycentric> MakePosition(double const x_m, double const y_m, double const z_m) {
  Displacement<Barycentric> const d({x_m * Metre, y_m * Metre, z_m * Metre});
  return Barycentric::origin + d;
}

Velocity<Barycentric> MakeVelocity(double const vx_m_s, double const vy_m_s, double const vz_m_s) {
  return Velocity<Barycentric>({vx_m_s * Metre / Second,
                                vy_m_s * Metre / Second,
                                vz_m_s * Metre / Second});
}

// CORREÇÃO: const Plugin
Result PropagateOne(Plugin const* const plugin, Candidate const& c) {
  Result r;
  r.id = c.id; r.t0_s = c.t0_s; r.burn_t_s = c.burn_t_s; r.t1_s = c.t1_s;

  try {
    if (!(c.t0_s <= c.burn_t_s && c.burn_t_s <= c.t1_s)) {
      throw std::runtime_error("expected t0_s <= burn_t_s <= t1_s");
    }

    // CORREÇÃO: const_cast para liberar o motor adaptativo
    Ephemeris<Barycentric>* ephemeris = const_cast<Ephemeris<Barycentric>*>(plugin->ephemeris_for_testing());

    Instant const t0 = Instant() + c.t0_s * Second;
    Instant const tb = Instant() + c.burn_t_s * Second;
    Instant const t1 = Instant() + c.t1_s * Second;

    // --- PRIMEIRO ARCO: t0 -> burn_t ---
    auto incoming = std::make_unique<DiscreteTrajectory<Barycentric>>();
    incoming->Append(t0, DegreesOfFreedom<Barycentric>(
            MakePosition(c.x_m, c.y_m, c.z_m),
            MakeVelocity(c.vx_m_s, c.vy_m_s, c.vz_m_s))).IgnoreError(); // CORREÇÃO: IgnoreError()

    // CORREÇÃO: Bypass no NewInstance, usa o integrador nativo
    absl::Status const status_in = ephemeris->FlowWithAdaptiveStep(
        incoming.get(),
        Ephemeris<Barycentric>::NoIntrinsicAcceleration,
        tb,
        plugin->parameters_for_testing(),
        Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

    if (!status_in.ok()) {
      throw std::runtime_error("incoming FlowWithAdaptiveStep failed: " + std::string(status_in.message()));
    }

    DegreesOfFreedom<Barycentric> const burn_before = incoming->EvaluateDegreesOfFreedom(tb);

    // CORREÇÃO: Subtrair origem antes de ler coordinates
    Displacement<Barycentric> const d_burn = burn_before.position() - Barycentric::origin;
    auto const& v_before = burn_before.velocity();

    r.burn_x_m = d_burn.coordinates()[0] / Metre;
    r.burn_y_m = d_burn.coordinates()[1] / Metre;
    r.burn_z_m = d_burn.coordinates()[2] / Metre;

    r.burn_vx_before_m_s = v_before.coordinates()[0] / (Metre / Second);
    r.burn_vy_before_m_s = v_before.coordinates()[1] / (Metre / Second);
    r.burn_vz_before_m_s = v_before.coordinates()[2] / (Metre / Second);

    r.burn_vx_after_m_s = r.burn_vx_before_m_s + c.burn_dvx_m_s;
    r.burn_vy_after_m_s = r.burn_vy_before_m_s + c.burn_dvy_m_s;
    r.burn_vz_after_m_s = r.burn_vz_before_m_s + c.burn_dvz_m_s;

    // --- SEGUNDO ARCO: burn_t -> t1 ---
    auto outgoing = std::make_unique<DiscreteTrajectory<Barycentric>>();
    outgoing->Append(tb, DegreesOfFreedom<Barycentric>(
            MakePosition(r.burn_x_m, r.burn_y_m, r.burn_z_m),
            MakeVelocity(r.burn_vx_after_m_s, r.burn_vy_after_m_s, r.burn_vz_after_m_s))).IgnoreError();

    absl::Status const status_out = ephemeris->FlowWithAdaptiveStep(
        outgoing.get(),
        Ephemeris<Barycentric>::NoIntrinsicAcceleration,
        t1,
        plugin->parameters_for_testing(),
        Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

    if (!status_out.ok()) {
      throw std::runtime_error("outgoing FlowWithAdaptiveStep failed: " + std::string(status_out.message()));
    }

    DegreesOfFreedom<Barycentric> const final_dof = outgoing->EvaluateDegreesOfFreedom(t1);

    Displacement<Barycentric> const d_final = final_dof.position() - Barycentric::origin;
    auto const& v_final = final_dof.velocity();

    r.final_x_m = d_final.coordinates()[0] / Metre;
    r.final_y_m = d_final.coordinates()[1] / Metre;
    r.final_z_m = d_final.coordinates()[2] / Metre;

    r.final_vx_m_s = v_final.coordinates()[0] / (Metre / Second);
    r.final_vy_m_s = v_final.coordinates()[1] / (Metre / Second);
    r.final_vz_m_s = v_final.coordinates()[2] / (Metre / Second);

    r.status = "ok";
    r.message = "";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}

int Main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage:\n  " << argv[0]
              << " plugin_serialized.b64 impulsive_candidates.csv results.csv\n";
    return 2;
  }

  std::string const plugin_path = argv[1];
  std::string const input_path = argv[2];
  std::string const output_path = argv[3];

  auto plugin = ReadPluginOrDie(plugin_path);
  auto const candidates = ReadCandidates(input_path);

  std::vector<Result> results;
  results.reserve(candidates.size());

  for (std::size_t i = 0; i < candidates.size(); ++i) {
    results.push_back(PropagateOne(plugin.get(), candidates[i]));
  }

  WriteResults(output_path, results);
  return 0;
}

}  // namespace
}  // namespace principia_impulsive_particle_validator

int main(int argc, char** argv) {
  return principia_impulsive_particle_validator::Main(argc, argv);
}
