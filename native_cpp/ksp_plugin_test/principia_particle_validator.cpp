#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
#include <string>
#include <vector>

#include "absl/status/status.h"
#include "absl/status/statusor.h"

#include "base/not_null.hpp"
#include "geometry/frame.hpp"
#include "geometry/instant.hpp"
#include "geometry/point.hpp"
#include "geometry/space.hpp"       // CORREÇÃO: Para Position, Velocity, Displacement
#include "geometry/grassmann.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin/frames.hpp"    // CORREÇÃO: Para Barycentric
#include "ksp_plugin_test/plugin_io.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "quantities/quantities.hpp"
#include "quantities/named_quantities.hpp" // CORREÇÃO: Para Speed
#include "integrators/embedded_explicit_runge_kutta_integrator.hpp"
#include "integrators/methods.hpp"

namespace principia_particle_validator {
namespace {

using principia::base::_not_null::not_null;
using principia::geometry::_frame::Frame;
using principia::geometry::_instant::Instant;
using principia::geometry::_space::Position;      // CORREÇÃO: Namespace exato
using principia::geometry::_space::Velocity;      // CORREÇÃO: Namespace exato
using principia::geometry::_space::Displacement;  // CORREÇÃO: Adicionado
using principia::physics::_degrees_of_freedom::DegreesOfFreedom;
using principia::physics::_discrete_trajectory::DiscreteTrajectory;
using principia::physics::_ephemeris::Ephemeris;
using principia::quantities::_quantities::Length;
using principia::quantities::_named_quantities::Speed; // CORREÇÃO: Namespace exato
using principia::quantities::_quantities::Time;
using principia::quantities::_si::Metre;
using principia::quantities::_si::Second;

using principia::ksp_plugin::_frames::Barycentric; // CORREÇÃO: Namespace exato
using principia::ksp_plugin::_plugin::Plugin;

struct Candidate {
  std::string id;
  double t0_s;
  double t1_s;
  double x_m;
  double y_m;
  double z_m;
  double vx_m_s;
  double vy_m_s;
  double vz_m_s;
};

struct Result {
  std::string id;
  std::string status;
  double t0_s = 0.0;
  double t1_s = 0.0;
  double x_m = 0.0;
  double y_m = 0.0;
  double z_m = 0.0;
  double vx_m_s = 0.0;
  double vy_m_s = 0.0;
  double vz_m_s = 0.0;
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
    if (f.size() < 9) throw std::runtime_error("expected 9 fields");

    Candidate c;
    c.id = f[0];
    c.t0_s = ParseDouble(f[1], "t0_s");
    c.t1_s = ParseDouble(f[2], "t1_s");
    c.x_m = ParseDouble(f[3], "x_m");
    c.y_m = ParseDouble(f[4], "y_m");
    c.z_m = ParseDouble(f[5], "z_m");
    c.vx_m_s = ParseDouble(f[6], "vx_m_s");
    c.vy_m_s = ParseDouble(f[7], "vy_m_s");
    c.vz_m_s = ParseDouble(f[8], "vz_m_s");
    candidates.push_back(c);
  }
  return candidates;
}

void WriteResults(std::string const& path, std::vector<Result> const& results) {
  std::ofstream file(path);
  if (!file.good()) throw std::runtime_error("cannot open output CSV: " + path);

  file << "id,status,t0_s,t1_s,x_m,y_m,z_m,vx_m_s,vy_m_s,vz_m_s,message\n";
  file << std::setprecision(17);
  for (auto const& r : results) {
    file << r.id << ',' << r.status << ',' << r.t0_s << ',' << r.t1_s << ','
         << r.x_m << ',' << r.y_m << ',' << r.z_m << ','
         << r.vx_m_s << ',' << r.vy_m_s << ',' << r.vz_m_s << ','
         << '"' << r.message << '"' << '\n';
  }
}

not_null<std::unique_ptr<Plugin const>> ReadPluginOrDie(std::string const& plugin_b64_path) {
  std::int64_t bytes_processed = 0;
  // CORREÇÃO: Assinatura completa do leitor
  return principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile(
      plugin_b64_path, "gipfeli", "base64", bytes_processed);
}

// 1. Assinatura corrigida (Plugin const* const plugin)
Result PropagateOne(Plugin const* const plugin, Candidate const& c) {
  Result r;
  r.id = c.id; r.t0_s = c.t0_s; r.t1_s = c.t1_s;

  try {
    if (c.t1_s < c.t0_s) throw std::runtime_error("t1_s < t0_s");

    // 2. Chama a efeméride constante
    Ephemeris<Barycentric>* ephemeris = const_cast<Ephemeris<Barycentric>*>(plugin->ephemeris_for_testing());
    
    Instant const t0 = Instant() + c.t0_s * Second;
    Instant const t1 = Instant() + c.t1_s * Second;

    auto trajectory = std::make_unique<DiscreteTrajectory<Barycentric>>();

    Displacement<Barycentric> const d({c.x_m * Metre, c.y_m * Metre, c.z_m * Metre});
    Position<Barycentric> const q = Barycentric::origin + d;

    Velocity<Barycentric> const v({c.vx_m_s * Metre / Second,
                                   c.vy_m_s * Metre / Second,
                                   c.vz_m_s * Metre / Second});

    trajectory->Append(t0, DegreesOfFreedom<Barycentric>(q, v)).IgnoreError();

    // 3. A mágica acontece aqui: usamos os parâmetros nativos do motor!
    absl::Status const status = ephemeris->FlowWithAdaptiveStep(
        trajectory.get(),
        Ephemeris<Barycentric>::NoIntrinsicAcceleration,
        t1,
        plugin->parameters_for_testing(),
        Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

    if (!status.ok()) {
      throw std::runtime_error(std::string("FlowWithAdaptiveStep failed: ") +
                               std::string(status.message()));
    }

    DegreesOfFreedom<Barycentric> const dof = trajectory->EvaluateDegreesOfFreedom(t1);

    auto const& final_q = dof.position();
    auto const& final_v = dof.velocity();

    Displacement<Barycentric> const final_d = final_q - Barycentric::origin;

    r.x_m = (final_d.coordinates()[0] / Metre);
    r.y_m = (final_d.coordinates()[1] / Metre);
    r.z_m = (final_d.coordinates()[2] / Metre);
    r.vx_m_s = (final_v.coordinates()[0] / (Metre / Second));
    r.vy_m_s = (final_v.coordinates()[1] / (Metre / Second));
    r.vz_m_s = (final_v.coordinates()[2] / (Metre / Second));
    
    r.status = "ok"; r.message = "";
  } catch (std::exception const& e) {
    r.status = "error"; r.message = e.what();
  }

  return r;
}

int Main(int argc, char** argv) {
  if (argc != 4) {
    std::cerr << "usage:\n  " << argv[0]
              << " plugin_serialized.b64 candidates.csv results.csv\n";
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
}  // namespace principia_particle_validator

int main(int argc, char** argv) {
  return principia_particle_validator::Main(argc, argv);
}
