#include <cstdlib>
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
#include "geometry/space.hpp"      // CORREÇÃO: space
#include "geometry/grassmann.hpp"  // CORREÇÃO: grassmann para Displacement
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin/frames.hpp"   // CORREÇÃO: frames
#include "ksp_plugin_test/plugin_io.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "quantities/quantities.hpp"

namespace principia_impulsive_particle_server {
namespace {

using principia::base::_not_null::not_null;
using principia::geometry::_instant::Instant;
using principia::geometry::_space::Position;      // CORREÇÃO
using principia::geometry::_space::Velocity;      // CORREÇÃO
using principia::geometry::_space::Displacement;  // CORREÇÃO
using principia::physics::_degrees_of_freedom::DegreesOfFreedom;
using principia::physics::_discrete_trajectory::DiscreteTrajectory;
using principia::physics::_ephemeris::Ephemeris;
using principia::quantities::_si::Metre;
using principia::quantities::_si::Second;

using principia::ksp_plugin::_frames::Barycentric; // CORREÇÃO
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
  std::string message;

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
};

std::vector<std::string> SplitTSV(std::string const& line) {
  std::vector<std::string> fields;
  std::string field;
  std::stringstream ss(line);
  while (std::getline(ss, field, '\t')) {
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

Candidate ParsePropRequest(std::vector<std::string> const& f) {
  if (f.size() != 14) {
    throw std::runtime_error("PROP expects 14 TSV fields including command; got " +
                             std::to_string(f.size()));
  }
  Candidate c;
  c.id = f[1];
  c.t0_s = ParseDouble(f[2], "t0_s");
  c.burn_t_s = ParseDouble(f[3], "burn_t_s");
  c.t1_s = ParseDouble(f[4], "t1_s");
  c.x_m = ParseDouble(f[5], "x_m");
  c.y_m = ParseDouble(f[6], "y_m");
  c.z_m = ParseDouble(f[7], "z_m");
  c.vx_m_s = ParseDouble(f[8], "vx_m_s");
  c.vy_m_s = ParseDouble(f[9], "vy_m_s");
  c.vz_m_s = ParseDouble(f[10], "vz_m_s");
  c.burn_dvx_m_s = ParseDouble(f[11], "burn_dvx_m_s");
  c.burn_dvy_m_s = ParseDouble(f[12], "burn_dvy_m_s");
  c.burn_dvz_m_s = ParseDouble(f[13], "burn_dvz_m_s");
  return c;
}

// CORREÇÃO: Assinatura para extrair o plugin const com gipfeli e base64
not_null<std::unique_ptr<Plugin const>> ReadPluginOrDie(std::string const& plugin_b64_path) {
  std::int64_t bytes_processed = 0;
  return principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile(
      plugin_b64_path, "gipfeli", "base64", bytes_processed);
}

// CORREÇÃO: Usar Displacement + origin
Position<Barycentric> MakePosition(double const x_m, double const y_m, double const z_m) {
  Displacement<Barycentric> const d({x_m * Metre, y_m * Metre, z_m * Metre});
  return Barycentric::origin + d;
}

Velocity<Barycentric> MakeVelocity(double const vx_m_s, double const vy_m_s, double const vz_m_s) {
  return Velocity<Barycentric>({vx_m_s * Metre / Second,
                                vy_m_s * Metre / Second,
                                vz_m_s * Metre / Second});
}

// CORREÇÃO: Plugin const*
Result PropagateOne(Plugin const* const plugin, Candidate const& c) {
  Result r;
  r.id = c.id; r.t0_s = c.t0_s; r.burn_t_s = c.burn_t_s; r.t1_s = c.t1_s;

  try {
    if (!(c.t0_s <= c.burn_t_s && c.burn_t_s <= c.t1_s)) {
      throw std::runtime_error("expected t0_s <= burn_t_s <= t1_s");
    }

    // CORREÇÃO: const_cast para o ephemeris_for_testing
    Ephemeris<Barycentric>* ephemeris = const_cast<Ephemeris<Barycentric>*>(plugin->ephemeris_for_testing());

    Instant const t0 = Instant() + c.t0_s * Second;
    Instant const tb = Instant() + c.burn_t_s * Second;
    Instant const t1 = Instant() + c.t1_s * Second;

    auto incoming = std::make_unique<DiscreteTrajectory<Barycentric>>();
    incoming->Append(t0, DegreesOfFreedom<Barycentric>(
            MakePosition(c.x_m, c.y_m, c.z_m),
            MakeVelocity(c.vx_m_s, c.vy_m_s, c.vz_m_s))).IgnoreError(); // CORREÇÃO: IgnoreError

    // CORREÇÃO: Integração nativa FlowWithAdaptiveStep
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
    
    // CORREÇÃO: subtrair origem
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
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}

void WriteOK(Result const& r) {
  std::cout << std::setprecision(17)
            << "OK\t" << r.id << '\t' << r.t0_s << '\t' << r.burn_t_s << '\t' << r.t1_s << '\t'
            << r.burn_x_m << '\t' << r.burn_y_m << '\t' << r.burn_z_m << '\t'
            << r.burn_vx_before_m_s << '\t' << r.burn_vy_before_m_s << '\t' << r.burn_vz_before_m_s << '\t'
            << r.burn_vx_after_m_s << '\t' << r.burn_vy_after_m_s << '\t' << r.burn_vz_after_m_s << '\t'
            << r.final_x_m << '\t' << r.final_y_m << '\t' << r.final_z_m << '\t'
            << r.final_vx_m_s << '\t' << r.final_vy_m_s << '\t' << r.final_vz_m_s << '\n';
  std::cout.flush(); // Crucial para não travar o Python!
}

void WriteERR(std::string const& id, std::string const& message) {
  std::cout << "ERR\t" << id << '\t' << message << '\n';
  std::cout.flush(); // Crucial
}

int Main(int argc, char** argv) {
  if (argc != 2) {
    std::cerr << "usage:\n  " << argv[0] << " plugin_serialized.b64\n";
    return 2;
  }

  std::string const plugin_path = argv[1];
  std::cerr << "[INFO] loading plugin once: " << plugin_path << "\n";
  auto plugin = ReadPluginOrDie(plugin_path);
  std::cerr << "[OK] plugin loaded; server ready\n";

  std::cout << "READY\tprincipia_impulsive_particle_server_v0_1\n";
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    auto fields = SplitTSV(line);
    if (fields.empty()) continue;

    std::string const& cmd = fields[0];
    if (cmd == "PING") {
      std::cout << "PONG\n"; std::cout.flush();
      continue;
    }
    if (cmd == "QUIT") {
      std::cout << "BYE\n"; std::cout.flush();
      break;
    }
    if (cmd == "PROP") {
      std::string id = fields.size() > 1 ? fields[1] : "";
      try {
        Candidate const c = ParsePropRequest(fields);
        Result const r = PropagateOne(plugin.get(), c);
        if (r.status == "ok") {
          WriteOK(r);
        } else {
          WriteERR(r.id, r.message);
        }
      } catch (std::exception const& e) {
        WriteERR(id, e.what());
      }
      continue;
    }

    WriteERR("", "unknown command: " + cmd);
  }

  return 0;
}

}  // namespace
}  // namespace principia_impulsive_particle_server

int main(int argc, char** argv) {
  return principia_impulsive_particle_server::Main(argc, argv);
}
