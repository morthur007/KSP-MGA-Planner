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
#include "geometry/space.hpp"
#include "geometry/grassmann.hpp"
#include "ksp_plugin/plugin.hpp"
#include "ksp_plugin/frames.hpp"
#include "ksp_plugin_test/plugin_io.hpp"
#include "physics/degrees_of_freedom.hpp"
#include "physics/discrete_trajectory.hpp"
#include "physics/ephemeris.hpp"
#include "quantities/quantities.hpp"

namespace principia_impulsive_particle_server {
namespace {

using principia::base::_not_null::not_null;
using principia::geometry::_instant::Instant;
using principia::geometry::_space::Displacement;
using principia::geometry::_space::Position;
using principia::geometry::_space::Velocity;
using principia::physics::_degrees_of_freedom::DegreesOfFreedom;
using principia::physics::_discrete_trajectory::DiscreteTrajectory;
using principia::physics::_ephemeris::Ephemeris;
using principia::quantities::_si::Metre;
using principia::quantities::_si::Second;

using principia::ksp_plugin::_frames::Barycentric;
using principia::ksp_plugin::_plugin::Plugin;

struct Vec3 {
  double x = 0.0;
  double y = 0.0;
  double z = 0.0;
};

struct Impulse {
  double burn_t_s = 0.0;
  Vec3 dv_m_s;
};

struct PropNRequest {
  std::string id;
  double t0_s = 0.0;
  double t1_s = 0.0;
  Vec3 r0_m;
  Vec3 v0_m_s;
  std::vector<Impulse> impulses;
};

struct BurnSnapshot {
  double burn_t_s = 0.0;
  Vec3 r_m;
  Vec3 v_before_m_s;
  Vec3 v_after_m_s;
};

struct PropNResult {
  std::string id;
  std::string status;
  std::string message;
  double t0_s = 0.0;
  double t1_s = 0.0;
  std::vector<BurnSnapshot> burns;
  Vec3 final_r_m;
  Vec3 final_v_m_s;
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

int ParseInt(std::string const& s, std::string const& field_name) {
  char* end = nullptr;
  long const value = std::strtol(s.c_str(), &end, 10);
  if (end == s.c_str() || *end != '\0') {
    throw std::runtime_error("bad integer field " + field_name + ": " + s);
  }
  if (value < 0 || value > 100000) {
    throw std::runtime_error("integer field out of range " + field_name + ": " + s);
  }
  return static_cast<int>(value);
}

not_null<std::unique_ptr<Plugin const>> ReadPluginOrDie(std::string const& plugin_b64_path) {
  std::int64_t bytes_processed = 0;
  return principia::ksp_plugin_test::_plugin_io::ReadPluginFromFile(
      plugin_b64_path, "gipfeli", "base64", bytes_processed);
}

Position<Barycentric> MakePosition(Vec3 const& r_m) {
  Displacement<Barycentric> const d({r_m.x * Metre, r_m.y * Metre, r_m.z * Metre});
  return Barycentric::origin + d;
}

Velocity<Barycentric> MakeVelocity(Vec3 const& v_m_s) {
  return Velocity<Barycentric>({v_m_s.x * Metre / Second,
                                v_m_s.y * Metre / Second,
                                v_m_s.z * Metre / Second});
}

Vec3 ExtractPosition(DegreesOfFreedom<Barycentric> const& dof) {
  Displacement<Barycentric> const d = dof.position() - Barycentric::origin;
  return Vec3{d.coordinates()[0] / Metre,
              d.coordinates()[1] / Metre,
              d.coordinates()[2] / Metre};
}

Vec3 ExtractVelocity(DegreesOfFreedom<Barycentric> const& dof) {
  auto const& v = dof.velocity();
  return Vec3{v.coordinates()[0] / (Metre / Second),
              v.coordinates()[1] / (Metre / Second),
              v.coordinates()[2] / (Metre / Second)};
}

Vec3 Add(Vec3 const& a, Vec3 const& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

DegreesOfFreedom<Barycentric> MakeDof(Vec3 const& r_m, Vec3 const& v_m_s) {
  return DegreesOfFreedom<Barycentric>(MakePosition(r_m), MakeVelocity(v_m_s));
}

DegreesOfFreedom<Barycentric> Flow(
    Plugin const* const plugin,
    Instant const& t0,
    DegreesOfFreedom<Barycentric> const& dof0,
    Instant const& t1,
    std::string const& label) {
  auto trajectory = std::make_unique<DiscreteTrajectory<Barycentric>>();
  trajectory->Append(t0, dof0).IgnoreError();

  Ephemeris<Barycentric>* ephemeris =
      const_cast<Ephemeris<Barycentric>*>(plugin->ephemeris_for_testing());

  absl::Status const status = ephemeris->FlowWithAdaptiveStep(
      trajectory.get(),
      Ephemeris<Barycentric>::NoIntrinsicAcceleration,
      t1,
      plugin->parameters_for_testing(),
      Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

  if (!status.ok()) {
    throw std::runtime_error(label + " FlowWithAdaptiveStep failed: " +
                             std::string(status.message()));
  }
  return trajectory->EvaluateDegreesOfFreedom(t1);
}

void ValidateMonotonic(PropNRequest const& q) {
  if (!(q.t0_s <= q.t1_s)) {
    throw std::runtime_error("expected t0_s <= t1_s");
  }
  double previous = q.t0_s;
  for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
    double const t = q.impulses[i].burn_t_s;
    if (!(previous <= t && t <= q.t1_s)) {
      throw std::runtime_error("expected monotonic impulse times: t0 <= burn_t[i] <= t1");
    }
    previous = t;
  }
}

PropNRequest ParsePropRequest(std::vector<std::string> const& f) {
  // Backward-compatible v0.1 command:
  // PROP id t0 burn_t t1 x y z vx vy vz dvx dvy dvz
  if (f.size() != 14) {
    throw std::runtime_error("PROP expects 14 TSV fields including command; got " +
                             std::to_string(f.size()));
  }
  PropNRequest q;
  q.id = f[1];
  q.t0_s = ParseDouble(f[2], "t0_s");
  q.t1_s = ParseDouble(f[4], "t1_s");
  q.r0_m = Vec3{ParseDouble(f[5], "x_m"),
                ParseDouble(f[6], "y_m"),
                ParseDouble(f[7], "z_m")};
  q.v0_m_s = Vec3{ParseDouble(f[8], "vx_m_s"),
                  ParseDouble(f[9], "vy_m_s"),
                  ParseDouble(f[10], "vz_m_s")};
  q.impulses.push_back(Impulse{
      ParseDouble(f[3], "burn_t_s"),
      Vec3{ParseDouble(f[11], "burn_dvx_m_s"),
           ParseDouble(f[12], "burn_dvy_m_s"),
           ParseDouble(f[13], "burn_dvz_m_s")}});
  return q;
}

PropNRequest ParseProp2Request(std::vector<std::string> const& f) {
  // PROP2 id t0 tb0 tb1 t1 x y z vx vy vz dv0x dv0y dv0z dv1x dv1y dv1z
  if (f.size() != 18) {
    throw std::runtime_error("PROP2 expects 18 TSV fields including command; got " +
                             std::to_string(f.size()));
  }
  PropNRequest q;
  q.id = f[1];
  q.t0_s = ParseDouble(f[2], "t0_s");
  double const tb0 = ParseDouble(f[3], "tb0_s");
  double const tb1 = ParseDouble(f[4], "tb1_s");
  q.t1_s = ParseDouble(f[5], "t1_s");
  q.r0_m = Vec3{ParseDouble(f[6], "x_m"),
                ParseDouble(f[7], "y_m"),
                ParseDouble(f[8], "z_m")};
  q.v0_m_s = Vec3{ParseDouble(f[9], "vx_m_s"),
                  ParseDouble(f[10], "vy_m_s"),
                  ParseDouble(f[11], "vz_m_s")};
  q.impulses.push_back(Impulse{tb0, Vec3{ParseDouble(f[12], "dv0x_m_s"),
                                          ParseDouble(f[13], "dv0y_m_s"),
                                          ParseDouble(f[14], "dv0z_m_s")}});
  q.impulses.push_back(Impulse{tb1, Vec3{ParseDouble(f[15], "dv1x_m_s"),
                                          ParseDouble(f[16], "dv1y_m_s"),
                                          ParseDouble(f[17], "dv1z_m_s")}});
  return q;
}

PropNRequest ParsePropNRequest(std::vector<std::string> const& f) {
  // PROPN id t0 t1 n x y z vx vy vz [burn_t dvx dvy dvz] * n
  if (f.size() < 11) {
    throw std::runtime_error("PROPN expects at least 11 TSV fields including command; got " +
                             std::to_string(f.size()));
  }
  PropNRequest q;
  q.id = f[1];
  q.t0_s = ParseDouble(f[2], "t0_s");
  q.t1_s = ParseDouble(f[3], "t1_s");
  int const n = ParseInt(f[4], "n_impulses");
  int const expected = 11 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("PROPN expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }
  q.r0_m = Vec3{ParseDouble(f[5], "x_m"),
                ParseDouble(f[6], "y_m"),
                ParseDouble(f[7], "z_m")};
  q.v0_m_s = Vec3{ParseDouble(f[8], "vx_m_s"),
                  ParseDouble(f[9], "vy_m_s"),
                  ParseDouble(f[10], "vz_m_s")};
  int offset = 11;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0], "burn_t_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1], "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2], "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3], "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }
  return q;
}

PropNResult PropagateN(Plugin const* const plugin, PropNRequest const& q) {
  PropNResult r;
  r.id = q.id;
  r.t0_s = q.t0_s;
  r.t1_s = q.t1_s;

  try {
    ValidateMonotonic(q);

    Instant current_t = Instant() + q.t0_s * Second;
    DegreesOfFreedom<Barycentric> current = MakeDof(q.r0_m, q.v0_m_s);

    for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
      Impulse const& impulse = q.impulses[i];
      Instant const burn_t = Instant() + impulse.burn_t_s * Second;

      DegreesOfFreedom<Barycentric> const before = Flow(
          plugin, current_t, current, burn_t,
          "leg_to_burn[" + std::to_string(i) + "]");

      Vec3 const burn_r = ExtractPosition(before);
      Vec3 const v_before = ExtractVelocity(before);
      Vec3 const v_after = Add(v_before, impulse.dv_m_s);

      r.burns.push_back(BurnSnapshot{impulse.burn_t_s, burn_r, v_before, v_after});

      current_t = burn_t;
      current = MakeDof(burn_r, v_after);
    }

    Instant const final_t = Instant() + q.t1_s * Second;
    DegreesOfFreedom<Barycentric> const final_dof = Flow(
        plugin, current_t, current, final_t, "leg_to_final");

    r.final_r_m = ExtractPosition(final_dof);
    r.final_v_m_s = ExtractVelocity(final_dof);
    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}

void WriteERR(std::string const& id, std::string const& message) {
  std::cout << "ERR\t" << id << '\t' << message << '\n';
  std::cout.flush();
}

void WriteOKLegacy(PropNResult const& r) {
  if (r.burns.size() != 1) {
    WriteERR(r.id, "legacy OK requires exactly one burn");
    return;
  }
  BurnSnapshot const& b = r.burns[0];
  std::cout << std::setprecision(17)
            << "OK\t" << r.id << '\t' << r.t0_s << '\t' << b.burn_t_s << '\t' << r.t1_s << '\t'
            << b.r_m.x << '\t' << b.r_m.y << '\t' << b.r_m.z << '\t'
            << b.v_before_m_s.x << '\t' << b.v_before_m_s.y << '\t' << b.v_before_m_s.z << '\t'
            << b.v_after_m_s.x << '\t' << b.v_after_m_s.y << '\t' << b.v_after_m_s.z << '\t'
            << r.final_r_m.x << '\t' << r.final_r_m.y << '\t' << r.final_r_m.z << '\t'
            << r.final_v_m_s.x << '\t' << r.final_v_m_s.y << '\t' << r.final_v_m_s.z << '\n';
  std::cout.flush();
}

void WriteOKN(PropNResult const& r, std::string const& tag) {
  std::cout << std::setprecision(17)
            << tag << '\t' << r.id << '\t' << r.t0_s << '\t' << r.t1_s << '\t'
            << r.burns.size();
  for (BurnSnapshot const& b : r.burns) {
    std::cout << '\t' << b.burn_t_s
              << '\t' << b.r_m.x << '\t' << b.r_m.y << '\t' << b.r_m.z
              << '\t' << b.v_before_m_s.x << '\t' << b.v_before_m_s.y << '\t' << b.v_before_m_s.z
              << '\t' << b.v_after_m_s.x << '\t' << b.v_after_m_s.y << '\t' << b.v_after_m_s.z;
  }
  std::cout << '\t' << r.final_r_m.x << '\t' << r.final_r_m.y << '\t' << r.final_r_m.z
            << '\t' << r.final_v_m_s.x << '\t' << r.final_v_m_s.y << '\t' << r.final_v_m_s.z
            << '\n';
  std::cout.flush();
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

  std::cout << "READY\tprincipia_impulsive_particle_server_v0_2\n";
  std::cout.flush();

  std::string line;
  while (std::getline(std::cin, line)) {
    if (line.empty()) continue;
    auto fields = SplitTSV(line);
    if (fields.empty()) continue;

    std::string const& cmd = fields[0];
    if (cmd == "PING") {
      std::cout << "PONG\n";
      std::cout.flush();
      continue;
    }
    if (cmd == "QUIT") {
      std::cout << "BYE\n";
      std::cout.flush();
      break;
    }

    std::string id = fields.size() > 1 ? fields[1] : "";
    try {
      if (cmd == "PROP") {
        PropNRequest const q = ParsePropRequest(fields);
        PropNResult const r = PropagateN(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKLegacy(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }
      if (cmd == "PROP2") {
        PropNRequest const q = ParseProp2Request(fields);
        PropNResult const r = PropagateN(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKN(r, "OK2");
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }
      if (cmd == "PROPN") {
        PropNRequest const q = ParsePropNRequest(fields);
        PropNResult const r = PropagateN(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKN(r, "OKN");
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }

      WriteERR(id, "unknown command: " + cmd);
    } catch (std::exception const& e) {
      WriteERR(id, e.what());
    }
  }

  return 0;
}

}  // namespace
}  // namespace principia_impulsive_particle_server

int main(int argc, char** argv) {
  return principia_impulsive_particle_server::Main(argc, argv);
}
