#include <cstdlib>
#include <iomanip>
#include <iostream>
#include <memory>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>
#include <cmath>
#include <algorithm>
#include <cctype>
#include <optional>
#include <limits>

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
#include "ksp_plugin/celestial.hpp"
#include "ksp_plugin/vessel.hpp"
#include "physics/body_centred_non_rotating_reference_frame.hpp"
#include "physics/massive_body.hpp"
#include "geometry/grassmann.hpp"
#include "geometry/orthogonal_map.hpp"
#include "geometry/space.hpp"
#include "physics/rigid_motion.hpp"
#include "physics/reference_frame.hpp"


using namespace principia::physics::_body_centred_non_rotating_reference_frame;
using namespace principia::ksp_plugin::_frames;
using namespace principia::physics::_massive_body;
using namespace principia::geometry::_grassmann;
using namespace principia::geometry::_orthogonal_map;
using namespace principia::geometry::_space;
using namespace principia::physics::_rigid_motion;
using namespace principia::physics::_reference_frame;

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
using principia::ksp_plugin::_vessel::Vessel;
using principia::ksp_plugin::_celestial::Celestial;

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

struct PropSampleRequest {
  PropNRequest base;
  std::vector<double> sample_times_s;
};

struct ParentSnapshot {
  Vec3 r_parent_m;
  Vec3 v_parent_m_s;
  double distance_m = 0.0;
  double speed_m_s = 0.0;
  double radial_velocity_m_s = 0.0;
};

using RelativeSnapshot = ParentSnapshot;

struct VesselPropNRequest {
  std::string id;
  std::string vessel_guid;
  double final_dt_s = 0.0;
  std::vector<Impulse> impulses;  // burn_t_s aqui é dt relativo ao t0 da vessel.
};

struct VesselPropNResult {
  std::string id;
  std::string vessel_guid;
  std::string status;
  std::string message;

  double t0_game_s = 0.0;
  double t1_game_s = 0.0;

  std::vector<BurnSnapshot> burns;

  Vec3 initial_r_m;
  Vec3 initial_v_m_s;
  ParentSnapshot initial_parent;

  Vec3 final_r_m;
  Vec3 final_v_m_s;
  ParentSnapshot final_parent;
};

struct VRelRequest {
  std::string id;
  std::string vessel_guid;
  std::string reference_body;
  double final_dt_s = 0.0;
  std::vector<Impulse> impulses;
};

struct VRelResult {
  std::string id;
  std::string vessel_guid;
  std::string reference_body;
  std::string status;
  std::string message;

  double t0_game_s = 0.0;
  double t1_game_s = 0.0;

  std::vector<BurnSnapshot> burns;

  Vec3 final_rel_r_m;
  Vec3 final_rel_v_m_s;
  double distance_m = 0.0;
  double speed_m_s = 0.0;
  double radial_velocity_m_s = 0.0;

  Vec3 final_abs_r_m;
  Vec3 final_abs_v_m_s;

  Vec3 reference_abs_r_m;
  Vec3 reference_abs_v_m_s;
};

struct VCARequest {
  std::string id;
  std::string vessel_guid;
  std::string target_body;
  double scan_start_dt_s = 0.0;
  double scan_end_dt_s = 0.0;
  int samples = 0;
  std::vector<Impulse> impulses;
};

struct VCAResult {
  std::string id;
  std::string vessel_guid;
  std::string target_body;
  std::string status;
  std::string message;

  double ca_dt_s = 0.0;
  double ca_t_game_s = 0.0;

  double t0_game_s = 0.0;

  Vec3 ca_rel_r_m;
  Vec3 ca_rel_v_m_s;
  double ca_distance_m = 0.0;
  double ca_speed_m_s = 0.0;
  double ca_radial_velocity_m_s = 0.0;

  int samples = 0;
  std::string ca_status;
};

struct VCARelRequest {
  std::string id;
  std::string dep_body;
  std::string arr_body;

  // Absolute game time relative to plugin->GameEpoch().
  double state_dt_s = 0.0;

  // Scan window relative to state_t.
  double scan_start_dt_s = 0.0;
  double scan_end_dt_s = 0.0;
  int samples = 0;

  // Initial particle state relative to dep_body, in raw/Barycentric axes.
  Vec3 rel_r_m;
  Vec3 rel_v_m_s;

  // Impulses relative to state_t, in raw/Barycentric axes.
  std::vector<Impulse> impulses;
};

struct VCARelResult {
  std::string id;
  std::string dep_body;
  std::string arr_body;
  std::string status;
  std::string message;

  double state_dt_s = 0.0;
  double state_t_game_s = 0.0;

  double ca_dt_s = 0.0;       // Relative to state_t.
  double ca_t_game_s = 0.0;   // Relative to plugin->GameEpoch().

  Vec3 ca_rel_r_m;
  Vec3 ca_rel_v_m_s;
  double ca_distance_m = 0.0;
  double ca_speed_m_s = 0.0;
  double ca_radial_velocity_m_s = 0.0;

  Vec3 ca_abs_debug_r_m;
  Vec3 ca_abs_debug_v_m_s;

  Vec3 arr_abs_debug_r_m;
  Vec3 arr_abs_debug_v_m_s;

  int samples = 0;
  std::string ca_status;

  std::vector<BurnSnapshot> burns;
};

struct NavImpulse {
  double burn_t_s = 0.0;  // Relative to state_t.
  Vec3 dv_tnb_m_s;
};

struct NavBurnSnapshot {
  double burn_t_s = 0.0;

  Vec3 burn_r_raw_m;
  Vec3 burn_v_before_raw_m_s;

  Vec3 dv_tnb_cmd_m_s;

  Vec3 tangent_raw;
  Vec3 normal_raw;
  Vec3 binormal_raw;

  Vec3 dv_raw_m_s;
  Vec3 burn_v_after_raw_m_s;
};

struct VCARelNavRequest {
  std::string id;
  std::string dep_body;
  std::string arr_body;
  std::string nav_body;

  // Absolute game time relative to plugin->GameEpoch().
  double state_dt_s = 0.0;

  // Scan window relative to state_t.
  double scan_start_dt_s = 0.0;
  double scan_end_dt_s = 0.0;
  int samples = 0;

  // Initial particle state relative to dep_body, in raw/Barycentric axes.
  Vec3 rel_r_m;
  Vec3 rel_v_m_s;

  // Impulses relative to state_t, in Frenet/TNB components.
  std::vector<NavImpulse> impulses;
};

struct VCARelNavResult {
  std::string id;
  std::string dep_body;
  std::string arr_body;
  std::string nav_body;
  std::string status;
  std::string message;

  double state_dt_s = 0.0;
  double state_t_game_s = 0.0;

  double ca_dt_s = 0.0;      // Relative to state_t.
  double ca_t_game_s = 0.0;  // Relative to plugin->GameEpoch().

  Vec3 ca_rel_r_m;
  Vec3 ca_rel_v_m_s;
  double ca_distance_m = 0.0;
  double ca_speed_m_s = 0.0;
  double ca_radial_velocity_m_s = 0.0;

  Vec3 ca_abs_debug_r_m;
  Vec3 ca_abs_debug_v_m_s;

  Vec3 arr_abs_debug_r_m;
  Vec3 arr_abs_debug_v_m_s;

  int samples = 0;
  std::string ca_status;

  std::vector<NavBurnSnapshot> burns;
};

struct StateSample {
  double t_s = 0.0;
  Vec3 r_m;
  Vec3 v_m_s;
};

struct PropSampleResult {
  std::string id;
  std::string status;
  std::string message;
  double t0_s = 0.0;
  double t1_s = 0.0;
  std::vector<StateSample> samples;
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

Vec3 ExtractDisplacement(Displacement<Barycentric> const& d) {
  return Vec3{d.coordinates()[0] / Metre,
              d.coordinates()[1] / Metre,
              d.coordinates()[2] / Metre};
}

Vec3 ExtractVelocityVector(Velocity<Barycentric> const& v) {
  return Vec3{v.coordinates()[0] / (Metre / Second),
              v.coordinates()[1] / (Metre / Second),
              v.coordinates()[2] / (Metre / Second)};
}

template<typename Frame>
Vec3 ExtractDirection(Vector<double, Frame> const& v) {
  return Vec3{
      v.coordinates()[0],
      v.coordinates()[1],
      v.coordinates()[2]};
}

Vec3 Add(Vec3 const& a, Vec3 const& b) {
  return Vec3{a.x + b.x, a.y + b.y, a.z + b.z};
}

Vec3 Sub(Vec3 const& a, Vec3 const& b) {
  return Vec3{a.x - b.x, a.y - b.y, a.z - b.z};
}

double Dot(Vec3 const& a, Vec3 const& b) {
  return a.x * b.x + a.y * b.y + a.z * b.z;
}

double Norm(Vec3 const& a) {
  return std::sqrt(Dot(a, a));
}

std::string CanonicalBodyName(std::string s) {
  std::string out;
  out.reserve(s.size());
  for (unsigned char c : s) {
    if (c == '_' || c == '-' || c == ' ' || c == '\t') {
      continue;
    }
    out.push_back(static_cast<char>(std::toupper(c)));
  }
  return out;
}

bool IsIntegerToken(std::string const& s) {
  if (s.empty()) {
    return false;
  }
  int i = 0;
  if (s[0] == '+' || s[0] == '-') {
    i = 1;
  }
  if (i == static_cast<int>(s.size())) {
    return false;
  }
  for (; i < static_cast<int>(s.size()); ++i) {
    if (!std::isdigit(static_cast<unsigned char>(s[i]))) {
      return false;
    }
  }
  return true;
}

DegreesOfFreedom<Barycentric> MakeDof(Vec3 const& r_m, Vec3 const& v_m_s) {
  return DegreesOfFreedom<Barycentric>(MakePosition(r_m), MakeVelocity(v_m_s));
}

void EnsureEphemerisCovers(Plugin const* const plugin,
                           Instant const& t,
                           std::string const& label) {
  absl::Status const status = plugin->ephemeris()->Prolong(t);
  if (!status.ok()) {
    std::ostringstream ss;
    ss << label
       << ": ephemeris Prolong failed at game_s="
       << ((t - plugin->GameEpoch()) / Second)
       << ": "
       << status.ToString();
    throw std::runtime_error(ss.str());
  }
}

DegreesOfFreedom<Barycentric> MakeDofRelativeToBody(
    Plugin const* const plugin,
    Celestial const& dep,
    Instant const& t,
    Vec3 const& rel_r_m,
    Vec3 const& rel_v_m_s) {
  EnsureEphemerisCovers(plugin, t, "MakeDofRelativeToBody");

  DegreesOfFreedom<Barycentric> const dep_dof =
      dep.current_degrees_of_freedom(t);

  Displacement<Barycentric> const rel_r({
      rel_r_m.x * Metre,
      rel_r_m.y * Metre,
      rel_r_m.z * Metre});

  Velocity<Barycentric> const rel_v({
      rel_v_m_s.x * Metre / Second,
      rel_v_m_s.y * Metre / Second,
      rel_v_m_s.z * Metre / Second});

  return DegreesOfFreedom<Barycentric>(
      dep_dof.position() + rel_r,
      dep_dof.velocity() + rel_v);
}

not_null<MassiveBody const*> MassiveBodyOf(Celestial const& celestial) {
  return not_null<MassiveBody const*>(
      static_cast<MassiveBody const*>(&*celestial.body()));
}

struct NavToRawConversion {
  Vec3 tangent_raw;
  Vec3 normal_raw;
  Vec3 binormal_raw;
  Vec3 dv_raw_m_s;
};

NavToRawConversion ConvertNavImpulseToRaw(
    Plugin const* const plugin,
    NavigationFrame const& navigation_frame,
    Instant const& t,
    DegreesOfFreedom<Barycentric> const& dof,
    Vec3 const& dv_tnb_m_s,
    std::string const& label) {
  EnsureEphemerisCovers(plugin, t, label + ": ConvertNavImpulseToRaw");

  RigidMotion<Barycentric, Navigation> const to_frame_at_t =
      navigation_frame.ToThisFrameAtTime(t);

  RigidMotion<Navigation, Barycentric> const from_frame_at_t =
      to_frame_at_t.Inverse();

  auto const frenet_to_barycentric =
      from_frame_at_t.orthogonal_map() *
      navigation_frame
          .FrenetFrame(t, to_frame_at_t(dof))
          .template Forget<OrthogonalMap>();

  Vector<double, Frenet<Navigation>> const tangent_tnb({1, 0, 0});
  Vector<double, Frenet<Navigation>> const normal_tnb({0, 1, 0});
  Vector<double, Frenet<Navigation>> const binormal_tnb({0, 0, 1});

  Velocity<Frenet<Navigation>> const dv_tnb({
      dv_tnb_m_s.x * Metre / Second,
      dv_tnb_m_s.y * Metre / Second,
      dv_tnb_m_s.z * Metre / Second});

  Velocity<Barycentric> const dv_raw = frenet_to_barycentric(dv_tnb);

  return NavToRawConversion{
      ExtractDirection(frenet_to_barycentric(tangent_tnb)),
      ExtractDirection(frenet_to_barycentric(normal_tnb)),
      ExtractDirection(frenet_to_barycentric(binormal_tnb)),
      ExtractVelocityVector(dv_raw)};
}

not_null<std::unique_ptr<NavigationFrame>> MakeBodyCentredNonRotatingNavFrame(
    Plugin const* const plugin,
    Celestial const& centre) {
  return make_not_null_unique<
      BodyCentredNonRotatingReferenceFrame<Barycentric, Navigation>>(
          plugin->ephemeris(),
          MassiveBodyOf(centre));
}

not_null<Celestial const*> ResolveCelestial(
    Plugin const* const plugin,
    std::string const& body_token) {
  if (body_token.empty()) {
    throw std::runtime_error("empty body token");
  }

  // 1. Numeric celestial index, useful for debugging.
  if (IsIntegerToken(body_token)) {
    int const index = ParseInt(body_token, "body_index");
    if (!plugin->HasCelestial(index)) {
      throw std::runtime_error("unknown celestial index: " + body_token);
    }
    return not_null<Celestial const*>(&plugin->GetCelestial(index));
  }

  // 2. Match by body name through the ephemeris. This avoids needing access to
  // Plugin::name_to_index_, which is private.
  std::string const wanted = CanonicalBodyName(body_token);

  std::vector<std::string> known_names;
  for (auto const body : plugin->ephemeris()->bodies()) {
    std::string const name = body->name();
    known_names.push_back(name);

    if (CanonicalBodyName(name) == wanted) {
      int const index = plugin->CelestialIndexOfBody(*body);
      if (!plugin->HasCelestial(index)) {
        throw std::runtime_error(
            "body name resolved to missing celestial index for: " + name);
      }
      return not_null<Celestial const*>(&plugin->GetCelestial(index));
    }
  }

  std::ostringstream message;
  message << "unknown celestial body: " << body_token << "; known bodies:";
  for (std::string const& name : known_names) {
    message << " " << name;
  }
  throw std::runtime_error(message.str());
}

ParentSnapshot ComputeParentSnapshot(
    Plugin const* const plugin,
    Vessel const* const vessel,
    Instant const& t,
    DegreesOfFreedom<Barycentric> const& vessel_dof) {
  auto const parent_body = vessel->parent()->body();
  auto const parent_dof =
      plugin->ephemeris()->trajectory(parent_body)->EvaluateDegreesOfFreedom(t);

  Vec3 const r_parent_m =
      ExtractDisplacement(vessel_dof.position() - parent_dof.position());
  Vec3 const v_parent_m_s =
      ExtractVelocityVector(vessel_dof.velocity() - parent_dof.velocity());

  double const distance_m = Norm(r_parent_m);
  double const speed_m_s = Norm(v_parent_m_s);
  double const radial_velocity_m_s =
      distance_m > 0.0 ? Dot(r_parent_m, v_parent_m_s) / distance_m : 0.0;

  return ParentSnapshot{
      r_parent_m,
      v_parent_m_s,
      distance_m,
      speed_m_s,
      radial_velocity_m_s};
}


RelativeSnapshot ComputeRelativeSnapshot(
    Plugin const* const plugin,
    Celestial const& reference,
    Instant const& t,
    DegreesOfFreedom<Barycentric> const& vessel_dof) {
  EnsureEphemerisCovers(plugin, t, "ComputeRelativeSnapshot");

  auto const reference_dof = reference.current_degrees_of_freedom(t);

  Vec3 const rel_r_m =
      ExtractDisplacement(vessel_dof.position() - reference_dof.position());
  Vec3 const rel_v_m_s =
      ExtractVelocityVector(vessel_dof.velocity() - reference_dof.velocity());

  double const distance_m = Norm(rel_r_m);
  double const speed_m_s = Norm(rel_v_m_s);
  double const radial_velocity_m_s =
      distance_m > 0.0 ? Dot(rel_r_m, rel_v_m_s) / distance_m : 0.0;

  return RelativeSnapshot{
      rel_r_m,
      rel_v_m_s,
      distance_m,
      speed_m_s,
      radial_velocity_m_s};
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

  // --- A MUDANÇA OBRIGATÓRIA PARA O IPOPT ---
  if (!status.ok()) {
    // Não lance erro (throw). Apenas retorne o último ponto antes da colisão.
    // Assim o IPOPT "enxerga" a distância real do erro e calcula o gradiente.
    std::cerr << "[WARN] Soft-Fail at " << trajectory->back().time 
              << " (" << label << "): " << status.message() << std::endl;
    return trajectory->back().degrees_of_freedom;
  }
  // ------------------------------------------

  // Se chegou aqui, a simulação foi perfeita até t1.
  return trajectory->EvaluateDegreesOfFreedom(t1);
}

DegreesOfFreedom<Barycentric> FlowWithParameters(
    Plugin const* const plugin,
    Ephemeris<Barycentric>::AdaptiveStepParameters const& parameters,
    Instant const& t0,
    DegreesOfFreedom<Barycentric> const& dof0,
    Instant const& t1,
    std::string const& label) {
  auto trajectory = std::make_unique<DiscreteTrajectory<Barycentric>>();
  trajectory->Append(t0, dof0).IgnoreError();

  absl::Status const status =
      plugin->ephemeris()->FlowWithAdaptiveStep(
          trajectory.get(),
          Ephemeris<Barycentric>::NoIntrinsicAcceleration,
          t1,
          parameters,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

  // --- O MILAGRE DO SOFT-FAIL APLICADO NA FUNÇÃO CORRETA ---
  if (!status.ok()) {
    // Registra o aviso no terminal para você acompanhar, mas não derruba o servidor.
    std::cerr << "[WARN] Soft-Fail at " << trajectory->back().time 
              << " (" << label << "): " << status.ToString() << std::endl;
              
    // Devolve as coordenadas exatas de onde a nave bateu/simulação quebrou.
    // O IPOPT vai converter isso em um gradiente de "posição muito ruim, vire o foguete".
    return trajectory->back().degrees_of_freedom;
  }
  // ---------------------------------------------------------

  return trajectory->EvaluateDegreesOfFreedom(t1);
}

DegreesOfFreedom<Barycentric> FlowWithParametersStrict(
    Plugin const* const plugin,
    Ephemeris<Barycentric>::AdaptiveStepParameters const& parameters,
    Instant const& t0,
    DegreesOfFreedom<Barycentric> const& dof0,
    Instant const& t1,
    std::string const& label) {
  auto trajectory = std::make_unique<DiscreteTrajectory<Barycentric>>();
  trajectory->Append(t0, dof0).IgnoreError();

  absl::Status const status =
      plugin->ephemeris()->FlowWithAdaptiveStep(
          trajectory.get(),
          Ephemeris<Barycentric>::NoIntrinsicAcceleration,
          t1,
          parameters,
          Ephemeris<Barycentric>::unlimited_max_ephemeris_steps);

  if (!status.ok()) {
    std::ostringstream ss;
    ss << label << " FlowWithAdaptiveStep failed at "
       << ((trajectory->back().time - plugin->GameEpoch()) / Second)
       << " game seconds: " << status.ToString();
    throw std::runtime_error(ss.str());
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

void ValidateVesselMonotonic(VesselPropNRequest const& q) {
  if (!(q.final_dt_s >= 0.0)) {
    throw std::runtime_error("expected final_dt_s >= 0");
  }

  double previous = 0.0;
  for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
    double const dt = q.impulses[i].burn_t_s;
    if (!(previous <= dt && dt <= q.final_dt_s)) {
      throw std::runtime_error(
          "expected monotonic relative impulse times: 0 <= burn_dt[i] <= final_dt");
    }
    previous = dt;
  }
}

void ValidateRelativeImpulseTimes(double const final_dt_s,
                                  std::vector<Impulse> const& impulses,
                                  std::string const& label) {
  if (!(final_dt_s >= 0.0)) {
    throw std::runtime_error(label + ": expected final_dt_s >= 0");
  }

  double previous = 0.0;
  for (int i = 0; i < static_cast<int>(impulses.size()); ++i) {
    double const dt = impulses[i].burn_t_s;
    if (!(previous <= dt && dt <= final_dt_s)) {
      throw std::runtime_error(
          label + ": expected monotonic relative impulse times: "
                  "0 <= burn_dt[i] <= final_dt");
    }
    previous = dt;
  }
}

void ValidateVCA(VCARequest const& q) {
  if (!(q.scan_start_dt_s >= 0.0)) {
    throw std::runtime_error("VCA: expected scan_start_dt_s >= 0");
  }
  if (!(q.scan_end_dt_s > q.scan_start_dt_s)) {
    throw std::runtime_error("VCA: expected scan_end_dt_s > scan_start_dt_s");
  }
  if (q.samples < 2) {
    throw std::runtime_error("VCA: expected samples >= 2");
  }

  double previous = 0.0;
  for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
    double const dt = q.impulses[i].burn_t_s;
    if (!(previous <= dt && dt <= q.scan_end_dt_s)) {
      throw std::runtime_error(
          "VCA: expected monotonic impulse times: "
          "0 <= burn_dt[i] <= scan_end_dt_s");
    }
    previous = dt;
  }
}

void ValidateVCARel(VCARelRequest const& q) {
  if (!(q.state_dt_s >= 0.0)) {
    throw std::runtime_error("VCAREL: expected state_dt_s >= 0");
  }
  if (!(q.scan_start_dt_s >= 0.0)) {
    throw std::runtime_error("VCAREL: expected scan_start_dt_s >= 0");
  }
  if (!(q.scan_end_dt_s > q.scan_start_dt_s)) {
    throw std::runtime_error(
        "VCAREL: expected scan_end_dt_s > scan_start_dt_s");
  }
  if (q.samples < 2) {
    throw std::runtime_error("VCAREL: expected samples >= 2");
  }

  double previous = 0.0;
  for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
    double const dt = q.impulses[i].burn_t_s;
    if (!(previous <= dt && dt <= q.scan_end_dt_s)) {
      throw std::runtime_error(
          "VCAREL: expected monotonic impulse times: "
          "0 <= impulse_dt_s[i] <= scan_end_dt_s");
    }
    previous = dt;
  }
}

void ValidateVCARelNav(VCARelNavRequest const& q) {
  if (!(q.state_dt_s >= 0.0)) {
    throw std::runtime_error("VCAREL_NAV: expected state_dt_s >= 0");
  }
  if (!(q.scan_start_dt_s >= 0.0)) {
    throw std::runtime_error("VCAREL_NAV: expected scan_start_dt_s >= 0");
  }
  if (!(q.scan_end_dt_s > q.scan_start_dt_s)) {
    throw std::runtime_error(
        "VCAREL_NAV: expected scan_end_dt_s > scan_start_dt_s");
  }
  if (q.samples < 2) {
    throw std::runtime_error("VCAREL_NAV: expected samples >= 2");
  }

  double previous = 0.0;
  for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
    double const dt = q.impulses[i].burn_t_s;
    if (!(previous <= dt && dt <= q.scan_end_dt_s)) {
      throw std::runtime_error(
          "VCAREL_NAV: expected monotonic impulse times: "
          "0 <= impulse_dt_s[i] <= scan_end_dt_s");
    }
    previous = dt;
  }
}

void CheckImpulseApplied(Vec3 const& commanded,
                         Vec3 const& before,
                         Vec3 const& after,
                         std::string const& label) {
  Vec3 const applied = Sub(after, before);
  Vec3 const error = Sub(applied, commanded);
  double const error_norm = Norm(error);
  if (!(error_norm <= 1e-9)) {
    std::ostringstream ss;
    ss << label
       << ": impulse application mismatch; commanded=("
       << commanded.x << "," << commanded.y << "," << commanded.z
       << ") applied=("
       << applied.x << "," << applied.y << "," << applied.z
       << ") error_norm=" << error_norm;
    throw std::runtime_error(ss.str());
  }
}

struct VesselInitialContext {
  Vessel* vessel = nullptr;
  Instant t0;
  DegreesOfFreedom<Barycentric> dof0;
  Ephemeris<Barycentric>::AdaptiveStepParameters parameters;
};

struct RelativeParticleContext {
  Instant state_t;
  DegreesOfFreedom<Barycentric> dof0;
  Ephemeris<Barycentric>::AdaptiveStepParameters parameters;
};

VesselInitialContext GetVesselInitialContext(
    Plugin const* const plugin,
    std::string const& vessel_guid,
    std::int64_t min_max_steps) {
  if (!plugin->HasVessel(vessel_guid)) {
    throw std::runtime_error("unknown vessel_guid: " + vessel_guid);
  }

  Vessel* const vessel = plugin->GetVessel(vessel_guid);
  auto const psychohistory = vessel->psychohistory();
  if (psychohistory == vessel->trajectory().segments().end() ||
      psychohistory->empty()) {
    throw std::runtime_error("vessel psychohistory is missing or empty");
  }

  auto const& initial_point = psychohistory->back();

  auto parameters = vessel->prediction_adaptive_step_parameters();
  parameters.set_max_steps(
      std::max<std::int64_t>(parameters.max_steps(), min_max_steps));

  return VesselInitialContext{
      vessel,
      initial_point.time,
      initial_point.degrees_of_freedom,
      parameters};
}

RelativeParticleContext GetRelativeParticleContext(
    Plugin const* const plugin,
    VCARelRequest const& q) {
  Celestial const& dep = *ResolveCelestial(plugin, q.dep_body);

  Instant const state_t = plugin->GameEpoch() + q.state_dt_s * Second;

  DegreesOfFreedom<Barycentric> const dof0 =
    MakeDofRelativeToBody(plugin, dep, state_t, q.rel_r_m, q.rel_v_m_s);

  // Use the same integration parameters already used by the legacy particle
  // propagator. This keeps VCAREL independent of any vessel.
  auto parameters = plugin->parameters_for_testing();
  parameters.set_max_steps(
      std::max<std::int64_t>(parameters.max_steps(), 50000));

  return RelativeParticleContext{
      state_t,
      dof0,
      parameters};
}

DegreesOfFreedom<Barycentric> PropagateVesselWithImpulsesToDt(
    Plugin const* const plugin,
    VesselInitialContext const& ctx,
    std::vector<Impulse> const& impulses,
    double const final_dt_s,
    std::string const& label,
    std::vector<BurnSnapshot>* const burns_out = nullptr) {
  Instant current_t = ctx.t0;
  DegreesOfFreedom<Barycentric> current = ctx.dof0;

  for (int i = 0; i < static_cast<int>(impulses.size()); ++i) {
    Impulse const& impulse = impulses[i];
    if (impulse.burn_t_s > final_dt_s) {
      break;
    }

    Instant const burn_t = ctx.t0 + impulse.burn_t_s * Second;

    DegreesOfFreedom<Barycentric> const before =
        FlowWithParametersStrict(plugin,
                                 ctx.parameters,
                                 current_t,
                                 current,
                                 burn_t,
                                 label + "_to_burn[" + std::to_string(i) + "]");

    Vec3 const burn_r = ExtractPosition(before);
    Vec3 const v_before = ExtractVelocity(before);
    Vec3 const v_after = Add(v_before, impulse.dv_m_s);

    CheckImpulseApplied(impulse.dv_m_s,
                        v_before,
                        v_after,
                        label + " burn[" + std::to_string(i) + "]");

    if (burns_out != nullptr) {
      burns_out->push_back(
          BurnSnapshot{impulse.burn_t_s, burn_r, v_before, v_after});
    }

    current_t = burn_t;
    current = MakeDof(burn_r, v_after);
  }

  Instant const final_t = ctx.t0 + final_dt_s * Second;
  return FlowWithParametersStrict(plugin,
                                  ctx.parameters,
                                  current_t,
                                  current,
                                  final_t,
                                  label + "_to_final");
}

DegreesOfFreedom<Barycentric> PropagateRelativeParticleWithImpulsesToDt(
    Plugin const* const plugin,
    RelativeParticleContext const& ctx,
    std::vector<Impulse> const& impulses,
    double const final_dt_s,
    std::string const& label,
    std::vector<BurnSnapshot>* const burns_out = nullptr) {
  Instant current_t = ctx.state_t;
  DegreesOfFreedom<Barycentric> current = ctx.dof0;

  for (int i = 0; i < static_cast<int>(impulses.size()); ++i) {
    Impulse const& impulse = impulses[i];
    if (impulse.burn_t_s > final_dt_s) {
      break;
    }

    Instant const burn_t = ctx.state_t + impulse.burn_t_s * Second;

    DegreesOfFreedom<Barycentric> const before =
        FlowWithParametersStrict(plugin,
                                 ctx.parameters,
                                 current_t,
                                 current,
                                 burn_t,
                                 label + "_to_impulse[" +
                                     std::to_string(i) + "]");

    Vec3 const burn_r = ExtractPosition(before);
    Vec3 const v_before = ExtractVelocity(before);
    Vec3 const v_after = Add(v_before, impulse.dv_m_s);

    CheckImpulseApplied(impulse.dv_m_s,
                        v_before,
                        v_after,
                        label + " impulse[" + std::to_string(i) + "]");

    if (burns_out != nullptr) {
      burns_out->push_back(
          BurnSnapshot{impulse.burn_t_s, burn_r, v_before, v_after});
    }

    current_t = burn_t;
    current = MakeDof(burn_r, v_after);
  }

  Instant const final_t = ctx.state_t + final_dt_s * Second;
  return FlowWithParametersStrict(plugin,
                                  ctx.parameters,
                                  current_t,
                                  current,
                                  final_t,
                                  label + "_to_final");
}

DegreesOfFreedom<Barycentric> PropagateRelativeParticleWithNavImpulsesToDt(
    Plugin const* const plugin,
    RelativeParticleContext const& ctx,
    NavigationFrame const& navigation_frame,
    std::vector<NavImpulse> const& impulses,
    double const final_dt_s,
    std::string const& label,
    std::vector<NavBurnSnapshot>* const burns_out = nullptr) {
  Instant current_t = ctx.state_t;
  DegreesOfFreedom<Barycentric> current = ctx.dof0;

  for (int i = 0; i < static_cast<int>(impulses.size()); ++i) {
    NavImpulse const& impulse = impulses[i];
    if (impulse.burn_t_s > final_dt_s) {
      break;
    }

    Instant const burn_t = ctx.state_t + impulse.burn_t_s * Second;

    DegreesOfFreedom<Barycentric> const before =
        FlowWithParametersStrict(plugin,
                                 ctx.parameters,
                                 current_t,
                                 current,
                                 burn_t,
                                 label + "_to_nav_impulse[" +
                                     std::to_string(i) + "]");

    NavToRawConversion const converted =
        ConvertNavImpulseToRaw(plugin,
                               navigation_frame,
                               burn_t,
                               before,
                               impulse.dv_tnb_m_s,
                               label + "_nav_impulse[" +
                                   std::to_string(i) + "]");

    Vec3 const burn_r = ExtractPosition(before);
    Vec3 const v_before = ExtractVelocity(before);
    Vec3 const v_after = Add(v_before, converted.dv_raw_m_s);

    CheckImpulseApplied(converted.dv_raw_m_s,
                        v_before,
                        v_after,
                        label + " nav_impulse[" + std::to_string(i) + "]");

    if (burns_out != nullptr) {
      burns_out->push_back(
          NavBurnSnapshot{
              impulse.burn_t_s,
              burn_r,
              v_before,
              impulse.dv_tnb_m_s,
              converted.tangent_raw,
              converted.normal_raw,
              converted.binormal_raw,
              converted.dv_raw_m_s,
              v_after});
    }

    current_t = burn_t;
    current = MakeDof(burn_r, v_after);
  }

  Instant const final_t = ctx.state_t + final_dt_s * Second;
  return FlowWithParametersStrict(plugin,
                                  ctx.parameters,
                                  current_t,
                                  current,
                                  final_t,
                                  label + "_to_final");
}

struct RelativeAtDt {
  double dt_s = 0.0;
  double game_s = 0.0;
  Vec3 rel_r_m;
  Vec3 rel_v_m_s;
  double distance_m = 0.0;
  double speed_m_s = 0.0;
  double radial_velocity_m_s = 0.0;
};

RelativeAtDt EvaluateRelativeAtDt(
    Plugin const* const plugin,
    VesselInitialContext const& ctx,
    Celestial const& target,
    std::vector<Impulse> const& impulses,
    double const dt_s,
    std::string const& label) {
  DegreesOfFreedom<Barycentric> const vessel_dof =
      PropagateVesselWithImpulsesToDt(plugin, ctx, impulses, dt_s, label);

  Instant const t = ctx.t0 + dt_s * Second;
  RelativeSnapshot const rel =
    ComputeRelativeSnapshot(plugin, target, t, vessel_dof);

  return RelativeAtDt{
      dt_s,
      (t - plugin->GameEpoch()) / Second,
      rel.r_parent_m,
      rel.v_parent_m_s,
      rel.distance_m,
      rel.speed_m_s,
      rel.radial_velocity_m_s};
}

RelativeAtDt EvaluateRelativeParticleAtDt(
    Plugin const* const plugin,
    RelativeParticleContext const& ctx,
    Celestial const& arr,
    std::vector<Impulse> const& impulses,
    double const dt_s,
    std::string const& label) {
  DegreesOfFreedom<Barycentric> const particle_dof =
      PropagateRelativeParticleWithImpulsesToDt(
          plugin, ctx, impulses, dt_s, label);

  Instant const t = ctx.state_t + dt_s * Second;
  RelativeSnapshot const rel =
    ComputeRelativeSnapshot(plugin, arr, t, particle_dof);

  return RelativeAtDt{
      dt_s,
      (t - plugin->GameEpoch()) / Second,
      rel.r_parent_m,
      rel.v_parent_m_s,
      rel.distance_m,
      rel.speed_m_s,
      rel.radial_velocity_m_s};
}

RelativeAtDt EvaluateRelativeParticleNavAtDt(
    Plugin const* const plugin,
    RelativeParticleContext const& ctx,
    NavigationFrame const& navigation_frame,
    Celestial const& arr,
    std::vector<NavImpulse> const& impulses,
    double const dt_s,
    std::string const& label) {
  DegreesOfFreedom<Barycentric> const particle_dof =
      PropagateRelativeParticleWithNavImpulsesToDt(
          plugin,
          ctx,
          navigation_frame,
          impulses,
          dt_s,
          label);

  Instant const t = ctx.state_t + dt_s * Second;
  RelativeSnapshot const rel =
      ComputeRelativeSnapshot(plugin, arr, t, particle_dof);

  return RelativeAtDt{
      dt_s,
      (t - plugin->GameEpoch()) / Second,
      rel.r_parent_m,
      rel.v_parent_m_s,
      rel.distance_m,
      rel.speed_m_s,
      rel.radial_velocity_m_s};
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

VesselPropNRequest ParseVesselPropNRequest(std::vector<std::string> const& f) {
  // VPROPN id vessel_guid final_dt_s n [burn_dt_s dvx dvy dvz] * n
  //
  // Todos os tempos são relativos a vessel->psychohistory()->back().time.
  // dv é Barycentric raw em m/s.
  if (f.size() < 5) {
    throw std::runtime_error("VPROPN expects at least 5 TSV fields including command; got " +
                             std::to_string(f.size()));
  }

  VesselPropNRequest q;
  q.id = f[1];
  q.vessel_guid = f[2];
  q.final_dt_s = ParseDouble(f[3], "final_dt_s");

  int const n = ParseInt(f[4], "n_impulses");
  int const expected = 5 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("VPROPN expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }

  int offset = 5;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0], "burn_dt_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1], "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2], "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3], "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }

  return q;
}

VRelRequest ParseVRelRequest(std::vector<std::string> const& f) {
  // VREL id vessel_guid reference_body final_dt_s n_burns
  //      [burn_dt_s dvx_raw_m_s dvy_raw_m_s dvz_raw_m_s] * n
  //
  // Todos os tempos são relativos a vessel->psychohistory()->back().time.
  // Delta-v é Barycentric raw em m/s.
  if (f.size() < 6) {
    throw std::runtime_error(
        "VREL expects at least 6 TSV fields including command; got " +
        std::to_string(f.size()));
  }

  VRelRequest q;
  q.id = f[1];
  q.vessel_guid = f[2];
  q.reference_body = f[3];
  q.final_dt_s = ParseDouble(f[4], "final_dt_s");

  int const n = ParseInt(f[5], "n_burns");
  int const expected = 6 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("VREL expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }

  int offset = 6;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0], "burn_dt_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1], "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2], "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3], "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }

  return q;
}

VCARequest ParseVCARequest(std::vector<std::string> const& f) {
  // VCA id vessel_guid target_body scan_start_dt_s scan_end_dt_s samples n_burns
  //     [burn_dt_s dvx_raw_m_s dvy_raw_m_s dvz_raw_m_s] * n
  if (f.size() < 8) {
    throw std::runtime_error(
        "VCA expects at least 8 TSV fields including command; got " +
        std::to_string(f.size()));
  }

  VCARequest q;
  q.id = f[1];
  q.vessel_guid = f[2];
  q.target_body = f[3];
  q.scan_start_dt_s = ParseDouble(f[4], "scan_start_dt_s");
  q.scan_end_dt_s = ParseDouble(f[5], "scan_end_dt_s");
  q.samples = ParseInt(f[6], "samples");

  int const n = ParseInt(f[7], "n_burns");
  int const expected = 8 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("VCA expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }

  int offset = 8;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0], "burn_dt_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1], "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2], "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3], "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }

  return q;
}

VCARelRequest ParseVCARelRequest(std::vector<std::string> const& f) {
  // VCAREL id dep_body arr_body state_dt_s scan_start_dt_s scan_end_dt_s samples
  //        rel_r_x rel_r_y rel_r_z rel_v_x rel_v_y rel_v_z
  //        n_impulses [impulse_dt_s dvx dvy dvz] * n
  if (f.size() < 15) {
    throw std::runtime_error(
        "VCAREL expects at least 15 TSV fields including command; got " +
        std::to_string(f.size()));
  }

  VCARelRequest q;
  q.id = f[1];
  q.dep_body = f[2];
  q.arr_body = f[3];

  q.state_dt_s = ParseDouble(f[4], "state_dt_s");
  q.scan_start_dt_s = ParseDouble(f[5], "scan_start_dt_s");
  q.scan_end_dt_s = ParseDouble(f[6], "scan_end_dt_s");
  q.samples = ParseInt(f[7], "samples");

  q.rel_r_m = Vec3{
      ParseDouble(f[8], "rel_r_x_m"),
      ParseDouble(f[9], "rel_r_y_m"),
      ParseDouble(f[10], "rel_r_z_m")};

  q.rel_v_m_s = Vec3{
      ParseDouble(f[11], "rel_v_x_m_s"),
      ParseDouble(f[12], "rel_v_y_m_s"),
      ParseDouble(f[13], "rel_v_z_m_s")};

  int const n = ParseInt(f[14], "n_impulses");
  int const expected = 15 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("VCAREL expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }

  int offset = 15;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0],
                    "impulse_dt_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1],
                         "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2],
                         "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3],
                         "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }

  return q;
}

VCARelNavRequest ParseVCARelNavRequest(std::vector<std::string> const& f) {
  // VCAREL_NAV id dep_body arr_body nav_body
  //            state_dt_s scan_start_dt_s scan_end_dt_s samples
  //            rel_r_x rel_r_y rel_r_z rel_v_x rel_v_y rel_v_z
  //            n_impulses [impulse_dt_s dvt dvn dvb] * n
  //
  // dvt/dvn/dvb são componentes Frenet/TNB no NavigationFrame centrado em nav_body.
  if (f.size() < 16) {
    throw std::runtime_error(
        "VCAREL_NAV expects at least 16 TSV fields including command; got " +
        std::to_string(f.size()));
  }

  VCARelNavRequest q;
  q.id = f[1];
  q.dep_body = f[2];
  q.arr_body = f[3];
  q.nav_body = f[4];

  q.state_dt_s = ParseDouble(f[5], "state_dt_s");
  q.scan_start_dt_s = ParseDouble(f[6], "scan_start_dt_s");
  q.scan_end_dt_s = ParseDouble(f[7], "scan_end_dt_s");
  q.samples = ParseInt(f[8], "samples");

  q.rel_r_m = Vec3{
      ParseDouble(f[9], "rel_r_x_m"),
      ParseDouble(f[10], "rel_r_y_m"),
      ParseDouble(f[11], "rel_r_z_m")};

  q.rel_v_m_s = Vec3{
      ParseDouble(f[12], "rel_v_x_m_s"),
      ParseDouble(f[13], "rel_v_y_m_s"),
      ParseDouble(f[14], "rel_v_z_m_s")};

  int const n = ParseInt(f[15], "n_impulses");
  int const expected = 16 + 4 * n;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("VCAREL_NAV expects " + std::to_string(expected) +
                             " TSV fields for n=" + std::to_string(n) +
                             "; got " + std::to_string(f.size()));
  }

  int offset = 16;
  for (int i = 0; i < n; ++i) {
    q.impulses.push_back(NavImpulse{
        ParseDouble(f[offset + 0],
                    "impulse_dt_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1],
                         "dvt_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2],
                         "dvn_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3],
                         "dvb_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }

  return q;
}

PropSampleRequest ParsePropSampleRequest(std::vector<std::string> const& f) {
  // PROPS id t0 t1 n_impulses n_samples x y z vx vy vz
  //       [burn_t dvx dvy dvz] * n_impulses [sample_t] * n_samples
  if (f.size() < 12) {
    throw std::runtime_error("PROPS expects at least 12 TSV fields including command; got " +
                             std::to_string(f.size()));
  }
  PropSampleRequest s;
  s.base.id = f[1];
  s.base.t0_s = ParseDouble(f[2], "t0_s");
  s.base.t1_s = ParseDouble(f[3], "t1_s");
  int const n_impulses = ParseInt(f[4], "n_impulses");
  int const n_samples = ParseInt(f[5], "n_samples");
  int const expected = 12 + 4 * n_impulses + n_samples;
  if (static_cast<int>(f.size()) != expected) {
    throw std::runtime_error("PROPS expects " + std::to_string(expected) +
                             " TSV fields for n_impulses=" + std::to_string(n_impulses) +
                             " n_samples=" + std::to_string(n_samples) +
                             "; got " + std::to_string(f.size()));
  }
  s.base.r0_m = Vec3{ParseDouble(f[6], "x_m"),
                     ParseDouble(f[7], "y_m"),
                     ParseDouble(f[8], "z_m")};
  s.base.v0_m_s = Vec3{ParseDouble(f[9], "vx_m_s"),
                       ParseDouble(f[10], "vy_m_s"),
                       ParseDouble(f[11], "vz_m_s")};
  int offset = 12;
  for (int i = 0; i < n_impulses; ++i) {
    s.base.impulses.push_back(Impulse{
        ParseDouble(f[offset + 0], "burn_t_s[" + std::to_string(i) + "]"),
        Vec3{ParseDouble(f[offset + 1], "dvx_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 2], "dvy_m_s[" + std::to_string(i) + "]"),
             ParseDouble(f[offset + 3], "dvz_m_s[" + std::to_string(i) + "]")}});
    offset += 4;
  }
  for (int i = 0; i < n_samples; ++i) {
    s.sample_times_s.push_back(ParseDouble(f[offset + i],
                                           "sample_t_s[" + std::to_string(i) + "]"));
  }
  return s;
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

VesselPropNResult PropagateVesselN(
    Plugin const* const plugin,
    VesselPropNRequest const& q) {
  VesselPropNResult r;
  r.id = q.id;
  r.vessel_guid = q.vessel_guid;

  try {
    ValidateVesselMonotonic(q);

    if (!plugin->HasVessel(q.vessel_guid)) {
      throw std::runtime_error("unknown vessel_guid: " + q.vessel_guid);
    }

    Vessel* const vessel = plugin->GetVessel(q.vessel_guid);

    auto const psychohistory = vessel->psychohistory();
    if (psychohistory == vessel->trajectory().segments().end() ||
        psychohistory->empty()) {
      throw std::runtime_error("vessel psychohistory is missing or empty");
    }

    auto const& initial_point = psychohistory->back();

    Instant const t0 = initial_point.time;
    DegreesOfFreedom<Barycentric> current =
        initial_point.degrees_of_freedom;
    Instant current_t = t0;

    auto parameters = vessel->prediction_adaptive_step_parameters();
    parameters.set_max_steps(std::max<std::int64_t>(parameters.max_steps(), 50000));

    r.t0_game_s = (t0 - plugin->GameEpoch()) / Second;
    r.t1_game_s = r.t0_game_s + q.final_dt_s;

    r.initial_r_m = ExtractPosition(current);
    r.initial_v_m_s = ExtractVelocity(current);
    r.initial_parent = ComputeParentSnapshot(plugin, vessel, t0, current);

    for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
      Impulse const& impulse = q.impulses[i];
      Instant const burn_t = t0 + impulse.burn_t_s * Second;

      DegreesOfFreedom<Barycentric> const before =
          FlowWithParameters(plugin,
                             parameters,
                             current_t,
                             current,
                             burn_t,
                             "vessel_leg_to_burn[" + std::to_string(i) + "]");

      Vec3 const burn_r = ExtractPosition(before);
      Vec3 const v_before = ExtractVelocity(before);
      Vec3 const v_after = Add(v_before, impulse.dv_m_s);

      r.burns.push_back(
          BurnSnapshot{impulse.burn_t_s, burn_r, v_before, v_after});

      current_t = burn_t;
      current = MakeDof(burn_r, v_after);
    }

    Instant const final_t = t0 + q.final_dt_s * Second;

    DegreesOfFreedom<Barycentric> const final_dof =
        FlowWithParameters(plugin,
                           parameters,
                           current_t,
                           current,
                           final_t,
                           "vessel_leg_to_final");

    r.final_r_m = ExtractPosition(final_dof);
    r.final_v_m_s = ExtractVelocity(final_dof);
    r.final_parent = ComputeParentSnapshot(plugin, vessel, final_t, final_dof);

    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}


VRelResult PropagateVRel(Plugin const* const plugin, VRelRequest const& q) {
  VRelResult r;
  r.id = q.id;
  r.vessel_guid = q.vessel_guid;
  r.reference_body = q.reference_body;

  try {
    ValidateRelativeImpulseTimes(q.final_dt_s, q.impulses, "VREL");

    if (!plugin->HasVessel(q.vessel_guid)) {
      throw std::runtime_error("unknown vessel_guid: " + q.vessel_guid);
    }

    Vessel* const vessel = plugin->GetVessel(q.vessel_guid);
    Celestial const& reference = *ResolveCelestial(plugin, q.reference_body);

    auto const psychohistory = vessel->psychohistory();
    if (psychohistory == vessel->trajectory().segments().end() ||
        psychohistory->empty()) {
      throw std::runtime_error("vessel psychohistory is missing or empty");
    }

    auto const& initial_point = psychohistory->back();

    Instant const t0 = initial_point.time;
    Instant current_t = t0;
    DegreesOfFreedom<Barycentric> current =
        initial_point.degrees_of_freedom;

    auto parameters = vessel->prediction_adaptive_step_parameters();
    parameters.set_max_steps(
        std::max<std::int64_t>(parameters.max_steps(), 50000));

    r.t0_game_s = (t0 - plugin->GameEpoch()) / Second;
    r.t1_game_s = r.t0_game_s + q.final_dt_s;

    for (int i = 0; i < static_cast<int>(q.impulses.size()); ++i) {
      Impulse const& impulse = q.impulses[i];
      Instant const burn_t = t0 + impulse.burn_t_s * Second;

      DegreesOfFreedom<Barycentric> const before =
          FlowWithParameters(plugin,
                             parameters,
                             current_t,
                             current,
                             burn_t,
                             "vrel_leg_to_burn[" + std::to_string(i) + "]");

      Vec3 const burn_r = ExtractPosition(before);
      Vec3 const v_before = ExtractVelocity(before);
      Vec3 const v_after = Add(v_before, impulse.dv_m_s);

      CheckImpulseApplied(impulse.dv_m_s,
                          v_before,
                          v_after,
                          "VREL burn[" + std::to_string(i) + "]");

      r.burns.push_back(
          BurnSnapshot{impulse.burn_t_s, burn_r, v_before, v_after});

      current_t = burn_t;
      current = MakeDof(burn_r, v_after);
    }

    Instant const final_t = t0 + q.final_dt_s * Second;
    DegreesOfFreedom<Barycentric> const final_dof =
        FlowWithParameters(plugin,
                           parameters,
                           current_t,
                           current,
                           final_t,
                           "vrel_leg_to_final");

    DegreesOfFreedom<Barycentric> const reference_dof =
        reference.current_degrees_of_freedom(final_t);

    RelativeSnapshot const rel =
      ComputeRelativeSnapshot(plugin, reference, final_t, final_dof);

    r.final_rel_r_m = rel.r_parent_m;
    r.final_rel_v_m_s = rel.v_parent_m_s;
    r.distance_m = rel.distance_m;
    r.speed_m_s = rel.speed_m_s;
    r.radial_velocity_m_s = rel.radial_velocity_m_s;

    r.final_abs_r_m = ExtractPosition(final_dof);
    r.final_abs_v_m_s = ExtractVelocity(final_dof);

    r.reference_abs_r_m = ExtractPosition(reference_dof);
    r.reference_abs_v_m_s = ExtractVelocity(reference_dof);

    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}


VCAResult PropagateVCA(Plugin const* const plugin, VCARequest const& q) {
  VCAResult r;
  r.id = q.id;
  r.vessel_guid = q.vessel_guid;
  r.target_body = q.target_body;
  r.samples = q.samples;

  try {
    ValidateVCA(q);

    VesselInitialContext const ctx =
        GetVesselInitialContext(plugin, q.vessel_guid, 50000);

    r.t0_game_s = (ctx.t0 - plugin->GameEpoch()) / Second;

    Celestial const& target = *ResolveCelestial(plugin, q.target_body);

    // Uniform scan.
    int best_i = 0;
    RelativeAtDt best;
    best.distance_m = std::numeric_limits<double>::infinity();

    for (int i = 0; i < q.samples; ++i) {
      double const u =
          static_cast<double>(i) / static_cast<double>(q.samples - 1);
      double const dt =
          q.scan_start_dt_s +
          u * (q.scan_end_dt_s - q.scan_start_dt_s);

      RelativeAtDt const candidate =
          EvaluateRelativeAtDt(plugin,
                               ctx,
                               target,
                               q.impulses,
                               dt,
                               "vca_scan[" + std::to_string(i) + "]");

      if (candidate.distance_m < best.distance_m) {
        best = candidate;
        best_i = i;
      }
    }

    // Bracket around the best sample. If the best point is at an edge, refine
    // only inside the neighbouring interval.
    double const step =
        (q.scan_end_dt_s - q.scan_start_dt_s) /
        static_cast<double>(q.samples - 1);

    double a = std::max(q.scan_start_dt_s,
                        q.scan_start_dt_s + (best_i - 1) * step);
    double b = std::min(q.scan_end_dt_s,
                        q.scan_start_dt_s + (best_i + 1) * step);

    // Golden-section refinement on distance(dt).
    double constexpr phi = 0.6180339887498948482;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);

    auto eval = [&](double const dt, std::string const& tag) {
      return EvaluateRelativeAtDt(plugin,
                                  ctx,
                                  target,
                                  q.impulses,
                                  dt,
                                  tag);
    };

    RelativeAtDt fc = eval(c, "vca_refine_c");
    RelativeAtDt fd = eval(d, "vca_refine_d");

    // Fixed iteration count: deterministic and enough for now.
    for (int iter = 0; iter < 32; ++iter) {
      if (fc.distance_m < fd.distance_m) {
        b = d;
        d = c;
        fd = fc;
        c = b - phi * (b - a);
        fc = eval(c, "vca_refine_c");
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + phi * (b - a);
        fd = eval(d, "vca_refine_d");
      }
    }

    RelativeAtDt refined =
        fc.distance_m < fd.distance_m ? fc : fd;

    // Compare refined with best scanned sample, in case the local bracket was
    // not unimodal or refinement degraded.
    if (best.distance_m <= refined.distance_m) {
      refined = best;
      r.ca_status = "scan_best";
    } else {
      r.ca_status = "refined";
    }

    r.ca_dt_s = refined.dt_s;
    r.ca_t_game_s = refined.game_s;
    r.ca_rel_r_m = refined.rel_r_m;
    r.ca_rel_v_m_s = refined.rel_v_m_s;
    r.ca_distance_m = refined.distance_m;
    r.ca_speed_m_s = refined.speed_m_s;
    r.ca_radial_velocity_m_s = refined.radial_velocity_m_s;

    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}


VCARelResult PropagateVCARel(
    Plugin const* const plugin,
    VCARelRequest const& q) {
  VCARelResult r;
  r.id = q.id;
  r.dep_body = q.dep_body;
  r.arr_body = q.arr_body;
  r.state_dt_s = q.state_dt_s;
  r.samples = q.samples;

  try {
    ValidateVCARel(q);

    RelativeParticleContext const ctx =
        GetRelativeParticleContext(plugin, q);

    Celestial const& arr = *ResolveCelestial(plugin, q.arr_body);

    r.state_t_game_s = (ctx.state_t - plugin->GameEpoch()) / Second;

    int best_i = 0;
    RelativeAtDt best;
    best.distance_m = std::numeric_limits<double>::infinity();

    for (int i = 0; i < q.samples; ++i) {
      double const u =
          static_cast<double>(i) / static_cast<double>(q.samples - 1);
      double const dt =
          q.scan_start_dt_s +
          u * (q.scan_end_dt_s - q.scan_start_dt_s);

      RelativeAtDt const candidate =
          EvaluateRelativeParticleAtDt(
              plugin,
              ctx,
              arr,
              q.impulses,
              dt,
              "vcarel_scan[" + std::to_string(i) + "]");

      if (candidate.distance_m < best.distance_m) {
        best = candidate;
        best_i = i;
      }
    }

    double const step =
        (q.scan_end_dt_s - q.scan_start_dt_s) /
        static_cast<double>(q.samples - 1);

    double a = std::max(q.scan_start_dt_s,
                        q.scan_start_dt_s + (best_i - 1) * step);
    double b = std::min(q.scan_end_dt_s,
                        q.scan_start_dt_s + (best_i + 1) * step);

    double constexpr phi = 0.6180339887498948482;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);

    auto eval = [&](double const dt, std::string const& tag) {
      return EvaluateRelativeParticleAtDt(
          plugin,
          ctx,
          arr,
          q.impulses,
          dt,
          tag);
    };

    RelativeAtDt fc = eval(c, "vcarel_refine_c");
    RelativeAtDt fd = eval(d, "vcarel_refine_d");

    for (int iter = 0; iter < 32; ++iter) {
      if (fc.distance_m < fd.distance_m) {
        b = d;
        d = c;
        fd = fc;
        c = b - phi * (b - a);
        fc = eval(c, "vcarel_refine_c");
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + phi * (b - a);
        fd = eval(d, "vcarel_refine_d");
      }
    }

    RelativeAtDt refined =
        fc.distance_m < fd.distance_m ? fc : fd;

    if (best.distance_m <= refined.distance_m) {
      refined = best;
      r.ca_status = "scan_best";
    } else {
      r.ca_status = "refined";
    }

    r.ca_dt_s = refined.dt_s;
    r.ca_t_game_s = refined.game_s;
    r.ca_rel_r_m = refined.rel_r_m;
    r.ca_rel_v_m_s = refined.rel_v_m_s;
    r.ca_distance_m = refined.distance_m;
    r.ca_speed_m_s = refined.speed_m_s;
    r.ca_radial_velocity_m_s = refined.radial_velocity_m_s;

    // Debug absolute states at CA.
    DegreesOfFreedom<Barycentric> const ca_particle_dof =
        PropagateRelativeParticleWithImpulsesToDt(
            plugin,
            ctx,
            q.impulses,
            refined.dt_s,
            "vcarel_debug_ca",
            &r.burns);

    Instant const ca_t = ctx.state_t + refined.dt_s * Second;
    EnsureEphemerisCovers(plugin, ca_t, "VCAREL debug arrival body");

    DegreesOfFreedom<Barycentric> const arr_dof =
        arr.current_degrees_of_freedom(ca_t);

    r.ca_abs_debug_r_m = ExtractPosition(ca_particle_dof);
    r.ca_abs_debug_v_m_s = ExtractVelocity(ca_particle_dof);
    r.arr_abs_debug_r_m = ExtractPosition(arr_dof);
    r.arr_abs_debug_v_m_s = ExtractVelocity(arr_dof);

    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}


VCARelNavResult PropagateVCARelNav(
    Plugin const* const plugin,
    VCARelNavRequest const& q) {
  VCARelNavResult r;
  r.id = q.id;
  r.dep_body = q.dep_body;
  r.arr_body = q.arr_body;
  r.nav_body = q.nav_body;
  r.state_dt_s = q.state_dt_s;
  r.samples = q.samples;

  try {
    ValidateVCARelNav(q);

    RelativeParticleContext const ctx =
        GetRelativeParticleContext(plugin, VCARelRequest{
            q.id,
            q.dep_body,
            q.arr_body,
            q.state_dt_s,
            q.scan_start_dt_s,
            q.scan_end_dt_s,
            q.samples,
            q.rel_r_m,
            q.rel_v_m_s,
            {}});

    Celestial const& arr = *ResolveCelestial(plugin, q.arr_body);
    Celestial const& nav_centre = *ResolveCelestial(plugin, q.nav_body);

    auto const navigation_frame =
        MakeBodyCentredNonRotatingNavFrame(plugin, nav_centre);

    r.state_t_game_s = (ctx.state_t - plugin->GameEpoch()) / Second;

    int best_i = 0;
    RelativeAtDt best;
    best.distance_m = std::numeric_limits<double>::infinity();

    for (int i = 0; i < q.samples; ++i) {
      double const u =
          static_cast<double>(i) / static_cast<double>(q.samples - 1);
      double const dt =
          q.scan_start_dt_s +
          u * (q.scan_end_dt_s - q.scan_start_dt_s);

      RelativeAtDt const candidate =
          EvaluateRelativeParticleNavAtDt(
              plugin,
              ctx,
              *navigation_frame,
              arr,
              q.impulses,
              dt,
              "vcarel_nav_scan[" + std::to_string(i) + "]");

      if (candidate.distance_m < best.distance_m) {
        best = candidate;
        best_i = i;
      }
    }

    double const step =
        (q.scan_end_dt_s - q.scan_start_dt_s) /
        static_cast<double>(q.samples - 1);

    double a = std::max(q.scan_start_dt_s,
                        q.scan_start_dt_s + (best_i - 1) * step);
    double b = std::min(q.scan_end_dt_s,
                        q.scan_start_dt_s + (best_i + 1) * step);

    double constexpr phi = 0.6180339887498948482;
    double c = b - phi * (b - a);
    double d = a + phi * (b - a);

    auto eval = [&](double const dt, std::string const& tag) {
      return EvaluateRelativeParticleNavAtDt(
          plugin,
          ctx,
          *navigation_frame,
          arr,
          q.impulses,
          dt,
          tag);
    };

    RelativeAtDt fc = eval(c, "vcarel_nav_refine_c");
    RelativeAtDt fd = eval(d, "vcarel_nav_refine_d");

    for (int iter = 0; iter < 32; ++iter) {
      if (fc.distance_m < fd.distance_m) {
        b = d;
        d = c;
        fd = fc;
        c = b - phi * (b - a);
        fc = eval(c, "vcarel_nav_refine_c");
      } else {
        a = c;
        c = d;
        fc = fd;
        d = a + phi * (b - a);
        fd = eval(d, "vcarel_nav_refine_d");
      }
    }

    RelativeAtDt refined =
        fc.distance_m < fd.distance_m ? fc : fd;

    if (best.distance_m <= refined.distance_m) {
      refined = best;
      r.ca_status = "scan_best";
    } else {
      r.ca_status = "refined";
    }

    r.ca_dt_s = refined.dt_s;
    r.ca_t_game_s = refined.game_s;
    r.ca_rel_r_m = refined.rel_r_m;
    r.ca_rel_v_m_s = refined.rel_v_m_s;
    r.ca_distance_m = refined.distance_m;
    r.ca_speed_m_s = refined.speed_m_s;
    r.ca_radial_velocity_m_s = refined.radial_velocity_m_s;

    // Debug absolute states and burn diagnostics at the selected CA.
    DegreesOfFreedom<Barycentric> const ca_particle_dof =
        PropagateRelativeParticleWithNavImpulsesToDt(
            plugin,
            ctx,
            *navigation_frame,
            q.impulses,
            refined.dt_s,
            "vcarel_nav_debug_ca",
            &r.burns);

    Instant const ca_t = ctx.state_t + refined.dt_s * Second;
    EnsureEphemerisCovers(plugin, ca_t, "VCAREL_NAV debug arrival body");

    DegreesOfFreedom<Barycentric> const arr_dof =
        arr.current_degrees_of_freedom(ca_t);

    r.ca_abs_debug_r_m = ExtractPosition(ca_particle_dof);
    r.ca_abs_debug_v_m_s = ExtractVelocity(ca_particle_dof);
    r.arr_abs_debug_r_m = ExtractPosition(arr_dof);
    r.arr_abs_debug_v_m_s = ExtractVelocity(arr_dof);

    r.status = "ok";
  } catch (std::exception const& e) {
    r.status = "error";
    r.message = e.what();
  }

  return r;
}


PropSampleResult PropagateSamples(Plugin const* const plugin, PropSampleRequest const& s) {
  PropSampleResult out;
  out.id = s.base.id;
  out.t0_s = s.base.t0_s;
  out.t1_s = s.base.t1_s;
  try {
    ValidateMonotonic(s.base);
    double previous = s.base.t0_s;
    for (double const sample_t : s.sample_times_s) {
      if (!(s.base.t0_s <= sample_t && sample_t <= s.base.t1_s)) {
        throw std::runtime_error("sample time outside [t0,t1]");
      }
      if (sample_t < previous) {
        throw std::runtime_error("sample times must be monotonic nondecreasing");
      }
      previous = sample_t;
    }

    // Simple and robust implementation: for each requested sample, replay the
    // impulse list from t0, truncated to impulses at or before sample_t.
    // This is intentionally not the fastest possible implementation, but it
    // preserves PROPN semantics exactly and is easy to validate.
    for (int i = 0; i < static_cast<int>(s.sample_times_s.size()); ++i) {
      double const sample_t = s.sample_times_s[i];
      PropNRequest q;
      q.id = s.base.id + ":sample" + std::to_string(i);
      q.t0_s = s.base.t0_s;
      q.t1_s = sample_t;
      q.r0_m = s.base.r0_m;
      q.v0_m_s = s.base.v0_m_s;
      for (Impulse const& impulse : s.base.impulses) {
        if (impulse.burn_t_s <= sample_t) {
          q.impulses.push_back(impulse);
        }
      }
      PropNResult const r = PropagateN(plugin, q);
      if (r.status != "ok") {
        throw std::runtime_error("sample propagation failed at index " +
                                 std::to_string(i) + ": " + r.message);
      }
      out.samples.push_back(StateSample{sample_t, r.final_r_m, r.final_v_m_s});
    }
    out.status = "ok";
  } catch (std::exception const& e) {
    out.status = "error";
    out.message = e.what();
  }
  return out;
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


void WriteOKVesselN(VesselPropNResult const& r) {
  std::cout << std::setprecision(17)
            << "OKVN"
            << '\t' << r.id
            << '\t' << r.vessel_guid
            << '\t' << r.t0_game_s
            << '\t' << r.t1_game_s
            << '\t' << r.burns.size();

  for (BurnSnapshot const& b : r.burns) {
    std::cout << '\t' << b.burn_t_s
              << '\t' << b.r_m.x << '\t' << b.r_m.y << '\t' << b.r_m.z
              << '\t' << b.v_before_m_s.x << '\t' << b.v_before_m_s.y << '\t' << b.v_before_m_s.z
              << '\t' << b.v_after_m_s.x << '\t' << b.v_after_m_s.y << '\t' << b.v_after_m_s.z;
  }

  std::cout
      << '\t' << r.initial_r_m.x
      << '\t' << r.initial_r_m.y
      << '\t' << r.initial_r_m.z
      << '\t' << r.initial_v_m_s.x
      << '\t' << r.initial_v_m_s.y
      << '\t' << r.initial_v_m_s.z

      << '\t' << r.initial_parent.r_parent_m.x
      << '\t' << r.initial_parent.r_parent_m.y
      << '\t' << r.initial_parent.r_parent_m.z
      << '\t' << r.initial_parent.v_parent_m_s.x
      << '\t' << r.initial_parent.v_parent_m_s.y
      << '\t' << r.initial_parent.v_parent_m_s.z
      << '\t' << r.initial_parent.distance_m
      << '\t' << r.initial_parent.speed_m_s
      << '\t' << r.initial_parent.radial_velocity_m_s

      << '\t' << r.final_r_m.x
      << '\t' << r.final_r_m.y
      << '\t' << r.final_r_m.z
      << '\t' << r.final_v_m_s.x
      << '\t' << r.final_v_m_s.y
      << '\t' << r.final_v_m_s.z

      << '\t' << r.final_parent.r_parent_m.x
      << '\t' << r.final_parent.r_parent_m.y
      << '\t' << r.final_parent.r_parent_m.z
      << '\t' << r.final_parent.v_parent_m_s.x
      << '\t' << r.final_parent.v_parent_m_s.y
      << '\t' << r.final_parent.v_parent_m_s.z
      << '\t' << r.final_parent.distance_m
      << '\t' << r.final_parent.speed_m_s
      << '\t' << r.final_parent.radial_velocity_m_s
      << '\n';

  std::cout.flush();
}


void WriteOKVRel(VRelResult const& r) {
  std::cout << std::setprecision(17)
            << "OKVR"
            << '\t' << r.id
            << '\t' << r.vessel_guid
            << '\t' << r.reference_body
            << '\t' << r.t0_game_s
            << '\t' << r.t1_game_s
            << '\t' << r.burns.size()

            << '\t' << r.final_rel_r_m.x
            << '\t' << r.final_rel_r_m.y
            << '\t' << r.final_rel_r_m.z

            << '\t' << r.final_rel_v_m_s.x
            << '\t' << r.final_rel_v_m_s.y
            << '\t' << r.final_rel_v_m_s.z

            << '\t' << r.distance_m
            << '\t' << r.speed_m_s
            << '\t' << r.radial_velocity_m_s

            << '\t' << r.final_abs_r_m.x
            << '\t' << r.final_abs_r_m.y
            << '\t' << r.final_abs_r_m.z

            << '\t' << r.final_abs_v_m_s.x
            << '\t' << r.final_abs_v_m_s.y
            << '\t' << r.final_abs_v_m_s.z

            << '\t' << r.reference_abs_r_m.x
            << '\t' << r.reference_abs_r_m.y
            << '\t' << r.reference_abs_r_m.z

            << '\t' << r.reference_abs_v_m_s.x
            << '\t' << r.reference_abs_v_m_s.y
            << '\t' << r.reference_abs_v_m_s.z;

  for (BurnSnapshot const& b : r.burns) {
    std::cout << '\t' << b.burn_t_s
              << '\t' << b.r_m.x
              << '\t' << b.r_m.y
              << '\t' << b.r_m.z
              << '\t' << b.v_before_m_s.x
              << '\t' << b.v_before_m_s.y
              << '\t' << b.v_before_m_s.z
              << '\t' << b.v_after_m_s.x
              << '\t' << b.v_after_m_s.y
              << '\t' << b.v_after_m_s.z;
  }

  std::cout << '\n';
  std::cout.flush();
}


void WriteOKVCA(VCAResult const& r) {
  std::cout << std::setprecision(17)
            << "OKCA"
            << '\t' << r.id
            << '\t' << r.vessel_guid
            << '\t' << r.target_body
            << '\t' << r.t0_game_s
            << '\t' << r.ca_dt_s
            << '\t' << r.ca_t_game_s

            << '\t' << r.ca_rel_r_m.x
            << '\t' << r.ca_rel_r_m.y
            << '\t' << r.ca_rel_r_m.z

            << '\t' << r.ca_rel_v_m_s.x
            << '\t' << r.ca_rel_v_m_s.y
            << '\t' << r.ca_rel_v_m_s.z

            << '\t' << r.ca_distance_m
            << '\t' << r.ca_speed_m_s
            << '\t' << r.ca_radial_velocity_m_s

            << '\t' << r.samples
            << '\t' << r.ca_status
            << '\n';

  std::cout.flush();
}


void WriteOKVCARel(VCARelResult const& r) {
  std::cout << std::setprecision(17)
            << "OKCAREL"
            << '\t' << r.id
            << '\t' << r.dep_body
            << '\t' << r.arr_body
            << '\t' << r.state_dt_s
            << '\t' << r.state_t_game_s
            << '\t' << r.ca_dt_s
            << '\t' << r.ca_t_game_s

            << '\t' << r.ca_rel_r_m.x
            << '\t' << r.ca_rel_r_m.y
            << '\t' << r.ca_rel_r_m.z

            << '\t' << r.ca_rel_v_m_s.x
            << '\t' << r.ca_rel_v_m_s.y
            << '\t' << r.ca_rel_v_m_s.z

            << '\t' << r.ca_distance_m
            << '\t' << r.ca_speed_m_s
            << '\t' << r.ca_radial_velocity_m_s

            << '\t' << r.samples
            << '\t' << r.ca_status

            << '\t' << r.ca_abs_debug_r_m.x
            << '\t' << r.ca_abs_debug_r_m.y
            << '\t' << r.ca_abs_debug_r_m.z

            << '\t' << r.ca_abs_debug_v_m_s.x
            << '\t' << r.ca_abs_debug_v_m_s.y
            << '\t' << r.ca_abs_debug_v_m_s.z

            << '\t' << r.arr_abs_debug_r_m.x
            << '\t' << r.arr_abs_debug_r_m.y
            << '\t' << r.arr_abs_debug_r_m.z

            << '\t' << r.arr_abs_debug_v_m_s.x
            << '\t' << r.arr_abs_debug_v_m_s.y
            << '\t' << r.arr_abs_debug_v_m_s.z

            << '\t' << r.burns.size();

  for (BurnSnapshot const& b : r.burns) {
    std::cout << '\t' << b.burn_t_s
              << '\t' << b.r_m.x
              << '\t' << b.r_m.y
              << '\t' << b.r_m.z
              << '\t' << b.v_before_m_s.x
              << '\t' << b.v_before_m_s.y
              << '\t' << b.v_before_m_s.z
              << '\t' << b.v_after_m_s.x
              << '\t' << b.v_after_m_s.y
              << '\t' << b.v_after_m_s.z;
  }

  std::cout << '\n';
  std::cout.flush();
}


void WriteOKVCARelNav(VCARelNavResult const& r) {
  std::cout << std::setprecision(17)
            << "OKCARELNAV"
            << '\t' << r.id
            << '\t' << r.dep_body
            << '\t' << r.arr_body
            << '\t' << r.nav_body
            << '\t' << r.state_dt_s
            << '\t' << r.state_t_game_s
            << '\t' << r.ca_dt_s
            << '\t' << r.ca_t_game_s

            << '\t' << r.ca_rel_r_m.x
            << '\t' << r.ca_rel_r_m.y
            << '\t' << r.ca_rel_r_m.z

            << '\t' << r.ca_rel_v_m_s.x
            << '\t' << r.ca_rel_v_m_s.y
            << '\t' << r.ca_rel_v_m_s.z

            << '\t' << r.ca_distance_m
            << '\t' << r.ca_speed_m_s
            << '\t' << r.ca_radial_velocity_m_s

            << '\t' << r.samples
            << '\t' << r.ca_status

            << '\t' << r.ca_abs_debug_r_m.x
            << '\t' << r.ca_abs_debug_r_m.y
            << '\t' << r.ca_abs_debug_r_m.z

            << '\t' << r.ca_abs_debug_v_m_s.x
            << '\t' << r.ca_abs_debug_v_m_s.y
            << '\t' << r.ca_abs_debug_v_m_s.z

            << '\t' << r.arr_abs_debug_r_m.x
            << '\t' << r.arr_abs_debug_r_m.y
            << '\t' << r.arr_abs_debug_r_m.z

            << '\t' << r.arr_abs_debug_v_m_s.x
            << '\t' << r.arr_abs_debug_v_m_s.y
            << '\t' << r.arr_abs_debug_v_m_s.z

            << '\t' << r.burns.size();

  for (NavBurnSnapshot const& b : r.burns) {
    std::cout << '\t' << b.burn_t_s

              << '\t' << b.burn_r_raw_m.x
              << '\t' << b.burn_r_raw_m.y
              << '\t' << b.burn_r_raw_m.z

              << '\t' << b.burn_v_before_raw_m_s.x
              << '\t' << b.burn_v_before_raw_m_s.y
              << '\t' << b.burn_v_before_raw_m_s.z

              << '\t' << b.dv_tnb_cmd_m_s.x
              << '\t' << b.dv_tnb_cmd_m_s.y
              << '\t' << b.dv_tnb_cmd_m_s.z

              << '\t' << b.tangent_raw.x
              << '\t' << b.tangent_raw.y
              << '\t' << b.tangent_raw.z

              << '\t' << b.normal_raw.x
              << '\t' << b.normal_raw.y
              << '\t' << b.normal_raw.z

              << '\t' << b.binormal_raw.x
              << '\t' << b.binormal_raw.y
              << '\t' << b.binormal_raw.z

              << '\t' << b.dv_raw_m_s.x
              << '\t' << b.dv_raw_m_s.y
              << '\t' << b.dv_raw_m_s.z

              << '\t' << b.burn_v_after_raw_m_s.x
              << '\t' << b.burn_v_after_raw_m_s.y
              << '\t' << b.burn_v_after_raw_m_s.z;
  }

  std::cout << '\n';
  std::cout.flush();
}


void WriteOKSamples(PropSampleResult const& r) {
  std::cout << std::setprecision(17)
            << "OKS\t" << r.id << '\t' << r.t0_s << '\t' << r.t1_s << '\t'
            << r.samples.size();
  for (StateSample const& s : r.samples) {
    std::cout << '\t' << s.t_s
              << '\t' << s.r_m.x << '\t' << s.r_m.y << '\t' << s.r_m.z
              << '\t' << s.v_m_s.x << '\t' << s.v_m_s.y << '\t' << s.v_m_s.z;
  }
  std::cout << '\n';
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

  std::cout << "READY\tprincipia_impulsive_particle_server_v0_5_targeter_vcarel_nav\n";
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
      if (cmd == "PROPS") {
        PropSampleRequest const q = ParsePropSampleRequest(fields);
        PropSampleResult const r = PropagateSamples(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKSamples(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }
      
      if (cmd == "VPROPN") {
        VesselPropNRequest const q = ParseVesselPropNRequest(fields);
        VesselPropNResult const r = PropagateVesselN(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKVesselN(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }

      if (cmd == "VREL") {
        VRelRequest const q = ParseVRelRequest(fields);
        VRelResult const r = PropagateVRel(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKVRel(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }

      if (cmd == "VCA") {
        VCARequest const q = ParseVCARequest(fields);
        VCAResult const r = PropagateVCA(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKVCA(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }

      if (cmd == "VCAREL") {
        VCARelRequest const q = ParseVCARelRequest(fields);
        VCARelResult const r = PropagateVCARel(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKVCARel(r);
        } else {
          WriteERR(r.id, r.message);
        }
        continue;
      }

      if (cmd == "VCAREL_NAV") {
        VCARelNavRequest const q = ParseVCARelNavRequest(fields);
        VCARelNavResult const r = PropagateVCARelNav(plugin.get(), q);
        if (r.status == "ok") {
          WriteOKVCARelNav(r);
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
