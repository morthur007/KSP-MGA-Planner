using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Globalization;
using UnityEngine;

[KSPAddon(KSPAddon.Startup.Flight, false)]
public sealed class MGAPrincipiaBridgeMissionEventDaemon : MonoBehaviour {
  private static MGAPrincipiaBridgeMissionEventDaemon instance = null;

  private static string lastRequestId = "";
  // IMPORTANT: Polling must use real time, not UT.  When a save is reloaded at an
  // earlier UT, a static UT gate can suppress polling for hours of in-game time.
  private static double nextPollRealtime = 0.0;
  private static double nextHeartbeatRealtime = 0.0;
  private const double PollPeriodS = 1.0;
  private const double HeartbeatPeriodS = 10.0;

  public void Awake() {
    // Reset volatile daemon state on every Flight scene start/reload.
    // Doing this before the instance check ensures that even duplicate components
    // reset the static variables before they are destroyed by the singleton check.
    lastRequestId = "";
    nextPollRealtime = 0.0;
    nextHeartbeatRealtime = 0.0;

    if (instance != null && instance != this) {
      Destroy(gameObject);
      return;
    }

    instance = this;
    DontDestroyOnLoad(gameObject);

    Log("MGAPrincipiaBridgeMissionEventDaemon.Awake reset_poll_state realtime=" +
        Time.realtimeSinceStartup.ToString("R", CultureInfo.InvariantCulture) +
        " ut=" + Planetarium.GetUniversalTime().ToString("R", CultureInfo.InvariantCulture));
  }

  public void OnDestroy() {
    if (instance == this) {
      instance = null;
    }
  }

  public void Update() {
    try {
      UpdateImpl();
    } catch (Exception e) {
      Log("UPDATE_EXCEPTION " + e);
    }
  }

  private void UpdateImpl() {
    if (!HighLogic.LoadedSceneIsFlight) return;

    double now = Time.realtimeSinceStartup;
    if (now >= nextHeartbeatRealtime) {
      nextHeartbeatRealtime = now + HeartbeatPeriodS;
      string heartbeatPath = Path.Combine(
          KSPUtil.ApplicationRootPath,
          "GameData/MGAPlanner/mission_event.json");
      Log("HEARTBEAT realtime=" + now.ToString("R", CultureInfo.InvariantCulture) +
          " ut=" + Planetarium.GetUniversalTime().ToString("R", CultureInfo.InvariantCulture) +
          " active_vessel=" + (FlightGlobals.ActiveVessel == null ? "null" : FlightGlobals.ActiveVessel.vesselName) +
          " event_exists=" + File.Exists(heartbeatPath));
    }

    if (FlightGlobals.ActiveVessel == null) return;

    if (now < nextPollRealtime) return;
    nextPollRealtime = now + PollPeriodS;

    string jsonPath = Path.Combine(
        KSPUtil.ApplicationRootPath,
        "GameData/MGAPlanner/mission_event.json");

    if (!File.Exists(jsonPath)) return;

    string json = "";
    try {
      json = File.ReadAllText(jsonPath);
    } catch (Exception e) {
      Log("EVENT_READ_EXCEPTION " + e.GetType().Name + ": " + e.Message);
      return;
    }

    MissionEvent ev = ParseMissionEvent(json);
    Log("POLL request_id=" + ev.request_id + " enabled=" + ev.enabled +
        " mode=" + ev.mode + " nonce=" + ev.force_nonce);

    if (!ev.enabled) {
      Log("SKIP disabled request_id=" + ev.request_id);
      return;
    }

    if (String.IsNullOrEmpty(ev.request_id)) {
      Log("EVENT_IGNORED missing request_id");
      return;
    }

    if (ev.mode == "reset_bridge_state") {
      lastRequestId = "";
      EventResult reset = new EventResult();
      reset.request_id = ev.request_id;
      reset.mode = ev.mode;
      reset.success = true;
      reset.status = "bridge_state_reset";
      reset.message = "Bridge polling state reset";
      WriteResult(reset);
      Log("RESET_BRIDGE_STATE done request_id=" + ev.request_id);
      return;
    }

    if (ev.request_id == lastRequestId && String.IsNullOrEmpty(ev.force_nonce)) {
      Log("SKIP already_processed request_id=" + ev.request_id);
      return;
    }

    lastRequestId = ev.request_id;

    EventResult result;
    try {
      result = Run(ev, jsonPath);
    } catch (Exception e) {
      Log("TOP_EXCEPTION " + e);
      result = EventResult.Fail(ev, "exception", e.ToString());
    }

    WriteResult(result);
  }

  private sealed class MissionEvent {
    public bool enabled = false;
    public string request_id = "";
    public string mode = "insert_navigation";
    public string dedupe_tag = "";
    public string force_nonce = "";
    public string vessel_guid = "";
    public int insert_index = -1;
    public int clone_from_index = -1;
    public double initial_time = Double.NaN;
    public double[] delta_v_navigation_m_s = null;
    public double[] delta_v_levela_m_s = null;
    public double placeholder_dv_m_s = 0.001;
    public double tolerance_time_s = 0.01;
    public double tolerance_dv_m_s = 1e-6;

    // Added fields from patch
    public bool ensure_flight_plan = true;
    public double plan_final_time = Double.NaN;
    public double mass_tonnes = 0.0;

    public string burn_template = "json";
    public double thrust_kN = 0.0;
    public double specific_impulse_s_g0 = 0.0;
    public bool is_inertially_fixed = false;

    public int frame_extension = -1;
    public int frame_centre_index = -1;
    public int frame_primary_index = -1;
    public int frame_secondary_index = -1;
    public bool frame_centre_from_active_body = false;
  }

  private sealed class EventResult {
    public string request_id = "";
    public bool success = false;
    public string status = "";
    public string message = "";
    public string mode = "";
    public int before = -1;
    public int after = -1;
    public int insert_index = -1;
    public int segments_before = -1;
    public int segments_after = -1;
    public double navigation_error_m_s = Double.NaN;
    public double levela_error_m_s = Double.NaN;

    public static EventResult Fail(MissionEvent ev, string status, string message) {
      return new EventResult {
        request_id = ev == null ? "" : ev.request_id,
        mode = ev == null ? "" : ev.mode,
        success = false,
        status = status,
        message = message
      };
    }
  }

  private static EventResult Run(MissionEvent ev, string jsonPath) {
    EventResult res = new EventResult();
    res.request_id = ev.request_id;
    res.mode = ev.mode;

    Log("============================================================");
    Log("MGA PRINCIPIA MISSION EVENT DAEMON REQUEST");
    Log("event_file=" + jsonPath);
    Log("request_id=" + ev.request_id);
    Log("enabled=" + ev.enabled);
    Log("mode=" + ev.mode);
    Log("dedupe_tag=" + ev.dedupe_tag);

    Vessel vessel = FlightGlobals.ActiveVessel;
    string activeGuid = vessel.id.ToString();

    Log("ut=" + Planetarium.GetUniversalTime());
    Log("active_vessel_name=" + vessel.vesselName);
    Log("active_vessel_guid_for_principia=" + activeGuid);
    Log("active_vessel_situation=" + vessel.situation);

    if (!String.IsNullOrEmpty(ev.vessel_guid) && ev.vessel_guid != activeGuid) {
      Log("FAIL vessel guid mismatch event=" + ev.vessel_guid + " active=" + activeGuid);
      res.success = false;
      res.status = "vessel_guid_mismatch";
      res.message = "event vessel_guid does not match active vessel";
      return res;
    }

    Assembly adapter = FindAdapterAssembly();
    if (adapter == null) {
      Log("FAIL no principia.ksp_plugin_adapter assembly");
      res.success = false;
      res.status = "no_adapter";
      res.message = "principia.ksp_plugin_adapter assembly not found";
      return res;
    }

    Type iface = adapter.GetType("principia.ksp_plugin_adapter.Interface");
    if (iface == null) {
      Log("FAIL no Interface type");
      res.success = false;
      res.status = "no_interface";
      res.message = "Interface type not found";
      return res;
    }

    IntPtr plugin = FindPluginPtr();
    Log("plugin_ptr=" + Ptr(plugin));
    if (plugin == IntPtr.Zero) {
      Log("FAIL no plugin IntPtr found");
      res.success = false;
      res.status = "no_plugin_ptr";
      res.message = "plugin_ IntPtr not found";
      return res;
    }

    bool exists = Convert.ToBoolean(InvokeExact(iface, "FlightPlanExists", plugin, activeGuid));
    Log("FlightPlanExists.before=" + exists);

    if (!exists && ev.ensure_flight_plan) {
      double finalTime = Double.IsNaN(ev.plan_final_time)
          ? ev.initial_time + 3600.0
          : ev.plan_final_time;

      if (ev.mass_tonnes <= 0.0) {
        Log("FAIL mass_tonnes required to create FlightPlan");
        res.success = false;
        res.status = "missing_mass_tonnes";
        res.message = "FlightPlanCreate requires mass_tonnes";
        return res;
      }

      Log("CREATE_FLIGHT_PLAN final_time=" + Num(finalTime) +
          " mass_tonnes=" + Num(ev.mass_tonnes));

      InvokeExact(iface, "FlightPlanCreate", plugin, activeGuid, finalTime, ev.mass_tonnes);
    }

    exists = Convert.ToBoolean(InvokeExact(iface, "FlightPlanExists", plugin, activeGuid));
    Log("FlightPlanExists.after=" + exists);

    if (!exists) {
      res.success = false;
      res.status = "no_flight_plan";
      res.message = "No existing Principia flight plan and creation failed/disabled";
      return res;
    }

    int before = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfManoeuvres", plugin, activeGuid));
    int segmentsBefore = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfSegments", plugin, activeGuid));
    res.before = before;
    res.segments_before = segmentsBefore;
    Log("BEFORE manoeuvres=" + before + " segments=" + segmentsBefore);

    if (before < 1 && ev.burn_template == "clone") {
      Log("FAIL clone requested but no existing manoeuvre exists");
      res.success = false;
      res.status = "clone_without_source";
      res.message = "burn_template=clone requires an existing manoeuvre";
      return res;
    }

    if (EventAlreadyPresent(iface, plugin, activeGuid, ev, before)) {
      Log("EVENT_ALREADY_PRESENT; not inserting again");
      res.success = true;
      res.status = "already_present";
      res.message = "Matching event already exists";
      res.after = before;
      res.segments_after = segmentsBefore;
      return res;
    }

    int insertIndex = ev.insert_index >= 0 ? ev.insert_index : before;
    if (insertIndex < 0 || insertIndex > before) {
      Log("FAIL invalid insert_index=" + insertIndex + " before=" + before);
      res.success = false;
      res.status = "invalid_insert_index";
      res.message = "insert_index out of range";
      return res;
    }
    res.insert_index = insertIndex;

    object burn = null;

    if (ev.burn_template == "clone") {
      int cloneIndex = ev.clone_from_index >= 0 ? ev.clone_from_index : before - 1;
      if (cloneIndex < 0 || cloneIndex >= before) {
        res.success = false;
        res.status = "invalid_clone_index";
        res.message = "clone_from_index out of range";
        return res;
      }

      object sourceManoeuvre =
          InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, cloneIndex);
      burn = GetField(sourceManoeuvre, "burn");
    } else if (ev.burn_template == "json") {
      burn = CreateBurnFromJson(adapter, ev);
    } else if (ev.burn_template == "json_frame_from_clone") {
      if (before < 1) {
        Log("WARN json_frame_from_clone requested but no manoeuvre exists; falling back to json frame");
        burn = CreateBurnFromJson(adapter, ev);
      } else {
        int cloneIndex = ev.clone_from_index >= 0 ? ev.clone_from_index : before - 1;
        object sourceManoeuvre =
            InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, cloneIndex);
        object sourceBurn = GetField(sourceManoeuvre, "burn");
        object sourceFrame = GetField(sourceBurn, "frame");

        burn = CreateBurnFromJson(adapter, ev);
        SetField(burn, "frame", sourceFrame);
      }
    } else {
      res.success = false;
      res.status = "unsupported_burn_template";
      res.message = "Unsupported burn_template: " + ev.burn_template;
      return res;
    }

    if (burn == null) {
      Log("FAIL could not create or clone burn");
      res.success = false;
      res.status = "burn_creation_failed";
      res.message = "Could not create or clone burn";
      return res;
    }

    SetField(burn, "initial_time", ev.initial_time);
    SetField(burn, "is_inertially_fixed", ev.is_inertially_fixed);

    double[] navToInsert = null;

    if (ev.mode == "insert_navigation") {
      if (ev.delta_v_navigation_m_s == null) {
        Log("FAIL insert_navigation requires delta_v_navigation_m_s");
        res.success = false;
        res.status = "missing_delta_v_navigation";
        res.message = "insert_navigation requires delta_v_navigation_m_s";
        return res;
      }

      navToInsert = ev.delta_v_navigation_m_s;
      SetField(burn, "delta_v", MakeXYZ(adapter, navToInsert));
    } else if (ev.mode == "insert_levela") {
      if (ev.delta_v_levela_m_s == null) {
        Log("FAIL insert_levela requires delta_v_levela_m_s");
        res.success = false;
        res.status = "missing_delta_v_levela";
        res.message = "insert_levela requires delta_v_levela_m_s";
        return res;
      }

      double[] placeholder = new double[] { ev.placeholder_dv_m_s, 0.0, 0.0 };
      SetField(burn, "delta_v", MakeXYZ(adapter, placeholder));
    } else {
      Log("FAIL unsupported mode=" + ev.mode);
      res.success = false;
      res.status = "unsupported_mode";
      res.message = "Unsupported mode: " + ev.mode;
      return res;
    }

    Log("INSERT mode=" + ev.mode);
    Log("INSERT index=" + insertIndex);
    Log("INSERT initial_time=" + ev.initial_time.ToString("R", CultureInfo.InvariantCulture));

    if (navToInsert != null) LogVec("INSERT delta_v_navigation_m_s", navToInsert);
    if (ev.delta_v_levela_m_s != null) LogVec("INSERT requested_delta_v_levela_m_s", ev.delta_v_levela_m_s);

    object status = InvokeExact(iface, "FlightPlanInsert", plugin, activeGuid, burn, insertIndex);
    DumpObjectFields("INSERT_STATUS", status);

    int afterInsert = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfManoeuvres", plugin, activeGuid));
    Log("AFTER_INSERT manoeuvres=" + afterInsert);

    if (afterInsert != before + 1) {
      Log("FAIL count did not increase after insert");
      res.success = false;
      res.status = "insert_count_failed";
      res.message = "FlightPlanInsert did not increase manoeuvre count";
      res.after = afterInsert;
      return res;
    }

    if (ev.mode == "insert_levela") {
      object tri = InvokeExact(iface, "FlightPlanGetManoeuvreFrenetTrihedron", plugin, activeGuid, insertIndex);
      double[][] triBasis = ReadTrihedron(tri);

      double[] rawDesired = LevelAToRaw(ev.delta_v_levela_m_s);
      double[] navDesired = RawToNavigation(rawDesired, triBasis);
      navToInsert = navDesired;

      LogVec("CONVERT levela_to_raw_m_s", rawDesired);
      LogVec("CONVERT raw_to_navigation_m_s", navDesired);

      object inserted = InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, insertIndex);
      object insertedBurn = GetField(inserted, "burn");
      SetField(insertedBurn, "delta_v", MakeXYZ(adapter, navDesired));
      SetField(insertedBurn, "initial_time", ev.initial_time);
      SetField(insertedBurn, "is_inertially_fixed", ev.is_inertially_fixed);

      object replaceStatus = InvokeExact(iface, "FlightPlanReplace", plugin, activeGuid, insertedBurn, insertIndex);
      DumpObjectFields("REPLACE_STATUS", replaceStatus);
    }

    int after = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfManoeuvres", plugin, activeGuid));
    int segmentsAfter = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfSegments", plugin, activeGuid));
    res.after = after;
    res.segments_after = segmentsAfter;
    Log("AFTER manoeuvres=" + after + " segments=" + segmentsAfter);

    object readback = InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, insertIndex);
    DumpObjectFields("READBACK_MANOEUVRE", readback);

    object readbackTri = InvokeExact(iface, "FlightPlanGetManoeuvreFrenetTrihedron", plugin, activeGuid, insertIndex);
    DumpObjectFields("READBACK_TRIHEDRON", readbackTri);

    object plotted = InvokeExact(iface, "FlightPlanGetManoeuvreInitialPlottedVelocity", plugin, activeGuid, insertIndex);
    DumpObjectFields("READBACK_INITIAL_PLOTTED_VELOCITY", plotted);

    RoundtripLog(readback, readbackTri, ev, res);

    Log("SUCCESS mission event inserted/read back");

    res.success = true;
    res.status = "ok";
    res.message = "Mission event inserted/read back";
    return res;
  }

  private static bool EventAlreadyPresent(Type iface, IntPtr plugin, string guid, MissionEvent ev, int n) {
    for (int i = 0; i < n; ++i) {
      object m = InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, guid, i);
      object burn = GetField(m, "burn");
      if (burn == null) continue;

      double t = GetDoubleField(burn, "initial_time");
      if (Math.Abs(t - ev.initial_time) > ev.tolerance_time_s) continue;

      object dvObj = GetField(burn, "delta_v");
      double[] nav = XYZToArray(dvObj);

      if (ev.mode == "insert_navigation" && ev.delta_v_navigation_m_s != null) {
        if (Norm(Sub(nav, ev.delta_v_navigation_m_s)) <= ev.tolerance_dv_m_s) {
          Log("DEDUP_MATCH index=" + i + " by navigation vector");
          return true;
        }
      }

      if (ev.mode == "insert_levela" && ev.delta_v_levela_m_s != null) {
        object tri = InvokeExact(iface, "FlightPlanGetManoeuvreFrenetTrihedron", plugin, guid, i);
        double[][] basis = ReadTrihedron(tri);
        double[] raw = NavigationToRaw(nav, basis);
        double[] levela = RawToLevelA(raw);

        if (Norm(Sub(levela, ev.delta_v_levela_m_s)) <= ev.tolerance_dv_m_s) {
          Log("DEDUP_MATCH index=" + i + " by levela vector");
          return true;
        }
      }
    }

    return false;
  }

  private static void RoundtripLog(object manoeuvre, object tri, MissionEvent ev, EventResult res) {
    object burn = GetField(manoeuvre, "burn");
    object dvObj = GetField(burn, "delta_v");
    double[] nav = XYZToArray(dvObj);
    double[][] basis = ReadTrihedron(tri);

    double[] raw = NavigationToRaw(nav, basis);
    double[] levela = RawToLevelA(raw);

    LogVec("ROUNDTRIP dv_navigation_m_s", nav);
    LogVec("ROUNDTRIP dv_raw_m_s", raw);
    LogVec("ROUNDTRIP dv_levela_m_s", levela);

    if (ev.mode == "insert_navigation" && ev.delta_v_navigation_m_s != null) {
      res.navigation_error_m_s = Norm(Sub(nav, ev.delta_v_navigation_m_s));
      Log("ROUNDTRIP navigation_error_m_s=" + res.navigation_error_m_s.ToString("R", CultureInfo.InvariantCulture));
    }

    if (ev.mode == "insert_levela" && ev.delta_v_levela_m_s != null) {
      res.levela_error_m_s = Norm(Sub(levela, ev.delta_v_levela_m_s));
      Log("ROUNDTRIP levela_error_m_s=" + res.levela_error_m_s.ToString("R", CultureInfo.InvariantCulture));
    }
  }

  private static MissionEvent ParseMissionEvent(string json) {
    MissionEvent ev = new MissionEvent();
    ev.enabled = GetBool(json, "enabled", false);
    ev.request_id = GetString(json, "request_id", "");
    ev.mode = GetString(json, "mode", ev.mode);
    ev.dedupe_tag = GetString(json, "dedupe_tag", "");
    ev.force_nonce = GetString(json, "force_nonce", "");
    ev.vessel_guid = GetString(json, "vessel_guid", "");
    ev.insert_index = GetInt(json, "insert_index", -1);
    ev.clone_from_index = GetInt(json, "clone_from_index", -1);
    ev.initial_time = GetDouble(json, "initial_time", Double.NaN);
    ev.placeholder_dv_m_s = GetDouble(json, "placeholder_dv_m_s", 0.001);
    ev.tolerance_time_s = GetDouble(json, "tolerance_time_s", 0.01);
    ev.tolerance_dv_m_s = GetDouble(json, "tolerance_dv_m_s", 1e-6);
    ev.delta_v_navigation_m_s = GetArray3(json, "delta_v_navigation_m_s");
    ev.delta_v_levela_m_s = GetArray3(json, "delta_v_levela_m_s");

    // Parsed new fields
    ev.ensure_flight_plan = GetBool(json, "ensure_flight_plan", true);
    ev.plan_final_time = GetDouble(json, "plan_final_time", Double.NaN);
    ev.mass_tonnes = GetDouble(json, "mass_tonnes", 0.0);

    ev.burn_template = GetString(json, "burn_template", "json");
    ev.thrust_kN = GetDouble(json, "thrust_kN", 0.0);
    ev.specific_impulse_s_g0 = GetDouble(json, "specific_impulse_s_g0", 0.0);
    ev.is_inertially_fixed = GetBool(json, "is_inertially_fixed", false);

    ev.frame_extension = GetInt(json, "frame_extension", -1);
    ev.frame_centre_index = GetInt(json, "frame_centre_index", -1);
    ev.frame_primary_index = GetInt(json, "frame_primary_index", -1);
    ev.frame_secondary_index = GetInt(json, "frame_secondary_index", -1);
    ev.frame_centre_from_active_body = GetBool(json, "frame_centre_from_active_body", false);

    return ev;
  }

  private static string GetString(string json, string key, string fallback) {
    Match m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*\"([^\"]*)\"");
    return m.Success ? m.Groups[1].Value : fallback;
  }

  private static bool GetBool(string json, string key, bool fallback) {
    Match m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(true|false)", RegexOptions.IgnoreCase);
    return m.Success ? Boolean.Parse(m.Groups[1].Value.ToLowerInvariant()) : fallback;
  }

  private static int GetInt(string json, string key, int fallback) {
    Match m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(-?\\d+)");
    return m.Success ? Int32.Parse(m.Groups[1].Value) : fallback;
  }

  private static double GetDouble(string json, string key, double fallback) {
    Match m = Regex.Match(json, "\"" + Regex.Escape(key) + "\"\\s*:\\s*(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)");
    return m.Success ? Double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture) : fallback;
  }

  private static double[] GetArray3(string json, string key) {
    Match m = Regex.Match(
        json,
        "\"" + Regex.Escape(key) + "\"\\s*:\\s*\\[\\s*" +
        "(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*,\\s*" +
        "(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*,\\s*" +
        "(-?(?:\\d+\\.?\\d*|\\.\\d+)(?:[eE][+-]?\\d+)?)\\s*\\]");
    if (!m.Success) return null;

    return new double[] {
      Double.Parse(m.Groups[1].Value, CultureInfo.InvariantCulture),
      Double.Parse(m.Groups[2].Value, CultureInfo.InvariantCulture),
      Double.Parse(m.Groups[3].Value, CultureInfo.InvariantCulture)
    };
  }

  private static object CreateBurnFromJson(Assembly adapter, MissionEvent ev) {
    Type burnType = adapter.GetType("principia.ksp_plugin_adapter.Burn");
    Type frameType = adapter.GetType("principia.ksp_plugin_adapter.NavigationFrameParameters");

    if (burnType == null || frameType == null) {
      throw new InvalidOperationException("Could not locate Principia Burn or NavigationFrameParameters type");
    }

    object burn = Activator.CreateInstance(burnType);
    object frame = Activator.CreateInstance(frameType);

    if (ev.frame_centre_from_active_body) {
      int activeIndex = ActiveBodyIndex();
      Log("FRAME_AUTO_CENTRE active_body=" + ActiveBodyName() + " ksp_body_index=" + activeIndex);
      if (activeIndex >= 0) ev.frame_centre_index = activeIndex;
    }

    if (ev.frame_extension <= 0) {
      throw new InvalidOperationException(
          "Invalid frame_extension=" + ev.frame_extension +
          ". Principia requires protobuf extension field number 6000..6003 for navigation frames.");
    }

    Log("FRAME_JSON extension=" + ev.frame_extension +
        " centre_index=" + ev.frame_centre_index +
        " primary_index=" + ev.frame_primary_index +
        " secondary_index=" + ev.frame_secondary_index);

    SetField(frame, "extension", ev.frame_extension);
    SetField(frame, "centre_index", ev.frame_centre_index);
    SetField(frame, "primary_index", ev.frame_primary_index);
    SetField(frame, "secondary_index", ev.frame_secondary_index);

    SetField(burn, "thrust_in_kilonewtons", ev.thrust_kN);
    SetField(burn, "specific_impulse_in_seconds_g0", ev.specific_impulse_s_g0);
    SetField(burn, "frame", frame);
    SetField(burn, "initial_time", ev.initial_time);
    SetField(burn, "delta_v", MakeXYZ(adapter, new double[] { ev.placeholder_dv_m_s, 0.0, 0.0 }));
    SetField(burn, "is_inertially_fixed", ev.is_inertially_fixed);

    return burn;
  }

  private static string ActiveBodyName() {
    try {
      if (FlightGlobals.ActiveVessel == null) return "null";
      CelestialBody b = FlightGlobals.ActiveVessel.mainBody;
      if (b == null && FlightGlobals.ActiveVessel.orbit != null) b = FlightGlobals.ActiveVessel.orbit.referenceBody;
      return b == null ? "null" : b.bodyName;
    } catch { return "exception"; }
  }

  private static int ActiveBodyIndex() {
    try {
      if (FlightGlobals.ActiveVessel == null) return -1;
      CelestialBody body = FlightGlobals.ActiveVessel.mainBody;
      if (body == null && FlightGlobals.ActiveVessel.orbit != null) body = FlightGlobals.ActiveVessel.orbit.referenceBody;
      if (body == null || FlightGlobals.Bodies == null) return -1;
      for (int i = 0; i < FlightGlobals.Bodies.Count; ++i) {
        if (System.Object.ReferenceEquals(FlightGlobals.Bodies[i], body)) return i;
        if (FlightGlobals.Bodies[i] != null && FlightGlobals.Bodies[i].bodyName == body.bodyName) return i;
      }
      return -1;
    } catch (Exception e) {
      Log("ACTIVE_BODY_INDEX_EXCEPTION " + e.GetType().Name + ": " + e.Message);
      return -1;
    }
  }

  private static Assembly FindAdapterAssembly() {
    foreach (Assembly a in AppDomain.CurrentDomain.GetAssemblies()) {
      string name = Safe(() => a.GetName().Name);
      if (name == "principia.ksp_plugin_adapter") return a;
    }
    return null;
  }

  private static IntPtr FindPluginPtr() {
    foreach (MonoBehaviour mb in UnityEngine.Object.FindObjectsOfType<MonoBehaviour>()) {
      if (mb == null) continue;

      Type t = mb.GetType();
      if (TypeName(t) != "principia.ksp_plugin_adapter.PrincipiaPluginAdapter") continue;

      FieldInfo f = FindFieldRecursive(t, "plugin_");
      if (f == null) return IntPtr.Zero;

      object value = f.GetValue(mb);
      if (value is IntPtr) return (IntPtr)value;
    }
    return IntPtr.Zero;
  }

  private static object InvokeExact(Type type, string name, params object[] args) {
    try {
      MethodInfo m = type.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static)
                         .FirstOrDefault(x => x.Name == name && x.GetParameters().Length == args.Length);
      if (m == null) {
        Log("MISSING_METHOD " + name);
        return null;
      }

      object result = m.Invoke(null, args);
      Log("INVOKE_OK " + name + " => " + ObjString(result));
      return result;
    } catch (TargetInvocationException e) {
      Log("INVOKE_TARGET_EXCEPTION " + name + " inner=" +
          (e.InnerException == null ? e.ToString() : e.InnerException.ToString()));
      return null;
    } catch (Exception e) {
      Log("INVOKE_EXCEPTION " + name + " " + e.GetType().Name + ": " + e.Message);
      return null;
    }
  }

  private static object MakeXYZ(Assembly adapter, double[] v) {
    Type xyzType = adapter.GetType("principia.ksp_plugin_adapter.XYZ");
    object xyz = Activator.CreateInstance(xyzType);
    SetField(xyz, "x", v[0]);
    SetField(xyz, "y", v[1]);
    SetField(xyz, "z", v[2]);
    return xyz;
  }

  private static double[] XYZToArray(object xyz) {
    return new double[] {
      GetDoubleField(xyz, "x"),
      GetDoubleField(xyz, "y"),
      GetDoubleField(xyz, "z")
    };
  }

  private static double[][] ReadTrihedron(object tri) {
    return new double[][] {
      XYZToArray(GetField(tri, "tangent")),
      XYZToArray(GetField(tri, "normal")),
      XYZToArray(GetField(tri, "binormal"))
    };
  }

  private static double[] NavigationToRaw(double[] nav, double[][] basis) {
    return Add(Add(Scale(nav[0], basis[0]), Scale(nav[1], basis[1])), Scale(nav[2], basis[2]));
  }

  private static double[] RawToNavigation(double[] raw, double[][] basis) {
    return new double[] {
      Dot(raw, basis[0]),
      Dot(raw, basis[1]),
      Dot(raw, basis[2])
    };
  }

  private static double[] RawToLevelA(double[] raw) {
    return new double[] { -raw[1], raw[2], raw[0] };
  }

  private static double[] LevelAToRaw(double[] levela) {
    return new double[] { levela[2], -levela[0], levela[1] };
  }

  private static double Dot(double[] a, double[] b) {
    return a[0]*b[0] + a[1]*b[1] + a[2]*b[2];
  }

  private static double[] Add(double[] a, double[] b) {
    return new double[] { a[0]+b[0], a[1]+b[1], a[2]+b[2] };
  }

  private static double[] Sub(double[] a, double[] b) {
    return new double[] { a[0]-b[0], a[1]-b[1], a[2]-b[2] };
  }

  private static double[] Scale(double s, double[] v) {
    return new double[] { s*v[0], s*v[1], s*v[2] };
  }

  private static double Norm(double[] v) {
    return Math.Sqrt(Dot(v, v));
  }

  private static FieldInfo FindFieldRecursive(Type type, string name) {
    for (Type t = type; t != null; t = t.BaseType) {
      FieldInfo f = t.GetField(name, BindingFlags.Public | BindingFlags.NonPublic |
                                     BindingFlags.Instance | BindingFlags.Static |
                                     BindingFlags.DeclaredOnly);
      if (f != null) return f;
    }
    return null;
  }

  private static object GetField(object obj, string name) {
    if (obj == null) return null;
    FieldInfo f = FindFieldRecursive(obj.GetType(), name);
    if (f == null) return null;
    return f.GetValue(obj);
  }

  private static void SetField(object obj, string name, object value) {
    FieldInfo f = FindFieldRecursive(obj.GetType(), name);
    if (f == null) {
      Log("SET_FIELD_MISSING " + TypeName(obj.GetType()) + "." + name);
      return;
    }
    f.SetValue(obj, value);
  }

  private static double GetDoubleField(object obj, string name) {
    object value = GetField(obj, name);
    if (value == null) return Double.NaN;
    return Convert.ToDouble(value);
  }

  private static void DumpObjectFields(string label, object obj) {
    if (obj == null) {
      Log(label + " = null");
      return;
    }

    Type t = obj.GetType();
    Log(label + " type=" + TypeName(t) + " value=" + ObjString(obj));

    foreach (FieldInfo f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic |
                                        BindingFlags.Instance | BindingFlags.DeclaredOnly)) {
      object value = null;
      bool got = false;
      try {
        value = f.GetValue(obj);
        got = true;
      } catch {}

      Log(label + "." + f.Name + " : " + TypeName(f.FieldType) +
          (got ? " = " + ObjString(value) : ""));

      if (got && value != null && !IsScalar(value.GetType())) {
        DumpNestedFields(label + "." + f.Name, value);
      }
    }
  }

  private static void DumpNestedFields(string label, object obj) {
    Type t = obj.GetType();

    foreach (FieldInfo f in t.GetFields(BindingFlags.Public | BindingFlags.NonPublic |
                                        BindingFlags.Instance | BindingFlags.DeclaredOnly)) {
      object value = null;
      bool got = false;
      try {
        value = f.GetValue(obj);
        got = true;
      } catch {}

      Log(label + "." + f.Name + " : " + TypeName(f.FieldType) +
          (got ? " = " + ObjString(value) : ""));
    }
  }

  private static bool IsScalar(Type t) {
    return t.IsPrimitive || t == typeof(string) || t == typeof(IntPtr) ||
           t == typeof(decimal) || t.IsEnum;
  }

  private static void LogVec(string label, double[] v) {
    Log(label + "=[" + Num(v[0]) + "," + Num(v[1]) + "," + Num(v[2]) + "] norm=" + Num(Norm(v)));
  }

  private static string ObjString(object o) {
    if (o == null) return "null";
    if (o is IntPtr) return Ptr((IntPtr)o);
    if (o is double) return Num((double)o);
    return o.ToString();
  }

  private static string Ptr(IntPtr p) {
    if (p == IntPtr.Zero) return "0x0";
    return "0x" + p.ToInt64().ToString("x");
  }

  private static string TypeName(Type t) {
    if (t == null) return "";
    return t.FullName ?? t.Name;
  }

  private static string Safe(Func<string> f) {
    try { return f() ?? ""; } catch { return ""; }
  }

  private static string Num(double x) {
    return x.ToString("R", CultureInfo.InvariantCulture);
  }

  private static string JsonNum(double x) {
    if (Double.IsNaN(x) || Double.IsInfinity(x)) return "null";
    return Num(x);
  }

  private static string JsonEscape(string s) {
    if (s == null) return "";
    return s.Replace("\\", "\\\\").Replace("\"", "\\\"");
  }

  private static void WriteResult(EventResult r) {
    try {
      string path = Path.Combine(KSPUtil.ApplicationRootPath, "GameData/MGAPlanner/mission_event_result.json");
      string text =
        "{\n" +
        "  \"request_id\": \"" + JsonEscape(r.request_id) + "\",\n" +
        "  \"success\": " + (r.success ? "true" : "false") + ",\n" +
        "  \"status\": \"" + JsonEscape(r.status) + "\",\n" +
        "  \"message\": \"" + JsonEscape(r.message) + "\",\n" +
        "  \"mode\": \"" + JsonEscape(r.mode) + "\",\n" +
        "  \"before\": " + r.before + ",\n" +
        "  \"after\": " + r.after + ",\n" +
        "  \"insert_index\": " + r.insert_index + ",\n" +
        "  \"segments_before\": " + r.segments_before + ",\n" +
        "  \"segments_after\": " + r.segments_after + ",\n" +
        "  \"navigation_error_m_s\": " + JsonNum(r.navigation_error_m_s) + ",\n" +
        "  \"levela_error_m_s\": " + JsonNum(r.levela_error_m_s) + "\n" +
        "}\n";
      File.WriteAllText(path, text);
      Log("WROTE_RESULT " + path);
    } catch (Exception e) {
      Log("WRITE_RESULT_EXCEPTION " + e.GetType().Name + ": " + e.Message);
    }
  }

  private static void Log(string msg) {
    string line = "[MGAPrincipiaBridge] " + msg;
    Debug.Log(line);

    try {
      string path = Path.Combine(KSPUtil.ApplicationRootPath, "MGAPrincipiaBridge_probe.log");
      File.AppendAllText(path, DateTime.Now.ToString("o") + " " + line + "\n");
    } catch {}
  }
}
