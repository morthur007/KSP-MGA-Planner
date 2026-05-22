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

    public bool require_warp_close = true;
    public double max_lead_before_insert_s = 600.0;
    public bool reject_long_plan_before_first_burn = true;
    public double max_first_plan_duration_s = 900.0;
    public bool require_status_ok = true;
    public bool rollback_on_status_error = true;
    public bool auto_sort_insert_index = true;
    public bool delete_existing_flight_plan = false;

    // Live navigation snapshot v0.1.
    public string snapshot_output_path = "";
    public bool snapshot_include_ksp_bodies = true;
    public bool snapshot_include_debug = true;
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
    // Optional preformatted JSON fields appended to mission_event_result.json.
    public string extra_json = "";

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

  private static int StatusError(object status) {
    if (status == null) return -9999;
    object v = GetField(status, "error");
    if (v == null) return -9999;
    return Convert.ToInt32(v);
  }

  private static string StatusMessage(object status) {
    if (status == null) return "null status";
    object v = GetField(status, "message");
    return v == null ? null : v.ToString();
  }

  private static bool FlightPlanExists(Type iface, object plugin, string vesselGuid) {
    object res = InvokeExact(iface, "FlightPlanExists", plugin, vesselGuid);
    if (res == null) return false;
    return Convert.ToBoolean(res);
  }

  private static int FlightPlanManoeuvreCount(Type iface, object plugin, string vesselGuid) {
    object n = InvokeExact(iface, "FlightPlanNumberOfManoeuvres", plugin, vesselGuid);
    if (n == null) return 0;
    return Convert.ToInt32(n);
  }

  private static void TryRemoveManoeuvre(Type iface, object plugin, string vesselGuid, int index) {
    try {
      Log("ROLLBACK remove manoeuvre index=" + index);
      InvokeExact(iface, "FlightPlanRemove", plugin, vesselGuid, index);
    } catch (Exception e) {
      Log("ROLLBACK_FAILED index=" + index + " exception=" + e);
    }
  }

  private static void RemoveAllManoeuvres(Type iface, object plugin, string vesselGuid) {
    int count = FlightPlanManoeuvreCount(iface, plugin, vesselGuid);
    Log("REMOVE_ALL_MANOEUVRES before=" + count);

    for (int i = count - 1; i >= 0; --i) {
      TryRemoveManoeuvre(iface, plugin, vesselGuid, i);
    }

    int after = FlightPlanManoeuvreCount(iface, plugin, vesselGuid);
    Log("REMOVE_ALL_MANOEUVRES after=" + after);
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

    if (ev.mode == "export_active_vessel_state") {
      return ExportActiveVesselState(ev, vessel);
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

    if (ev.mode == "dump_adapter_methods") {
      return DumpAdapterMethods(ev, iface);
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

    if (ev.mode == "export_principia_vessel_state") {
      return ExportPrincipiaVesselState(ev, adapter, iface, plugin, vessel);
    }

    if (ev.mode == "principia_live_navigation_snapshot_v0_1" ||
        ev.mode == "export_live_navigation_snapshot") {
      return ExportLiveNavigationSnapshot(ev, adapter, iface, plugin, vessel);
    }

    bool exists = FlightPlanExists(iface, plugin, activeGuid);
    Log("FlightPlanExists.initial=" + exists);

    if (ev.delete_existing_flight_plan && exists) {
      Log("DELETE_EXISTING_FLIGHT_PLAN requested");
      InvokeExact(iface, "FlightPlanDelete", plugin, activeGuid);
      exists = FlightPlanExists(iface, plugin, activeGuid);
      Log("FlightPlanExists.after_delete=" + exists);
    }

    if (ev.mode == "delete_flight_plan") {
      if (exists) {
        Log("DELETE_FLIGHT_PLAN");
        InvokeExact(iface, "FlightPlanDelete", plugin, activeGuid);
      }
      res.success = true;
      res.status = "ok";
      res.message = "FlightPlan deleted";
      return res;
    }

    if (ev.mode == "remove_all_manoeuvres") {
      if (!exists) {
        res.success = true;
        res.status = "ok";
        res.message = "No FlightPlan exists";
        return res;
      }
      RemoveAllManoeuvres(iface, plugin, activeGuid);
      res.success = true;
      res.status = "ok";
      res.message = "All manoeuvres removed";
      return res;
    }

    if (ev.mode == "set_final_time_only") {
      if (!exists) {
        res.success = false;
        res.status = "no_flight_plan";
        res.message = "Cannot set final time: no FlightPlan";
        return res;
      }
      Log("SET_FINAL_TIME_ONLY final_time=" + ev.plan_final_time.ToString("R"));
      InvokeExact(iface, "FlightPlanSetDesiredFinalTime", plugin, activeGuid, ev.plan_final_time);
      res.success = true;
      res.status = "ok";
      res.message = "Desired final time updated";
      return res;
    }

    double nowUt = Planetarium.GetUniversalTime();
    double leadToBurn = ev.initial_time - nowUt;

    if (ev.require_warp_close && leadToBurn > ev.max_lead_before_insert_s) {
      res.success = false;
      res.status = "warp_required";
      res.message = "Refusing to insert burn " + leadToBurn.ToString("F1") +
                    " s before initial_time. Warp closer first. " +
                    "max_lead_before_insert_s=" + ev.max_lead_before_insert_s.ToString("F1");
      return res;
    }

    if (!exists && ev.reject_long_plan_before_first_burn) {
      double testFinalTime = Double.IsNaN(ev.plan_final_time)
          ? ev.initial_time + 3600.0
          : ev.plan_final_time;

      if (testFinalTime > ev.initial_time + ev.max_first_plan_duration_s) {
        res.success = false;
        res.status = "plan_too_long_for_first_burn";
        res.message = "Refusing to create first FlightPlan with final_time far beyond first burn. " +
                      "Use staged insert: first plan_final_time <= initial_time + " +
                      ev.max_first_plan_duration_s.ToString("F1") + " s, then extend later.";
        return res;
      }
    }

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

    exists = FlightPlanExists(iface, plugin, activeGuid);
    Log("FlightPlanExists.after=" + exists);

    if (!exists) {
      res.success = false;
      res.status = "no_flight_plan";
      res.message = "No existing Principia flight plan and creation failed/disabled";
      return res;
    }

    int before = FlightPlanManoeuvreCount(iface, plugin, activeGuid);
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

    int insertIndex = ev.insert_index;

    if (ev.auto_sort_insert_index) {
      insertIndex = before;
      for (int i = 0; i < before; ++i) {
        object m = InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, i);
        object existingBurn = GetField(m, "burn");
        double t = GetDoubleField(existingBurn, "initial_time");
        if (ev.initial_time < t) {
          insertIndex = i;
          break;
        }
      }
      Log("AUTO_SORT_INSERT_INDEX requested=" + ev.insert_index +
          " resolved=" + insertIndex +
          " before=" + before);
    } else {
      if (insertIndex < 0) {
        insertIndex = before;
      }
    }

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
    
    int insertError = StatusError(status);
    string insertMessage = StatusMessage(status);
    Log("INSERT_STATUS.error=" + insertError);
    Log("INSERT_STATUS.message=" + insertMessage);
    DumpObjectFields("INSERT_STATUS_OBJ", status);

    int afterInsert = FlightPlanManoeuvreCount(iface, plugin, activeGuid);
    Log("AFTER_INSERT manoeuvres=" + afterInsert);

    if (ev.require_status_ok && insertError != 0) {
      Log("FAIL insert status error=" + insertError + " message=" + insertMessage);

      if (ev.rollback_on_status_error && afterInsert > before) {
        int rollbackIndex = Math.Min(insertIndex, afterInsert - 1);
        TryRemoveManoeuvre(iface, plugin, activeGuid, rollbackIndex);

        int afterRollback = FlightPlanManoeuvreCount(iface, plugin, activeGuid);
        Log("ROLLBACK_AFTER_INSERT_STATUS_ERROR before=" + afterInsert + " after=" + afterRollback);
      }

      res.success = false;
      res.status = "insert_status_error";
      res.message = insertMessage;
      res.before = before;
      res.after = FlightPlanManoeuvreCount(iface, plugin, activeGuid);
      res.insert_index = insertIndex;
      return res;
    }

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

    int after = FlightPlanManoeuvreCount(iface, plugin, activeGuid);
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


  private static EventResult ExportPrincipiaVesselState(MissionEvent ev, Assembly adapter, Type iface, IntPtr plugin, Vessel vessel) {
    EventResult res = new EventResult();
    res.request_id = ev.request_id;
    res.mode = ev.mode;
    res.success = true;
    res.status = "ok";
    res.message = "Active vessel state exported from Principia adapter";
    res.extra_json = "  \"active_vessel_state\": " + PrincipiaVesselStateJson(adapter, iface, plugin, vessel);
    Log("EXPORT_PRINCIPIA_VESSEL_STATE ok vessel=" + (vessel == null ? "null" : vessel.vesselName));
    return res;
  }

  private static EventResult ExportLiveNavigationSnapshot(
      MissionEvent ev,
      Assembly adapter,
      Type iface,
      IntPtr plugin,
      Vessel vessel) {
    EventResult res = new EventResult();
    res.request_id = ev.request_id;
    res.mode = ev.mode;
    res.success = true;
    res.status = "ok";

    string snapshot = LiveNavigationSnapshotJson(ev, adapter, iface, plugin, vessel);
    string snapshotPath = SnapshotOutputPath(ev);

    try {
      string dir = Path.GetDirectoryName(snapshotPath);
      if (!String.IsNullOrEmpty(dir)) Directory.CreateDirectory(dir);
      File.WriteAllText(snapshotPath, snapshot);
    } catch (Exception e) {
      res.success = false;
      res.status = "snapshot_write_failed";
      res.message = e.GetType().Name + ": " + e.Message;
      res.extra_json =
          "  \"live_navigation_snapshot_path\": \"" + JsonEscape(snapshotPath) + "\"";
      Log("EXPORT_LIVE_NAVIGATION_SNAPSHOT_WRITE_FAILED " + e);
      return res;
    }

    res.message = "Live navigation snapshot v0.1 exported";
    res.extra_json =
        "  \"live_navigation_snapshot_path\": \"" + JsonEscape(snapshotPath) + "\",\n" +
        "  \"live_navigation_snapshot\": " + snapshot;
    Log("EXPORT_LIVE_NAVIGATION_SNAPSHOT ok path=" + snapshotPath);
    return res;
  }

  private static string SnapshotOutputPath(MissionEvent ev) {
    if (!String.IsNullOrEmpty(ev.snapshot_output_path)) {
      if (Path.IsPathRooted(ev.snapshot_output_path)) {
        return ev.snapshot_output_path;
      }
      return Path.Combine(KSPUtil.ApplicationRootPath, ev.snapshot_output_path);
    }

    return Path.Combine(
        KSPUtil.ApplicationRootPath,
        "GameData/MGAPlanner/principia_live_navigation_snapshot_v0_1.json");
  }

  private static string LiveNavigationSnapshotJson(
      MissionEvent ev,
      Assembly adapter,
      Type iface,
      IntPtr plugin,
      Vessel vessel) {
    double ut = Planetarium.GetUniversalTime();
    string vesselJson = PrincipiaVesselStateJson(adapter, iface, plugin, vessel);

    string bodiesJson = ev.snapshot_include_ksp_bodies
        ? KspBodiesSnapshotJson(ut)
        : "[]";

    string activeBody = ActiveBodyName();
    int activeBodyIndex = ActiveBodyIndex();

    string text =
      "{\n" +
      "  \"schema\": \"principia_live_navigation_snapshot_v0_1\",\n" +
      "  \"source\": \"MGAPrincipiaBridgeMissionEventDaemon\",\n" +
      "  \"created_unix_utc_s\": " + JsonNum((DateTime.UtcNow - new DateTime(1970, 1, 1)).TotalSeconds) + ",\n" +
      "  \"t_game_s\": " + JsonNum(ut) + ",\n" +
      "  \"t_spice_s\": " + JsonNum(ut) + ",\n" +
      "  \"time_note\": \"t_spice_s equals game UT in this bridge snapshot; Python may add a TDB/J2000 anchor offset.\",\n" +
      "  \"frame_contract\": {\n" +
      "    \"principia_raw_to_levela\": \"(X,Y,Z)->(-Y,+Z,+X)\",\n" +
      "    \"levela_to_principia_raw\": \"(X,Y,Z)->(+Z,-X,+Y)\",\n" +
      "    \"vessel_state_frame\": \"Principia adapter VesselFromParent, corrected by qp_to_pipeline_permutation; see vessel.frame_fix\",\n" +
      "    \"body_state_frame\": \"KSP/Unity observable body state, diagnostic only in v0_1\"\n" +
      "  },\n" +
      "  \"capabilities\": {\n" +
      "    \"principia_vessel_relative_state\": true,\n" +
      "    \"principia_tnb_basis\": true,\n" +
      "    \"principia_body_proto_base64\": false,\n" +
      "    \"principia_body_raw_states\": false,\n" +
      "    \"ksp_body_observation\": " + (ev.snapshot_include_ksp_bodies ? "true" : "false") + "\n" +
      "  },\n" +
      "  \"active_body\": {\n" +
      "    \"name\": \"" + JsonEscape(activeBody) + "\",\n" +
      "    \"index\": " + activeBodyIndex + "\n" +
      "  },\n" +
      "  \"vessel\": " + vesselJson + ",\n" +
      "  \"bodies\": " + bodiesJson + ",\n" +
      "  \"limitations\": [\n" +
      "    \"v0_1 is a live navigation snapshot for vessel targeting and diagnostics.\",\n" +
      "    \"It does not contain serialized Principia MassiveBody protos; a native C++ exporter is still needed for a standalone snapshot targeter.\",\n" +
      "    \"KSP body positions/velocities in bodies[] are not advertised as Principia raw Barycentric truth. Use them only for diagnostics unless separately validated.\"\n" +
      "  ]\n" +
      "}";

    return text;
  }

  private static string KspBodiesSnapshotJson(double ut) {
    try {
      if (FlightGlobals.Bodies == null) return "[]";

      string json = "[\n";
      for (int i = 0; i < FlightGlobals.Bodies.Count; ++i) {
        CelestialBody body = FlightGlobals.Bodies[i];
        if (body == null) continue;

        object orbit = null;
        try { orbit = body.orbit; } catch {}

        double[] position = VectorLikeToArray(GetMemberValue(body, "position"));
        double[] relPosAtUt = VectorLikeToArray(InvokeInstance(orbit, "getRelativePositionAtUT", ut));
        double[] relVelAtUt = VectorLikeToArray(InvokeInstance(orbit, "getOrbitalVelocityAtUT", ut));

        double radius = ToDoubleOrNaN(GetMemberValue(body, "Radius"));
        double gravParameter = ToDoubleOrNaN(GetMemberValue(body, "gravParameter"));
        double mass = ToDoubleOrNaN(GetMemberValue(body, "Mass"));
        double soi = ToDoubleOrNaN(GetMemberValue(body, "sphereOfInfluence"));
        double atmosphereDepth = ToDoubleOrNaN(GetMemberValue(body, "atmosphereDepth"));
        double rotationPeriod = ToDoubleOrNaN(GetMemberValue(body, "rotationPeriod"));

        json +=
          "    {\n" +
          "      \"index\": " + i + ",\n" +
          "      \"name\": \"" + JsonEscape(body.bodyName) + "\",\n" +
          "      \"body_proto_base64\": null,\n" +
          "      \"state_source\": \"ksp_celestialbody_observation_diagnostic\",\n" +
          "      \"radius_m\": " + JsonNum(radius) + ",\n" +
          "      \"grav_parameter_m3_s2\": " + JsonNum(gravParameter) + ",\n" +
          "      \"mass_kg\": " + JsonNum(mass) + ",\n" +
          "      \"sphere_of_influence_m\": " + JsonNum(soi) + ",\n" +
          "      \"atmosphere_depth_m\": " + JsonNum(atmosphereDepth) + ",\n" +
          "      \"rotation_period_s\": " + JsonNum(rotationPeriod) + ",\n" +
          "      \"position_world_m\": " + JsonArray3(position) + ",\n" +
          "      \"orbit_rel_r_m\": " + JsonArray3(relPosAtUt) + ",\n" +
          "      \"orbit_rel_v_m_s\": " + JsonArray3(relVelAtUt) + "\n" +
          "    }" + (i + 1 < FlightGlobals.Bodies.Count ? "," : "") + "\n";
      }
      json += "  ]";
      return json;
    } catch (Exception e) {
      Log("KSP_BODIES_SNAPSHOT_EXCEPTION " + e);
      return "[]";
    }
  }

  private static string PrincipiaVesselStateJson(Assembly adapter, Type iface, IntPtr plugin, Vessel vessel) {
    if (vessel == null) return "null";

    double ut = Planetarium.GetUniversalTime();
    string guid = vessel.id.ToString();
    CelestialBody body = vessel.mainBody;
    if (body == null && vessel.orbit != null) body = vessel.orbit.referenceBody;
    string referenceBodyName = body == null ? "" : body.bodyName;
    int bodyIndex = ActiveBodyIndex();

    object qp = null;
    double[] q = null;
    double[] p = null;
    double[] vesselVelocity = null;
    double[] tangent = null;
    double[] normal = null;
    double[] binormal = null;

    try {
      qp = InvokeExact(iface, "VesselFromParent", plugin, bodyIndex, guid);
      q = QPPositionToArray(qp);
      p = QPVelocityToArray(qp);
      if (q == null || p == null) DumpObjectFields("PRINCIPIA_VESSEL_FROM_PARENT_QP", qp);
    } catch (Exception e) {
      Log("PRINCIPIA_VESSEL_FROM_PARENT_EXCEPTION " + e.GetType().Name + ": " + e.Message);
    }

    try { vesselVelocity = XYZToArray(InvokeExact(iface, "VesselVelocity", plugin, guid)); } catch (Exception e) { Log("VESSEL_VELOCITY_EXCEPTION " + e); }
    try { tangent = XYZToArray(InvokeExact(iface, "VesselTangent", plugin, guid)); } catch (Exception e) { Log("VESSEL_TANGENT_EXCEPTION " + e); }
    try { normal = XYZToArray(InvokeExact(iface, "VesselNormal", plugin, guid)); } catch (Exception e) { Log("VESSEL_NORMAL_EXCEPTION " + e); }
    try { binormal = XYZToArray(InvokeExact(iface, "VesselBinormal", plugin, guid)); } catch (Exception e) { Log("VESSEL_BINORMAL_EXCEPTION " + e); }

    if (binormal != null)
    {
        binormal = new double[] { -binormal[0], -binormal[1], -binormal[2] };
    }

    double[] qpPositionRawM = q;
    double[] qpVelocityRawMS = p != null ? p : vesselVelocity;
    
    // Fallback anti-null
    if (qpPositionRawM == null) qpPositionRawM = new double[] { 0.0, 0.0, 0.0 };
    if (qpVelocityRawMS == null) qpVelocityRawMS = new double[] { 0.0, 0.0, 0.0 };
    if (tangent == null) tangent = new double[] { 1.0, 0.0, 0.0 };

    double identityAngleDeg;
    double swapYZAngleDeg;

    string qpToPipelinePermutation = ChooseQpPermutationFromTangent(
        qpVelocityRawMS,
        tangent,
        out identityAngleDeg,
        out swapYZAngleDeg
    );

    double[] relRRawM = ApplyPermutation(qpPositionRawM, qpToPipelinePermutation);
    double[] relVRawMS = ApplyPermutation(qpVelocityRawMS, qpToPipelinePermutation);

    double massTonnes = SafeVesselMassTonnes(vessel, 2.6);
    double totalThrustKn = SafeAvailableThrustKN(vessel, 2686.87701225281);
    double specificImpulseS = SafeSpecificImpulseS(vessel, 1000.0);

    string text =
      "{\n" +
      "    \"schema\": \"active_vessel_state_v1\",\n" +
      "    \"source\": \"MGAPrincipiaBridge\",\n" +
      "    \"state_source\": \"principia.VesselFromParent+qp_permutation_" + qpToPipelinePermutation + "\",\n" +
      "    \"t_game_s\": " + JsonNum(ut) + ",\n" +
      "    \"t_spice_s\": " + JsonNum(ut) + ",\n" +
      "    \"time_note\": \"t_spice_s equals game UT here; Python may add anchor offset if needed\",\n" +
      "    \"vessel_guid\": \"" + JsonEscape(guid) + "\",\n" +
      "    \"vessel_name\": \"" + JsonEscape(vessel.vesselName) + "\",\n" +
      "    \"nav_body\": \"" + JsonEscape(referenceBodyName.ToUpperInvariant()) + "\",\n" +
      "    \"reference_body\": \"" + JsonEscape(referenceBodyName) + "\",\n" +
      "    \"reference_body_index\": " + bodyIndex + ",\n" +
      "    \"situation\": \"" + JsonEscape(vessel.situation.ToString()) + "\",\n" +
      "    \"rel_r_raw_m\": " + JsonArray3(relRRawM) + ",\n" +
      "    \"rel_v_raw_m_s\": " + JsonArray3(relVRawMS) + ",\n" +
      "    \"mass_tonnes\": " + JsonNum(massTonnes) + ",\n" +
      "    \"available_thrust_kN\": " + JsonNum(totalThrustKn) + ",\n" +
      "    \"specific_impulse_s_g0\": " + JsonNum(specificImpulseS) + ",\n" +
      "    \"frame_fix\": {\n" +
      "      \"schema\": \"active_vessel_state_frame_fix_v1\",\n" +
      "      \"qp_to_pipeline_permutation\": \"" + JsonEscape(qpToPipelinePermutation) + "\",\n" +
      "      \"identity_tangent_angle_deg\": " + JsonNum(identityAngleDeg) + ",\n" +
      "      \"swap_yz_tangent_angle_deg\": " + JsonNum(swapYZAngleDeg) + ",\n" +
      "      \"note\": \"QP.q/QP.p are exported in adapter order. The chosen permutation is applied to both position and velocity before writing rel_r_raw_m/rel_v_raw_m_s.\"\n" +
      "    },\n" +
      "    \"principia_basis\": {\n" +
      "      \"tangent_raw\": " + JsonArray3(tangent) + ",\n" +
      "      \"normal_raw\": " + JsonArray3(normal) + ",\n" +
      "      \"binormal_raw\": " + JsonArray3(binormal) + "\n" +
      "    },\n" +
      "    \"debug\": {\n" +
      "      \"qp_type\": \"" + JsonEscape(qp == null ? "null" : TypeName(qp.GetType())) + "\",\n" +
      "      \"qp_field_names\": " + ObjectFieldNamesJson(qp) + ",\n" +
      "      \"qp_position_adapter_order_m\": " + JsonArray3(qpPositionRawM) + ",\n" +
      "      \"qp_velocity_adapter_order_m_s\": " + JsonArray3(qpVelocityRawMS) + ",\n" +
      "      \"qp_position_pipeline_raw_m\": " + JsonArray3(relRRawM) + ",\n" +
      "      \"qp_velocity_pipeline_raw_m_s\": " + JsonArray3(relVRawMS) + ",\n" +
      "      \"vessel_velocity_raw_m_s\": " + JsonArray3(vesselVelocity) + "\n" +
      "    }\n" +
      "  }";

    return text;
  }

  private static double[] QPPositionToArray(object qp) {
    if (qp == null) return null;
    string[] names = new string[] { "q", "position", "r", "degrees_of_freedom", "displacement" };
    for (int i = 0; i < names.Length; ++i) {
      object v = GetField(qp, names[i]);
      double[] a = XYZToArraySafe(v);
      if (a != null) return a;
    }
    return null;
  }

  private static double[] QPVelocityToArray(object qp) {
    if (qp == null) return null;
    string[] names = new string[] { "p", "velocity", "v", "momentum" };
    for (int i = 0; i < names.Length; ++i) {
      object v = GetField(qp, names[i]);
      double[] a = XYZToArraySafe(v);
      if (a != null) return a;
    }
    return null;
  }

  private static double[] XYZToArraySafe(object xyz) {
    if (xyz == null) return null;
    object x = GetField(xyz, "x");
    object y = GetField(xyz, "y");
    object z = GetField(xyz, "z");
    if (x == null || y == null || z == null) return null;
    try {
      return new double[] { Convert.ToDouble(x), Convert.ToDouble(y), Convert.ToDouble(z) };
    } catch {
      return null;
    }
  }

  private static double VecNorm(double[] v) {
      if (v == null || v.Length < 3) return double.NaN;
      return Math.Sqrt(v[0] * v[0] + v[1] * v[1] + v[2] * v[2]);
  }

  private static double VecDot(double[] a, double[] b) {
      if (a == null || b == null || a.Length < 3 || b.Length < 3) return double.NaN;
      return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
  }

  private static double AngleDeg(double[] a, double[] b) {
      double na = VecNorm(a);
      double nb = VecNorm(b);

      if (!(na > 0.0) || !(nb > 0.0)) return double.NaN;

      double c = VecDot(a, b) / (na * nb);
      if (c > 1.0) c = 1.0;
      if (c < -1.0) c = -1.0;

      return Math.Acos(c) * 180.0 / Math.PI;
  }

  private static double[] ApplyPermutation(double[] v, string permutation) {
      if (v == null || v.Length < 3) return null;

      // identity: [x, y, z]
      if (permutation == "012") {
          return new double[] { v[0], v[1], v[2] };
      }

      // swap Y/Z: [x, z, y]
      if (permutation == "021") {
          return new double[] { v[0], v[2], v[1] };
      }

      // fallback seguro
      return new double[] { v[0], v[1], v[2] };
  }

  private static string ChooseQpPermutationFromTangent(
      double[] qpVelocityAdapterOrder,
      double[] principiaTangentRaw,
      out double identityAngleDeg,
      out double swapYZAngleDeg
  ) {
      identityAngleDeg = AngleDeg(qpVelocityAdapterOrder, principiaTangentRaw);
      swapYZAngleDeg = AngleDeg(ApplyPermutation(qpVelocityAdapterOrder, "021"), principiaTangentRaw);

      // Se não der para auditar, usa o comportamento confirmado nos testes.
      if (double.IsNaN(identityAngleDeg) || double.IsNaN(swapYZAngleDeg)) {
          return "021";
      }

      // Escolhe a ordem cuja velocidade fica alinhada com VesselTangent().
      if (swapYZAngleDeg + 1e-9 < identityAngleDeg) {
          return "021";
      }

      return "012";
  }

  private static double SafeDouble(double value, double fallback) {
      if (double.IsNaN(value) || double.IsInfinity(value)) return fallback;
      return value;
  }

  private static double SafeVesselMassTonnes(Vessel vessel, double fallback) {
      try {
          if (vessel != null && vessel.totalMass > 0.0) {
              return vessel.totalMass;
          }
      } catch {
      }

      return fallback;
  }

  private static double SafeAvailableThrustKN(Vessel vessel, double fallback) {
      try {
          if (vessel == null || vessel.Parts == null) return fallback;

          double thrust = 0.0;

          foreach (Part p in vessel.Parts) {
              if (p == null || p.Modules == null) continue;

              foreach (PartModule m in p.Modules) {
                  ModuleEngines engine = m as ModuleEngines;
                  if (engine == null) {
                      ModuleEnginesFX engineFx = m as ModuleEnginesFX;
                      if (engineFx != null) engine = engineFx;
                  }

                  if (engine == null) continue;
                  if (!engine.EngineIgnited) continue;
                  if (engine.flameout) continue;

                  thrust += engine.maxThrust;
              }
          }

          if (thrust > 0.0) return thrust;
      } catch {
      }

      return fallback;
  }

  private static double SafeSpecificImpulseS(Vessel vessel, double fallback) {
      try {
          if (vessel == null || vessel.Parts == null) return fallback;

          double weightedIsp = 0.0;
          double totalThrust = 0.0;

          foreach (Part p in vessel.Parts) {
              if (p == null || p.Modules == null) continue;

              foreach (PartModule m in p.Modules) {
                  ModuleEngines engine = m as ModuleEngines;
                  if (engine == null) {
                      ModuleEnginesFX engineFx = m as ModuleEnginesFX;
                      if (engineFx != null) engine = engineFx;
                  }

                  if (engine == null) continue;
                  if (!engine.EngineIgnited) continue;
                  if (engine.flameout) continue;

                  double thrust = engine.maxThrust;
                  double isp = engine.atmosphereCurve.Evaluate(0.0f);

                  if (thrust > 0.0 && isp > 0.0) {
                      weightedIsp += thrust * isp;
                      totalThrust += thrust;
                  }
              }
          }

          if (totalThrust > 0.0) return weightedIsp / totalThrust;
      } catch {
      }

      return fallback;
  }

  private static string ObjectFieldNamesJson(object obj) {
    if (obj == null) return "[]";
    try {
      FieldInfo[] fs = obj.GetType().GetFields(BindingFlags.Public | BindingFlags.NonPublic |
                                                BindingFlags.Instance | BindingFlags.DeclaredOnly);
      string json = "[";
      for (int i = 0; i < fs.Length; ++i) {
        if (i > 0) json += ", ";
        json += "\"" + JsonEscape(fs[i].Name) + "\"";
      }
      json += "]";
      return json;
    } catch {
      return "[]";
    }
  }


  private static EventResult ExportActiveVesselState(MissionEvent ev, Vessel vessel) {
    EventResult res = new EventResult();
    res.request_id = ev.request_id;
    res.mode = ev.mode;
    res.success = true;
    res.status = "ok";
    res.message = "Active vessel state exported";
    res.extra_json = "  \"active_vessel_state\": " + ActiveVesselStateJson(vessel);
    Log("EXPORT_ACTIVE_VESSEL_STATE ok vessel=" + (vessel == null ? "null" : vessel.vesselName));
    return res;
  }

  private static EventResult DumpAdapterMethods(MissionEvent ev, Type iface) {
    EventResult res = new EventResult();
    res.request_id = ev.request_id;
    res.mode = ev.mode;
    res.success = true;
    res.status = "ok";
    res.message = "Adapter methods dumped";

    MethodInfo[] methods = iface.GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Static);
    Array.Sort(methods, delegate(MethodInfo a, MethodInfo b) { return String.Compare(a.Name, b.Name, StringComparison.Ordinal); });

    string json = "  \"adapter_interface_methods\": [\n";
    for (int i = 0; i < methods.Length; ++i) {
      MethodInfo m = methods[i];
      ParameterInfo[] ps = m.GetParameters();
      string sig = m.Name + "(";
      for (int j = 0; j < ps.Length; ++j) {
        if (j > 0) sig += ", ";
        sig += TypeName(ps[j].ParameterType) + " " + ps[j].Name;
      }
      sig += ") -> " + TypeName(m.ReturnType);
      json += "    \"" + JsonEscape(sig) + "\"" + (i + 1 < methods.Length ? "," : "") + "\n";
      Log("ADAPTER_METHOD " + sig);
    }
    json += "  ]";
    res.extra_json = json;
    return res;
  }

  private static string ActiveVesselStateJson(Vessel vessel) {
    if (vessel == null) return "null";

    double ut = Planetarium.GetUniversalTime();
    string guid = vessel.id.ToString();
    CelestialBody body = vessel.mainBody;
    if (body == null && vessel.orbit != null) body = vessel.orbit.referenceBody;

    object orbit = null;
    try { orbit = vessel.orbit; } catch {}
    if (orbit == null) {
      object od = GetMemberValue(vessel, "orbitDriver");
      orbit = GetMemberValue(od, "orbit");
    }

    double[] orbitPos = VectorLikeToArray(GetMemberValue(orbit, "pos"));
    double[] orbitVel = VectorLikeToArray(GetMemberValue(orbit, "vel"));
    double[] relPosAtUt = VectorLikeToArray(InvokeInstance(orbit, "getRelativePositionAtUT", ut));
    double[] relVelAtUt = VectorLikeToArray(InvokeInstance(orbit, "getOrbitalVelocityAtUT", ut));

    double[] worldPos = VectorLikeToArray(InvokeInstance(vessel, "GetWorldPos3D"));
    double[] obtVel = VectorLikeToArray(GetMemberValue(vessel, "obt_velocity"));
    double[] srfVel = VectorLikeToArray(GetMemberValue(vessel, "srf_velocity"));
    double[] bodyWorldPos = VectorLikeToArray(GetMemberValue(body, "position"));
    double[] worldRelPos = (worldPos != null && bodyWorldPos != null) ? Sub(worldPos, bodyWorldPos) : null;

    // Legacy/KSP fallback only.  For Principia planning prefer
    // export_principia_vessel_state, which uses VesselFromParent.
    double[] selectedR = worldRelPos != null ? worldRelPos : (relPosAtUt != null ? relPosAtUt : orbitPos);
    double[] selectedV = obtVel != null ? obtVel : (relVelAtUt != null ? relVelAtUt : orbitVel);

    double massTonnes = TryInvokeDouble(vessel, "GetTotalMass", Double.NaN);
    if (Double.IsNaN(massTonnes)) massTonnes = ToDoubleOrNaN(GetMemberValue(vessel, "totalMass"));

    double totalThrustKn = TryInvokeDouble(vessel, "GetTotalThrust", Double.NaN);
    if (Double.IsNaN(totalThrustKn)) totalThrustKn = ToDoubleOrNaN(GetMemberValue(vessel, "totalThrust"));

    string referenceBodyName = body == null ? "" : body.bodyName;
    double bodyRadiusM = body == null ? Double.NaN : body.Radius;
    int bodyIndex = ActiveBodyIndex();

    string selectedSource = worldRelPos != null && obtVel != null
        ? "world_pos_minus_body_position/obt_velocity"
        : (relPosAtUt != null && relVelAtUt != null
            ? "orbit.getRelativePositionAtUT/getOrbitalVelocityAtUT"
            : "orbit.pos/orbit.vel");

    string text =
      "{\n" +
      "    \"schema\": \"active_vessel_state_v0\",\n" +
      "    \"source\": \"MGAPrincipiaBridge\",\n" +
      "    \"t_game_s\": " + JsonNum(ut) + ",\n" +
      "    \"t_spice_s\": " + JsonNum(ut) + ",\n" +
      "    \"time_note\": \"t_spice_s equals game UT here; Python may add anchor offset if needed\",\n" +
      "    \"vessel_guid\": \"" + JsonEscape(guid) + "\",\n" +
      "    \"vessel_name\": \"" + JsonEscape(vessel.vesselName) + "\",\n" +
      "    \"nav_body\": \"" + JsonEscape(referenceBodyName.ToUpperInvariant()) + "\",\n" +
      "    \"reference_body\": \"" + JsonEscape(referenceBodyName) + "\",\n" +
      "    \"reference_body_index\": " + bodyIndex + ",\n" +
      "    \"reference_body_radius_m\": " + JsonNum(bodyRadiusM) + ",\n" +
      "    \"situation\": \"" + JsonEscape(vessel.situation.ToString()) + "\",\n" +
      "    \"state_source\": \"" + JsonEscape(selectedSource) + "\",\n" +
      "    \"rel_r_raw_m\": " + JsonArray3(selectedR) + ",\n" +
      "    \"rel_v_raw_m_s\": " + JsonArray3(selectedV) + ",\n" +
      "    \"mass_tonnes\": " + JsonNum(massTonnes) + ",\n" +
      "    \"available_thrust_kN\": " + JsonNum(totalThrustKn) + ",\n" +
      "    \"specific_impulse_s_g0\": null,\n" +
      "    \"debug\": {\n" +
      "      \"orbit_pos_m\": " + JsonArray3(orbitPos) + ",\n" +
      "      \"orbit_vel_m_s\": " + JsonArray3(orbitVel) + ",\n" +
      "      \"orbit_relative_position_at_ut_m\": " + JsonArray3(relPosAtUt) + ",\n" +
      "      \"orbit_orbital_velocity_at_ut_m_s\": " + JsonArray3(relVelAtUt) + ",\n" +
      "      \"world_pos_m\": " + JsonArray3(worldPos) + ",\n" +
      "      \"body_world_pos_m\": " + JsonArray3(bodyWorldPos) + ",\n" +
      "      \"world_relative_position_m\": " + JsonArray3(worldRelPos) + ",\n" +
      "      \"obt_velocity_m_s\": " + JsonArray3(obtVel) + ",\n" +
      "      \"srf_velocity_m_s\": " + JsonArray3(srfVel) + "\n" +
      "    }\n" +
      "  }";

    return text;
  }

  private static object GetMemberValue(object obj, string name) {
    if (obj == null) return null;

    FieldInfo f = FindFieldRecursive(obj.GetType(), name);
    if (f != null) {
      try { return f.GetValue(obj); } catch {}
    }

    PropertyInfo p = FindPropertyRecursive(obj.GetType(), name);
    if (p != null) {
      try { return p.GetValue(obj, null); } catch {}
    }

    return null;
  }

  private static PropertyInfo FindPropertyRecursive(Type type, string name) {
    for (Type t = type; t != null; t = t.BaseType) {
      PropertyInfo p = t.GetProperty(name, BindingFlags.Public | BindingFlags.NonPublic |
                                           BindingFlags.Instance | BindingFlags.Static |
                                           BindingFlags.DeclaredOnly);
      if (p != null) return p;
    }
    return null;
  }

  private static object InvokeInstance(object obj, string name, params object[] args) {
    if (obj == null) return null;
    try {
      MethodInfo m = obj.GetType().GetMethods(BindingFlags.Public | BindingFlags.NonPublic | BindingFlags.Instance)
                         .FirstOrDefault(x => x.Name == name && x.GetParameters().Length == args.Length);
      if (m == null) return null;
      return m.Invoke(obj, args);
    } catch (TargetInvocationException e) {
      Log("INVOKE_INSTANCE_TARGET_EXCEPTION " + name + " inner=" +
          (e.InnerException == null ? e.ToString() : e.InnerException.ToString()));
      return null;
    } catch (Exception e) {
      Log("INVOKE_INSTANCE_EXCEPTION " + name + " " + e.GetType().Name + ": " + e.Message);
      return null;
    }
  }

  private static double TryInvokeDouble(object obj, string name, double fallback) {
    object value = InvokeInstance(obj, name);
    if (value == null) return fallback;
    try { return Convert.ToDouble(value); } catch { return fallback; }
  }

  private static double ToDoubleOrNaN(object value) {
    if (value == null) return Double.NaN;
    try { return Convert.ToDouble(value); } catch { return Double.NaN; }
  }

  private static double[] VectorLikeToArray(object vec) {
    if (vec == null) return null;
    object x = GetMemberValue(vec, "x");
    object y = GetMemberValue(vec, "y");
    object z = GetMemberValue(vec, "z");
    if (x == null || y == null || z == null) return null;
    try {
      return new double[] { Convert.ToDouble(x), Convert.ToDouble(y), Convert.ToDouble(z) };
    } catch {
      return null;
    }
  }

  private static string JsonArray3(double[] v) {
    if (v == null || v.Length < 3) return "null";
    return "[" + JsonNum(v[0]) + ", " + JsonNum(v[1]) + ", " + JsonNum(v[2]) + "]";
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

    ev.require_warp_close = GetBool(json, "require_warp_close", true);
    ev.max_lead_before_insert_s = GetDouble(json, "max_lead_before_insert_s", 600.0);
    ev.reject_long_plan_before_first_burn = GetBool(json, "reject_long_plan_before_first_burn", true);
    ev.max_first_plan_duration_s = GetDouble(json, "max_first_plan_duration_s", 900.0);
    ev.require_status_ok = GetBool(json, "require_status_ok", true);
    ev.rollback_on_status_error = GetBool(json, "rollback_on_status_error", true);
    ev.auto_sort_insert_index = GetBool(json, "auto_sort_insert_index", true);
    ev.delete_existing_flight_plan = GetBool(json, "delete_existing_flight_plan", false);

    ev.snapshot_output_path = GetString(json, "snapshot_output_path", "");
    ev.snapshot_include_ksp_bodies = GetBool(json, "snapshot_include_ksp_bodies", true);
    ev.snapshot_include_debug = GetBool(json, "snapshot_include_debug", true);

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
    foreach (MonoBehaviour mb in GameObject.FindObjectsOfType<MonoBehaviour>()) {
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
        "  \"levela_error_m_s\": " + JsonNum(r.levela_error_m_s) +
        (String.IsNullOrEmpty(r.extra_json) ? "\n" : ",\n" + r.extra_json + "\n") +
        "}\n";
      File.WriteAllText(path, text);
      Log("WROTE_RESULT " + path);
    } catch (Exception e) {
      Log("WRITE_RESULT_EXCEPTION " + e.GetType().Name + ": " + e.Message);
    }
  }

  private static void Log(string msg) {
    string line = "[MGAPrincipiaBridge] " + msg;
    UnityEngine.Debug.Log(line);

    try {
      string path = Path.Combine(KSPUtil.ApplicationRootPath, "MGAPrincipiaBridge_probe.log");
      File.AppendAllText(path, DateTime.Now.ToString("o") + " " + line + "\n");
    } catch {}
  }
}