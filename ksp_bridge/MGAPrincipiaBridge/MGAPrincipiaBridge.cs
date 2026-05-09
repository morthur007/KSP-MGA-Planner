using System;
using System.IO;
using System.Linq;
using System.Reflection;
using System.Text.RegularExpressions;
using System.Globalization;
using UnityEngine;

[KSPAddon(KSPAddon.Startup.Flight, false)]
public sealed class MGAPrincipiaBridgeMissionEventDaemon : MonoBehaviour {
  private static string lastRequestId = "";
  private static double nextPollUt = 0.0;
  private const double PollPeriodS = 1.0;

  public void Start() {
    DontDestroyOnLoad(this);
    Log("MGAPrincipiaBridgeMissionEventDaemon.Start");
  }

  public void Update() {
    if (!HighLogic.LoadedSceneIsFlight) return;
    if (FlightGlobals.ActiveVessel == null) return;

    double ut = Planetarium.GetUniversalTime();
    if (ut < nextPollUt) return;
    nextPollUt = ut + PollPeriodS;

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

    if (!ev.enabled) return;

    if (String.IsNullOrEmpty(ev.request_id)) {
      Log("EVENT_IGNORED missing request_id");
      return;
    }

    if (ev.request_id == lastRequestId) return;

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
    public string vessel_guid = "";
    public int insert_index = -1;
    public int clone_from_index = -1;
    public double initial_time = Double.NaN;
    public double[] delta_v_navigation_m_s = null;
    public double[] delta_v_levela_m_s = null;
    public double placeholder_dv_m_s = 0.001;
    public double tolerance_time_s = 0.01;
    public double tolerance_dv_m_s = 1e-6;
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
    Log("FlightPlanExists=" + exists);
    if (!exists) {
      Log("FAIL no existing flight plan; create one manually in Principia first");
      res.success = false;
      res.status = "no_flight_plan";
      res.message = "No existing Principia flight plan";
      return res;
    }

    int before = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfManoeuvres", plugin, activeGuid));
    int segmentsBefore = Convert.ToInt32(InvokeExact(iface, "FlightPlanNumberOfSegments", plugin, activeGuid));
    res.before = before;
    res.segments_before = segmentsBefore;
    Log("BEFORE manoeuvres=" + before + " segments=" + segmentsBefore);

    if (before < 1) {
      Log("FAIL expected at least one existing manoeuvre to clone");
      res.success = false;
      res.status = "no_source_manoeuvre";
      res.message = "Need at least one existing manoeuvre to clone engine/frame parameters";
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

    int cloneIndex = ev.clone_from_index >= 0 ? ev.clone_from_index : before - 1;
    if (cloneIndex < 0 || cloneIndex >= before) {
      Log("FAIL invalid clone_from_index=" + cloneIndex + " before=" + before);
      res.success = false;
      res.status = "invalid_clone_index";
      res.message = "clone_from_index out of range";
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

    object sourceManoeuvre = InvokeExact(iface, "FlightPlanGetManoeuvre", plugin, activeGuid, cloneIndex);
    object burn = GetField(sourceManoeuvre, "burn");
    if (burn == null) {
      Log("FAIL could not clone source burn");
      res.success = false;
      res.status = "clone_failed";
      res.message = "Could not clone source burn";
      return res;
    }

    SetField(burn, "initial_time", ev.initial_time);
    SetField(burn, "is_inertially_fixed", false);

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
    Log("INSERT clone_from_index=" + cloneIndex);
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
      SetField(insertedBurn, "is_inertially_fixed", false);

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
    ev.vessel_guid = GetString(json, "vessel_guid", "");
    ev.insert_index = GetInt(json, "insert_index", -1);
    ev.clone_from_index = GetInt(json, "clone_from_index", -1);
    ev.initial_time = GetDouble(json, "initial_time", Double.NaN);
    ev.placeholder_dv_m_s = GetDouble(json, "placeholder_dv_m_s", 0.001);
    ev.tolerance_time_s = GetDouble(json, "tolerance_time_s", 0.01);
    ev.tolerance_dv_m_s = GetDouble(json, "tolerance_dv_m_s", 1e-6);
    ev.delta_v_navigation_m_s = GetArray3(json, "delta_v_navigation_m_s");
    ev.delta_v_levela_m_s = GetArray3(json, "delta_v_levela_m_s");
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
