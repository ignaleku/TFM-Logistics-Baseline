import { useState, useMemo, useRef } from "react";
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from "recharts";

const SAMPLE_CSV = `regime,policy,total_sla,urgent_sla,normal_sla,urgent_late_orders,normal_late_orders,estimated_late_cost,savings_vs_fifo,savings_vs_urgent_first,cost_late_urgent,cost_late_normal,urgent_orders_assumed,normal_orders_assumed
s111,fifo,0.0271,0.0074,0.0298,1191.1,8537.8,66510,0,-11164,20,5,1200,8800
s111,urgent_first,0.0763,0.4939,0.0182,607.3,8639.8,55346,11164,0,20,5,1200,8800
s111,rl3_dqn,0.1388,1.0000,0.0190,0.0,8632.8,43163,23347,12183,20,5,1200,8800
s211,fifo,0.2063,0.0229,0.2318,1172.5,6759.8,57250,0,-20858,20,5,1200,8800
s211,urgent_first,0.2739,1.0000,0.1729,0.0,7278.5,36392,20858,0,20,5,1200,8800
s211,rl3_dqn,0.2794,1.0000,0.1792,0.0,7223.0,36116,21134,276,20,5,1200,8800
s221,fifo,0.4064,0.0565,0.4551,1132.2,4799.1,46621,0,-19677,20,5,1200,8800
s221,urgent_first,0.4624,1.0000,0.3876,0.0,5389.1,26944,19677,0,20,5,1200,8800
s221,rl3_dqn,0.4603,1.0000,0.3852,0.0,5410.0,27050,19571,-105,20,5,1200,8800
s311,fifo,0.2039,0.0221,0.2292,1173.5,6783.0,57385,0,-21038,20,5,1200,8800
s311,urgent_first,0.2748,1.0000,0.1739,0.0,7269.7,36347,21038,0,20,5,1200,8800
s311,rl3_dqn,0.2803,1.0000,0.1802,0.0,7214.3,36071,21314,276,20,5,1200,8800
s321,fifo,0.9161,0.3129,1.0000,824.5,0.0,16491,0,-16491,20,5,1200,8800
s321,urgent_first,1.0000,1.0000,1.0000,0.0,0.0,0,16491,0,20,5,1200,8800
s321,rl3_dqn,1.0000,1.0000,1.0000,0.0,0.0,0,16491,0,20,5,1200,8800
s222,fifo,0.4033,0.0557,0.4516,1133.2,4825.9,46791,0,-19997,20,5,1200,8800
s222,urgent_first,0.4654,1.0000,0.3910,0.0,5359.2,26794,19997,0,20,5,1200,8800
s222,rl3_dqn,0.4627,1.0000,0.3880,0.0,5385.6,26929,19862,-135,20,5,1200,8800
s332,fifo,0.9166,0.3170,1.0000,819.6,0.0,16393,0,-16393,20,5,1200,8800
s332,urgent_first,1.0000,1.0000,1.0000,0.0,0.0,0,16393,0,20,5,1200,8800
s332,rl3_dqn,1.0000,1.0000,1.0000,0.0,0.0,0,16393,0,20,5,1200,8800`;

const REQUIRED_COLS = ["regime","policy","total_sla","urgent_sla","normal_sla","urgent_late_orders","normal_late_orders","estimated_late_cost","savings_vs_fifo","savings_vs_urgent_first"];
const NUMERIC_COLS = ["total_sla","urgent_sla","normal_sla","urgent_late_orders","normal_late_orders","estimated_late_cost","savings_vs_fifo","savings_vs_urgent_first"];

const DISPLAY_POLICY = { fifo:"FIFO", urgent_first:"Urgent first", rl3_dqn:"RL-3" };
const disp = p => DISPLAY_POLICY[p] || p;

const PALETTE = ["#378ADD","#1D9E75","#D85A30","#7F77DD","#BA7517","#D4537E","#639922","#E24B4A"];
function policyColor(policies, p) {
  const i = policies.indexOf(p);
  return PALETTE[i % PALETTE.length];
}

function parseCSV(raw) {
  const lines = raw.trim().split("\n").map(l => l.trim()).filter(Boolean);
  if (lines.length < 2) return { error: "El CSV debe tener al menos una fila de cabecera y una de datos." };
  const headers = lines[0].split(",").map(h => h.trim());
  const missing = REQUIRED_COLS.filter(c => !headers.includes(c));
  if (missing.length) return { error: `Columnas requeridas no encontradas: ${missing.join(", ")}` };
  const rows = [];
  const warnings = [];
  for (let i = 1; i < lines.length; i++) {
    const vals = lines[i].split(",");
    const row = {};
    headers.forEach((h, j) => { row[h] = vals[j]?.trim() ?? ""; });
    let valid = true;
    for (const col of NUMERIC_COLS) {
      const v = parseFloat(row[col]);
      if (isNaN(v)) { warnings.push(`Fila ${i + 1}: valor no numérico en "${col}" ("${row[col]}")`); valid = false; }
      else row[col] = v;
    }
    if (valid) rows.push(row);
  }
  if (!rows.length) return { error: "No se han podido parsear filas válidas del CSV." };
  return { rows, warnings };
}

const fmt = n => Math.round(n).toLocaleString("es-ES");
const pct = n => (parseFloat(n) * 100).toFixed(1) + "%";
const fmtD = n => parseFloat(n).toFixed(1);

function KpiCard({ label, value, sub, accent, highlight }) {
  return (
    <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-md)", padding: "0.9rem 1rem", border: highlight ? `1.5px solid ${accent || "#1D9E75"}` : "0.5px solid var(--color-border-tertiary)", minWidth: 0 }}>
      <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: "0 0 5px" }}>{label}</p>
      <p style={{ fontSize: 20, fontWeight: 500, margin: 0, color: accent || "var(--color-text-primary)" }}>{value}</p>
      {sub && <p style={{ fontSize: 11, color: "var(--color-text-secondary)", margin: "4px 0 0" }}>{sub}</p>}
    </div>
  );
}

export default function App() {
  const [csvText, setCsvText] = useState("");
  const [data, setData] = useState(() => parseCSV(SAMPLE_CSV).rows);
  const [errors, setErrors] = useState([]);
  const [warnings, setWarnings] = useState([]);
  const [regimeFilter, setRegimeFilter] = useState("all");
  const [showPaste, setShowPaste] = useState(false);
  const fileRef = useRef();

  const regimes = useMemo(() => data ? [...new Set(data.map(r => r.regime))] : [], [data]);
  const policies = useMemo(() => data ? [...new Set(data.map(r => r.policy))] : [], [data]);

  function loadCSV(raw) {
    const result = parseCSV(raw);
    if (result.error) { setErrors([result.error]); setWarnings([]); return; }
    setErrors([]);
    setWarnings(result.warnings || []);
    setData(result.rows);
    setRegimeFilter("all");
    setCsvText("");
    setShowPaste(false);
  }

  function handleFile(e) {
    const f = e.target.files[0];
    if (!f) return;
    const reader = new FileReader();
    reader.onload = ev => loadCSV(ev.target.result);
    reader.readAsText(f);
    e.target.value = "";
  }

  const kpi = useMemo(() => {
    if (!data) return null;
    const totals = {};
    policies.forEach(p => { totals[p] = data.filter(r => r.policy === p).reduce((s, r) => s + r.estimated_late_cost, 0); });
    const best = policies.reduce((a, b) => totals[a] < totals[b] ? a : b);
    const rl3 = policies.find(p => p === "rl3_dqn");
    const uf = policies.find(p => p === "urgent_first");
    const fifo = policies.find(p => p === "fifo");
    const rl3SaveFifo = rl3 && fifo ? totals[fifo] - totals[rl3] : null;
    const rl3SaveUf = rl3 && uf ? totals[uf] - totals[rl3] : null;
    return { totals, best, rl3SaveFifo, rl3SaveUf };
  }, [data, policies]);

  const bestPerRegime = useMemo(() => {
    const map = {};
    regimes.forEach(reg => {
      const rows = data.filter(r => r.regime === reg);
      const minCost = Math.min(...rows.map(r => r.estimated_late_cost));
      map[reg] = rows.filter(r => Math.abs(r.estimated_late_cost - minCost) < 0.5).map(r => r.policy);
    });
    return map;
  }, [data, regimes]);

  const chartByRegime = useMemo(() => regimes.map(reg => {
    const obj = { regime: reg };
    policies.forEach(p => {
      const row = data.find(r => r.regime === reg && r.policy === p);
      obj[p] = row ? Math.round(row.estimated_late_cost) : null;
    });
    return obj;
  }), [data, regimes, policies]);

  const chartByPolicy = useMemo(() => policies.map(p => ({
    policy: disp(p),
    cost: Math.round(data.filter(r => r.policy === p).reduce((s, r) => s + r.estimated_late_cost, 0)),
    color: policyColor(policies, p),
  })), [data, policies]);

  const filtered = useMemo(() => regimeFilter === "all" ? data : data.filter(r => r.regime === regimeFilter), [data, regimeFilter]);

  const interpretation = useMemo(() => {
    if (!data || !kpi) return null;
    const rl3 = policies.find(p => p === "rl3_dqn");
    const uf = policies.find(p => p === "urgent_first");
    const rl3BestRegimes = rl3 ? regimes.filter(reg => bestPerRegime[reg]?.includes("rl3_dqn")) : [];
    const rl3CloseUf = (rl3 && uf) ? regimes.filter(reg => {
      const r = data.find(d => d.regime === reg && d.policy === "rl3_dqn");
      const u = data.find(d => d.regime === reg && d.policy === "urgent_first");
      if (!r || !u || bestPerRegime[reg]?.includes("rl3_dqn")) return false;
      return Math.abs(r.estimated_late_cost - u.estimated_late_cost) / (u.estimated_late_cost + 1) < 0.05;
    }) : [];
    return { rl3BestRegimes, rl3CloseUf };
  }, [data, kpi, policies, regimes, bestPerRegime]);

  const sectionStyle = { background: "var(--color-background-primary)", border: "0.5px solid var(--color-border-tertiary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem", marginBottom: "1.25rem" };
  const secTitle = txt => <p style={{ fontSize: 12, fontWeight: 500, color: "var(--color-text-secondary)", margin: "0 0 12px", textTransform: "uppercase", letterSpacing: "0.05em" }}>{txt}</p>;

  return (
    <div style={{ padding: "1.5rem 1rem", maxWidth: 920, margin: "0 auto", fontFamily: "var(--font-sans)" }}>
      <h2 style={{ position: "absolute", left: -9999 }} className="sr-only">SLA cost dashboard — CSV-driven logistics visualization</h2>

      {/* Header */}
      <h1 style={{ fontSize: 20, fontWeight: 500, margin: "0 0 4px", color: "var(--color-text-primary)" }}>SLA Cost Dashboard — Logistics RL-3</h1>
      <p style={{ fontSize: 13, color: "var(--color-text-secondary)", margin: "0 0 10px" }}>Visual dashboard for the SLA cost analysis CSV generated by the Python project.</p>
      <div style={{ fontSize: 12, background: "#E6F1FB", color: "#185FA5", border: "0.5px solid #B5D4F4", borderRadius: "var(--border-radius-md)", padding: "7px 12px", marginBottom: "1.25rem", display: "flex", gap: 8, alignItems: "flex-start" }}>
        <span style={{ fontSize: 14, marginTop: 1 }}>ℹ</span>
        <span>Los costes son calculados por el script Python <code style={{ background: "#B5D4F422", padding: "1px 4px", borderRadius: 3 }}>src/reporting/sla_cost_calculator.py</code>. Este dashboard solo visualiza el CSV generado, por lo que los resultados son reproducibles independientemente.</span>
      </div>

      <hr style={{ border: "none", borderTop: "0.5px solid var(--color-border-tertiary)", margin: "0 0 1.25rem" }} />

      {/* CSV Loader */}
      <div style={{ ...sectionStyle }}>
        {secTitle("Cargar CSV de resultados")}
        <div style={{ display: "flex", gap: 10, flexWrap: "wrap", alignItems: "center", marginBottom: 10 }}>
          <button onClick={() => fileRef.current.click()} style={{ fontSize: 13, padding: "6px 14px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", cursor: "pointer" }}>
            Subir archivo .csv
          </button>
          <input ref={fileRef} type="file" accept=".csv,text/csv" onChange={handleFile} style={{ display: "none" }} />
          <button onClick={() => setShowPaste(v => !v)} style={{ fontSize: 13, padding: "6px 14px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: showPaste ? "var(--color-background-info)" : "var(--color-background-secondary)", color: showPaste ? "var(--color-text-info)" : "var(--color-text-primary)", cursor: "pointer" }}>
            {showPaste ? "Ocultar área de pegado" : "Pegar CSV manualmente"}
          </button>
          <button onClick={() => { setData(parseCSV(SAMPLE_CSV).rows); setErrors([]); setWarnings([]); setRegimeFilter("all"); setCsvText(""); setShowPaste(false); }}
            style={{ fontSize: 13, padding: "6px 14px", borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-secondary)", cursor: "pointer" }}>
            Restablecer datos de ejemplo
          </button>
        </div>
        {showPaste && (
          <div style={{ display: "flex", flexDirection: "column", gap: 8 }}>
            <textarea value={csvText} onChange={e => setCsvText(e.target.value)} placeholder="Pega aquí el contenido del CSV…" rows={6}
              style={{ width: "100%", fontSize: 12, padding: 8, borderRadius: "var(--border-radius-md)", border: "0.5px solid var(--color-border-secondary)", background: "var(--color-background-secondary)", color: "var(--color-text-primary)", fontFamily: "var(--font-mono)", resize: "vertical", boxSizing: "border-box" }} />
            <button onClick={() => csvText.trim() && loadCSV(csvText)}
              style={{ alignSelf: "flex-start", fontSize: 13, padding: "6px 16px", borderRadius: "var(--border-radius-md)", border: "0.5px solid #185FA5", background: "#E6F1FB", color: "#185FA5", cursor: "pointer", fontWeight: 500 }}>
              Cargar resultados CSV
            </button>
          </div>
        )}
        {errors.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 12, background: "#FCEBEB", color: "#A32D2D", border: "0.5px solid #F7C1C1", borderRadius: "var(--border-radius-md)", padding: "8px 12px" }}>
            {errors.map((e, i) => <p key={i} style={{ margin: "2px 0" }}>✕ {e}</p>)}
          </div>
        )}
        {warnings.length > 0 && (
          <div style={{ marginTop: 10, fontSize: 12, background: "#FAEEDA", color: "#633806", border: "0.5px solid #FAC775", borderRadius: "var(--border-radius-md)", padding: "8px 12px" }}>
            {warnings.slice(0, 5).map((w, i) => <p key={i} style={{ margin: "2px 0" }}>⚠ {w}</p>)}
            {warnings.length > 5 && <p style={{ margin: "4px 0 0", color: "#854F0B" }}>… y {warnings.length - 5} advertencias más.</p>}
          </div>
        )}
      </div>

      {data && kpi && <>
        {/* KPI Cards */}
        <div style={{ display: "grid", gridTemplateColumns: "repeat(auto-fit, minmax(140px, 1fr))", gap: 10, marginBottom: "1.25rem" }}>
          {policies.map(p => (
            <KpiCard key={p} label={`Coste total — ${disp(p)}`} value={`€${fmt(kpi.totals[p])}`}
              accent={policyColor(policies, p)} highlight={kpi.best === p}
              sub={kpi.best === p ? "Mejor política global" : null} />
          ))}
          {kpi.rl3SaveFifo !== null && <KpiCard label="Ahorro RL-3 vs FIFO" value={`${kpi.rl3SaveFifo >= 0 ? "+" : ""}€${fmt(kpi.rl3SaveFifo)}`} accent={kpi.rl3SaveFifo >= 0 ? "#0F6E56" : "#993C1D"} />}
          {kpi.rl3SaveUf !== null && <KpiCard label="Ahorro RL-3 vs Urgent first" value={`${kpi.rl3SaveUf >= 0 ? "+" : ""}€${fmt(kpi.rl3SaveUf)}`} accent={kpi.rl3SaveUf >= 0 ? "#0F6E56" : "#993C1D"} />}
        </div>

        {/* Chart by regime */}
        <div style={sectionStyle}>
          {secTitle("Coste estimado por régimen y política (€)")}
          <div style={{ display: "flex", gap: 16, flexWrap: "wrap", marginBottom: 10 }}>
            {policies.map(p => (
              <span key={p} style={{ display: "flex", alignItems: "center", gap: 5, fontSize: 12, color: "var(--color-text-secondary)" }}>
                <span style={{ width: 10, height: 10, borderRadius: 2, background: policyColor(policies, p), display: "inline-block" }} />
                {disp(p)}
              </span>
            ))}
          </div>
          <div style={{ width: "100%", height: Math.max(200, regimes.length * 34 + 60) }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartByRegime} margin={{ top: 4, right: 8, left: 0, bottom: 0 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" />
                <XAxis dataKey="regime" tick={{ fontSize: 12, fill: "var(--color-text-secondary)" }} />
                <YAxis tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} tickFormatter={v => v >= 1000 ? "€" + Math.round(v / 1000) + "k" : "€" + v} />
                <Tooltip formatter={(v, name) => v !== null ? ["€" + fmt(v), disp(name)] : ["-", disp(name)]} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                {policies.map(p => <Bar key={p} dataKey={p} fill={policyColor(policies, p)} radius={[3, 3, 0, 0]} />)}
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Chart by policy total */}
        <div style={sectionStyle}>
          {secTitle("Coste total acumulado por política (todos los regímenes)")}
          <div style={{ width: "100%", height: 180 }}>
            <ResponsiveContainer width="100%" height="100%">
              <BarChart data={chartByPolicy} margin={{ top: 4, right: 8, left: 0, bottom: 0 }} layout="vertical">
                <CartesianGrid strokeDasharray="3 3" stroke="var(--color-border-tertiary)" horizontal={false} />
                <XAxis type="number" tick={{ fontSize: 11, fill: "var(--color-text-secondary)" }} tickFormatter={v => "€" + Math.round(v / 1000) + "k"} />
                <YAxis type="category" dataKey="policy" tick={{ fontSize: 13, fill: "var(--color-text-primary)", fontWeight: 500 }} width={90} />
                <Tooltip formatter={v => "€" + fmt(v)} contentStyle={{ fontSize: 12, borderRadius: 8 }} />
                <Bar dataKey="cost" radius={[0, 4, 4, 0]}>
                  {chartByPolicy.map((entry, i) => <Cell key={i} fill={entry.color} />)}
                </Bar>
              </BarChart>
            </ResponsiveContainer>
          </div>
        </div>

        {/* Regime filter */}
        <div style={{ display: "flex", gap: 8, flexWrap: "wrap", marginBottom: "1rem" }}>
          {["all", ...regimes].map(r => (
            <button key={r} onClick={() => setRegimeFilter(r)}
              style={{ fontSize: 12, padding: "4px 12px", borderRadius: "var(--border-radius-md)", border: regimeFilter === r ? "1.5px solid #378ADD" : "0.5px solid var(--color-border-secondary)", background: regimeFilter === r ? "#378ADD18" : "var(--color-background-primary)", color: regimeFilter === r ? "#185FA5" : "var(--color-text-secondary)", cursor: "pointer", fontWeight: regimeFilter === r ? 500 : 400 }}>
              {r === "all" ? "Todos" : r}
            </button>
          ))}
        </div>

        {/* Table */}
        <div style={{ ...sectionStyle, padding: 0, overflow: "hidden" }}>
          <div style={{ overflowX: "auto" }}>
            <table style={{ width: "100%", borderCollapse: "collapse", fontSize: 12 }}>
              <thead>
                <tr style={{ background: "var(--color-background-secondary)" }}>
                  {["Régimen","Política","SLA total","SLA urg.","SLA norm.","Tarde urg.","Tarde norm.","Coste est.","Ahorro vs FIFO","Ahorro vs UF"].map(h => (
                    <th key={h} style={{ padding: "8px 10px", textAlign: h === "Régimen" || h === "Política" ? "left" : "right", fontWeight: 500, color: "var(--color-text-secondary)", borderBottom: "0.5px solid var(--color-border-tertiary)", whiteSpace: "nowrap" }}>{h}</th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {filtered.map((row, i) => {
                  const isBest = bestPerRegime[row.regime]?.includes(row.policy);
                  return (
                    <tr key={i} style={{ background: isBest ? "#1D9E7508" : i % 2 === 0 ? "var(--color-background-primary)" : "var(--color-background-secondary)" }}>
                      <td style={{ padding: "7px 10px", fontWeight: 500, color: "var(--color-text-secondary)", whiteSpace: "nowrap" }}>{row.regime}</td>
                      <td style={{ padding: "7px 10px", whiteSpace: "nowrap" }}>
                        <span style={{ display: "flex", alignItems: "center", gap: 6 }}>
                          <span style={{ width: 8, height: 8, borderRadius: 2, background: policyColor(policies, row.policy), display: "inline-block", flexShrink: 0 }} />
                          {disp(row.policy)}
                          {isBest && <span style={{ fontSize: 10, background: "#1D9E7522", color: "#0F6E56", borderRadius: 4, padding: "1px 5px", fontWeight: 500 }}>★</span>}
                        </span>
                      </td>
                      <td style={{ padding: "7px 10px", textAlign: "right" }}>{pct(row.total_sla)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right" }}>{pct(row.urgent_sla)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right" }}>{pct(row.normal_sla)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right" }}>{fmtD(row.urgent_late_orders)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right" }}>{fmtD(row.normal_late_orders)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right", fontWeight: 500 }}>€{fmt(row.estimated_late_cost)}</td>
                      <td style={{ padding: "7px 10px", textAlign: "right", color: row.savings_vs_fifo > 0 ? "#0F6E56" : row.savings_vs_fifo < 0 ? "#993C1D" : "var(--color-text-secondary)" }}>
                        {row.savings_vs_fifo === 0 ? "—" : (row.savings_vs_fifo > 0 ? "+" : "") + "€" + fmt(row.savings_vs_fifo)}
                      </td>
                      <td style={{ padding: "7px 10px", textAlign: "right", color: row.savings_vs_urgent_first > 0 ? "#0F6E56" : row.savings_vs_urgent_first < 0 ? "#993C1D" : "var(--color-text-secondary)" }}>
                        {row.savings_vs_urgent_first === 0 ? "—" : (row.savings_vs_urgent_first > 0 ? "+" : "") + "€" + fmt(row.savings_vs_urgent_first)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        </div>

        {/* Interpretation */}
        {interpretation && (
          <div style={{ background: "var(--color-background-secondary)", borderRadius: "var(--border-radius-lg)", padding: "1rem 1.25rem", border: "0.5px solid var(--color-border-tertiary)", marginTop: "0.25rem" }}>
            {secTitle("Interpretación automática")}
            <div style={{ display: "flex", flexDirection: "column", gap: 8, fontSize: 13, color: "var(--color-text-primary)", lineHeight: 1.7 }}>
              <p style={{ margin: 0 }}>
                <strong style={{ fontWeight: 500 }}>Mejor política global:</strong>{" "}
                <span style={{ color: policyColor(policies, kpi.best), fontWeight: 500 }}>{disp(kpi.best)}</span>
                {" "}— coste total estimado de <strong>€{fmt(kpi.totals[kpi.best])}</strong> sumando todos los regímenes.
              </p>
              <p style={{ margin: 0 }}>
                <strong style={{ fontWeight: 500 }}>Mejor política por régimen:</strong>{" "}
                {regimes.map(reg => (
                  <span key={reg} style={{ display: "inline-flex", alignItems: "center", gap: 4, marginRight: 10, fontSize: 12 }}>
                    <span style={{ color: "var(--color-text-secondary)" }}>{reg}:</span>
                    {bestPerRegime[reg]?.map(p => (
                      <span key={p} style={{ background: policyColor(policies, p) + "22", color: policyColor(policies, p), border: `0.5px solid ${policyColor(policies, p)}55`, borderRadius: 4, padding: "1px 6px", fontSize: 11, fontWeight: 500 }}>{disp(p)}</span>
                    ))}
                  </span>
                ))}
              </p>
              {interpretation.rl3BestRegimes.length > 0 ? (
                <p style={{ margin: 0 }}>
                  <strong style={{ fontWeight: 500 }}>RL-3 tiene el menor coste en:</strong>{" "}
                  {interpretation.rl3BestRegimes.join(", ")}
                </p>
              ) : policies.includes("rl3_dqn") && (
                <p style={{ margin: 0 }}>
                  <strong style={{ fontWeight: 500 }}>RL-3</strong> no lidera en coste en ningún régimen con los datos cargados.
                </p>
              )}
              {interpretation.rl3CloseUf.length > 0 && (
                <p style={{ margin: 0 }}>
                  <strong style={{ fontWeight: 500 }}>RL-3 muy próximo a Urgent first (&lt;5% diferencia) en:</strong>{" "}
                  {interpretation.rl3CloseUf.join(", ")}
                </p>
              )}
              <p style={{ margin: "6px 0 0", fontSize: 12, color: "var(--color-text-secondary)", borderTop: "0.5px solid var(--color-border-tertiary)", paddingTop: 8 }}>
                ⚠ Este es un modelo de coste simplificado basado en los resultados de simulación generados y las suposiciones del script Python. No es un modelo contable real de empresa.
              </p>
            </div>
          </div>
        )}
      </>}
    </div>
  );
}
