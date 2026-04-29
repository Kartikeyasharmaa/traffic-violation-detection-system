import { ArrowLeftRight, AlertTriangle, HardHat, Play, Square, TrafficCone } from "lucide-react";
import { Link } from "react-router-dom";
import { useDashboardContext } from "../components/AppShell";
import type { DetectorStatus, ViolationType } from "../lib/types";

const detectorLabels: Record<ViolationType, string> = {
  helmet: "Helmet",
  red_light: "Red Light",
  wrong_side: "Wrong Side",
};

const detectorDescriptions: Record<ViolationType, string> = {
  helmet: "Runs helmet detection and saves each confirmed violation frame.",
  red_light: "Runs red-light detection with stop-line and signal analysis.",
  wrong_side: "Runs wrong-side detection with direction tracking and evidence capture.",
};

function formatPlateNumber(value: string | null) {
  if (!value || value.toUpperCase() === "UNKNOWN") {
    return "Plate Not Detected";
  }
  return value;
}

function formatTimestamp(value: string) {
  const date = new Date(value.endsWith("Z") ? value : `${value}Z`);
  return `${date.toLocaleString("en-IN", {
    timeZone: "Asia/Kolkata",
    dateStyle: "medium",
    timeStyle: "short",
  })} IST`;
}

function StatCard({
  title,
  value,
  icon: Icon,
  tone,
}: {
  title: string;
  value: number;
  icon: typeof AlertTriangle;
  tone: "primary" | "warning" | "danger" | "success";
}) {
  return (
    <article className={`stat-card tone-${tone}`}>
      <div className="stat-icon">
        <Icon size={20} />
      </div>
      <div>
        <p className="stat-label">{title}</p>
        <h3 className="stat-value">{value}</h3>
      </div>
    </article>
  );
}

function DetectorCard({
  type,
  status,
  starting,
  stopping,
  onStart,
  onStop,
}: {
  type: ViolationType;
  status: DetectorStatus;
  starting: boolean;
  stopping: boolean;
  onStart: (type: ViolationType) => Promise<void>;
  onStop: (type: ViolationType) => Promise<void>;
}) {
  const running = Boolean(status.running);

  return (
    <article className="detector-card">
      <div className="detector-card-top">
        <span className="detector-badge">{detectorLabels[type]}</span>
        <span className={`status-chip ${running ? "running" : "idle"}`}>{running ? "Running" : "Idle"}</span>
      </div>
      <h3>{detectorLabels[type]} Detection</h3>
      <p>{detectorDescriptions[type]}</p>
      <div className="detector-card-bottom">
        {running ? (
          <button className="danger-button detector-button" disabled={stopping} onClick={() => onStop(type)}>
            <Square size={16} />
            <span>{stopping ? "Stopping..." : `Stop ${detectorLabels[type]}`}</span>
          </button>
        ) : (
          <button className="primary-button detector-button" disabled={starting} onClick={() => onStart(type)}>
            <Play size={16} />
            <span>{starting ? "Starting..." : `Start ${detectorLabels[type]}`}</span>
          </button>
        )}
        <span className="muted-text">{running ? "Desktop preview is active" : "Click Start to launch"}</span>
      </div>
    </article>
  );
}

export default function DashboardPage() {
  const { stats, violations, detectors, startingDetector, stoppingDetector, startDetector, stopDetector } = useDashboardContext();

  const recentViolations = violations.slice(0, 5);
  const mix = [
    { label: "Helmet", value: stats.helmet_violations, tone: "warning" },
    { label: "Red Light", value: stats.red_light_violations, tone: "danger" },
    { label: "Wrong Side", value: stats.wrong_side_violations, tone: "success" },
  ];
  const maxMixValue = Math.max(1, ...mix.map((item) => item.value));

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Admin Overview</p>
          <h1>Traffic Control Dashboard</h1>
          <p className="page-copy">Track detector status, review recent violations, and manage the monitoring workflow from one place.</p>
        </div>
      </div>

      <div className="stats-grid">
        <StatCard title="Total Violations" value={stats.total_violations} icon={AlertTriangle} tone="primary" />
        <StatCard title="Helmet Violations" value={stats.helmet_violations} icon={HardHat} tone="warning" />
        <StatCard title="Red Light Violations" value={stats.red_light_violations} icon={TrafficCone} tone="danger" />
        <StatCard title="Wrong Side Violations" value={stats.wrong_side_violations} icon={ArrowLeftRight} tone="success" />
      </div>

      <div className="content-grid">
        <div className="panel">
          <div className="panel-head">
            <div>
              <h3>Violation Mix</h3>
              <p>Distribution of saved records across the three detection modules.</p>
            </div>
          </div>

          <div className="bar-list">
            {mix.map((item) => (
              <div key={item.label} className="bar-row">
                <div className="bar-meta">
                  <span>{item.label}</span>
                  <strong>{item.value}</strong>
                </div>
                <div className="bar-track">
                  <div className={`bar-fill tone-${item.tone}`} style={{ width: `${(item.value / maxMixValue) * 100}%` }} />
                </div>
              </div>
            ))}
          </div>
        </div>

        <div className="panel">
          <div className="panel-head">
            <div>
              <h3>Recent Violations</h3>
              <p>Latest saved evidence frames with number plate and timestamp details.</p>
            </div>
            <Link className="text-button link-button" to="/violations">
              View all
            </Link>
          </div>

          <div className="mini-table">
            {recentViolations.length ? (
              recentViolations.map((violation) => (
                <div key={violation.id} className="mini-row">
                  <div className="mini-row-main">
                    <span className={`type-pill ${violation.violation_type}`}>{detectorLabels[violation.violation_type]}</span>
                    <strong>{formatPlateNumber(violation.number_plate)}</strong>
                  </div>
                  <span className="muted-text">{formatTimestamp(violation.timestamp)}</span>
                </div>
              ))
            ) : (
              <div className="empty-block">No saved violations yet.</div>
            )}
          </div>
        </div>
      </div>

      <div className="panel">
        <div className="panel-head">
          <div>
            <h3>Live Detectors</h3>
            <p>Start or stop the configured video detectors directly from the dashboard.</p>
          </div>
        </div>

        <div className="detector-grid">
          {(Object.keys(detectors) as ViolationType[]).map((type) => (
            <DetectorCard
              key={type}
              type={type}
              status={detectors[type]}
              starting={startingDetector === type}
              stopping={stoppingDetector === type}
              onStart={startDetector}
              onStop={stopDetector}
            />
          ))}
        </div>
      </div>
    </section>
  );
}
