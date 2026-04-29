import { Play, Square } from "lucide-react";
import { useDashboardContext } from "../components/AppShell";
import type { DetectorStatus, ViolationType } from "../lib/types";

const detectorLabels: Record<ViolationType, string> = {
  helmet: "Helmet",
  red_light: "Red Light",
  wrong_side: "Wrong Side",
};

const detectorDescriptions: Record<ViolationType, string> = {
  helmet: "Runs the helmet violation video with rider, head, and plate marking.",
  red_light: "Runs the red-light video with signal, stop-line, and vehicle tracking.",
  wrong_side: "Runs the wrong-side video with direction tracking and evidence capture.",
};

function DetectorConsoleCard({
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
    <article className="detector-card detector-console-card">
      <div className="detector-card-top">
        <span className="detector-badge">{detectorLabels[type]}</span>
        <span className={`status-chip ${running ? "running" : "idle"}`}>{running ? "Running" : "Idle"}</span>
      </div>
      <h3>{detectorLabels[type]} Live Detection</h3>
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

export default function LiveDetectionPage() {
  const { detectors, startingDetector, stoppingDetector, startDetector, stopDetector } = useDashboardContext();

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Live Operations</p>
          <h1>Detection Operations</h1>
          <p className="page-copy">Start or stop each detector from one control panel.</p>
        </div>
      </div>

      <div className="panel live-console-panel">
        <div className="live-console-header">
          <h3>Detector Control</h3>
          <p>Each active detector opens a desktop preview window and saves evidence to the dashboard automatically.</p>
        </div>

        <div className="live-console-grid">
          {(Object.keys(detectors) as ViolationType[]).map((type) => (
            <DetectorConsoleCard
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
