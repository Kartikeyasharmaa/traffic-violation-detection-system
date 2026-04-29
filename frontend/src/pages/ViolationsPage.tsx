import { Search, Trash2, X } from "lucide-react";
import { useMemo, useState } from "react";
import EvidenceThumb from "../components/EvidenceThumb";
import { useDashboardContext } from "../components/AppShell";
import type { Violation, ViolationType } from "../lib/types";

const violationLabels: Record<ViolationType, string> = {
  helmet: "Helmet",
  red_light: "Red Light",
  wrong_side: "Wrong Side",
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

function ViolationModal({ violation, onClose }: { violation: Violation; onClose: () => void }) {
  return (
    <div className="modal-backdrop" onClick={onClose}>
      <div className="modal-card" onClick={(event) => event.stopPropagation()}>
        <div className="modal-head">
          <div>
            <h3>{formatPlateNumber(violation.number_plate)}</h3>
            <p>{violationLabels[violation.violation_type]}</p>
          </div>
          <button className="icon-button" onClick={onClose}>
            <X size={16} />
          </button>
        </div>
        <div className="modal-image">
          <EvidenceThumb violation={violation} />
        </div>
        <div className="modal-meta">
          <div>
            <span className="muted-label">Timestamp</span>
            <strong>{formatTimestamp(violation.timestamp)}</strong>
          </div>
          <div>
            <span className="muted-label">Detected Plate</span>
            <strong>{formatPlateNumber(violation.number_plate)}</strong>
          </div>
          <div>
            <span className="muted-label">Record ID</span>
            <strong>#{violation.id}</strong>
          </div>
        </div>
      </div>
    </div>
  );
}

export default function ViolationsPage() {
  const { violations, deleteViolation } = useDashboardContext();
  const [search, setSearch] = useState("");
  const [typeFilter, setTypeFilter] = useState<"all" | ViolationType>("all");
  const [sortOrder, setSortOrder] = useState<"desc" | "asc">("desc");
  const [selectedViolation, setSelectedViolation] = useState<Violation | null>(null);

  const filteredViolations = useMemo(() => {
    let items = [...violations];
    if (typeFilter !== "all") {
      items = items.filter((item) => item.violation_type === typeFilter);
    }
    if (search.trim()) {
      const query = search.trim().toLowerCase();
      items = items.filter((item) => (item.number_plate || "unknown").toLowerCase().includes(query));
    }
    items.sort((left, right) => {
      const leftTime = new Date(left.timestamp.endsWith("Z") ? left.timestamp : `${left.timestamp}Z`).getTime();
      const rightTime = new Date(right.timestamp.endsWith("Z") ? right.timestamp : `${right.timestamp}Z`).getTime();
      return sortOrder === "asc" ? leftTime - rightTime : rightTime - leftTime;
    });
    return items;
  }, [violations, search, typeFilter, sortOrder]);

  return (
    <section className="page-section">
      <div className="page-heading">
        <div>
          <p className="eyebrow">Records</p>
          <h1>Violations</h1>
          <p className="page-copy">Search, inspect, and manage saved evidence records from all detector modules.</p>
        </div>
      </div>

      <div className="panel">
        <div className="toolbar">
          <div className="search-wrap">
            <Search size={16} />
            <input
              className="text-input"
              type="text"
              placeholder="Search by number plate"
              value={search}
              onChange={(event) => setSearch(event.target.value)}
            />
          </div>
          <select className="select-input" value={typeFilter} onChange={(event) => setTypeFilter(event.target.value as "all" | ViolationType)}>
            <option value="all">All Types</option>
            <option value="helmet">Helmet</option>
            <option value="red_light">Red Light</option>
            <option value="wrong_side">Wrong Side</option>
          </select>
          <select className="select-input" value={sortOrder} onChange={(event) => setSortOrder(event.target.value as "desc" | "asc")}>
            <option value="desc">Newest First</option>
            <option value="asc">Oldest First</option>
          </select>
        </div>

        <div className="table-shell">
          <table className="data-table">
            <thead>
              <tr>
                <th>Plate Number</th>
                <th>Type</th>
                <th>Timestamp</th>
                <th>Image</th>
                <th>Action</th>
              </tr>
            </thead>
            <tbody>
              {filteredViolations.length ? (
                filteredViolations.map((violation) => (
                  <tr key={violation.id}>
                    <td>
                      <button className="row-link" onClick={() => setSelectedViolation(violation)}>
                        <strong>{formatPlateNumber(violation.number_plate)}</strong>
                        <span className="muted-text">Record #{violation.id}</span>
                      </button>
                    </td>
                    <td>
                      <span className={`type-pill ${violation.violation_type}`}>{violationLabels[violation.violation_type]}</span>
                    </td>
                    <td>{formatTimestamp(violation.timestamp)}</td>
                    <td>
                      <EvidenceThumb violation={violation} compact />
                    </td>
                    <td>
                      <button className="danger-button" onClick={() => deleteViolation(violation.id)}>
                        <Trash2 size={16} />
                        <span>Delete</span>
                      </button>
                    </td>
                  </tr>
                ))
              ) : (
                <tr>
                  <td colSpan={5} className="empty-block">
                    No violations found for the current filters.
                  </td>
                </tr>
              )}
            </tbody>
          </table>
        </div>
      </div>

      {selectedViolation ? <ViolationModal violation={selectedViolation} onClose={() => setSelectedViolation(null)} /> : null}
    </section>
  );
}
