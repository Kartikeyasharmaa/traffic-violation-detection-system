import { useState } from "react";
import type { Violation } from "../lib/types";

interface EvidenceThumbProps {
  violation: Violation;
  compact?: boolean;
}

export default function EvidenceThumb({ violation, compact = false }: EvidenceThumbProps) {
  const [missing, setMissing] = useState(false);

  if (!violation.image_url || missing) {
    return <div className={compact ? "evidence-fallback compact" : "evidence-fallback"}>No image</div>;
  }

  return (
    <a
      className={compact ? "evidence-link compact" : "evidence-link"}
      href={violation.image_url}
      target="_blank"
      rel="noreferrer"
    >
      <img
        className={compact ? "evidence-thumb compact" : "evidence-thumb"}
        src={violation.image_url}
        alt={`${violation.violation_type} evidence`}
        onError={() => setMissing(true)}
      />
      {!compact && <span className="evidence-open">Open Evidence</span>}
    </a>
  );
}
