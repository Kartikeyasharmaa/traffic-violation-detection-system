export type ViolationType = "helmet" | "red_light" | "wrong_side";

export interface Violation {
  id: number;
  violation_type: ViolationType;
  number_plate: string | null;
  image_path: string;
  image_url: string | null;
  timestamp: string;
}

export interface Stats {
  total_violations: number;
  helmet_violations: number;
  red_light_violations: number;
  wrong_side_violations: number;
}

export interface DetectorStatus {
  detector_type: ViolationType;
  running: boolean;
  pid: number | null;
  started_at?: string | null;
  log_path?: string | null;
  already_running?: boolean;
}

export interface DetectorListResponse {
  detectors: DetectorStatus[];
}

export interface AuthSessionResponse {
  authenticated: boolean;
  username: string;
}

export interface DashboardContextValue {
  stats: Stats;
  violations: Violation[];
  detectors: Record<ViolationType, DetectorStatus>;
  liveStatus: string;
  lastUpdated: string;
  refreshInFlight: boolean;
  startingDetector: ViolationType | "";
  stoppingDetector: ViolationType | "";
  authUser: string | null;
  refreshDashboard: () => Promise<void>;
  startDetector: (type: ViolationType) => Promise<void>;
  stopDetector: (type: ViolationType) => Promise<void>;
  logout: () => Promise<void>;
  deleteViolation: (id: number) => Promise<void>;
}
