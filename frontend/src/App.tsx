import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { HashRouter, Navigate, Route, Routes } from "react-router-dom";
import AppShell from "./components/AppShell";
import DashboardPage from "./pages/DashboardPage";
import LiveDetectionPage from "./pages/LiveDetectionPage";
import LoginPage from "./pages/LoginPage";
import ViolationsPage from "./pages/ViolationsPage";
import {
  deleteViolationRequest,
  fetchDetectors,
  fetchStats,
  fetchViolations,
  startDetectorRequest,
  stopDetectorRequest,
} from "./lib/api";
import type { DashboardContextValue, DetectorStatus, Stats, Violation, ViolationType } from "./lib/types";

function buildDetectorMap(detectors: DetectorStatus[]): Record<ViolationType, DetectorStatus> {
  const base: Record<ViolationType, DetectorStatus> = {
    helmet: { detector_type: "helmet", running: false, pid: null },
    red_light: { detector_type: "red_light", running: false, pid: null },
    wrong_side: { detector_type: "wrong_side", running: false, pid: null },
  };

  detectors.forEach((detector) => {
    base[detector.detector_type] = detector;
  });

  return base;
}

const LOCAL_AUTH_USER_KEY = "traffic_detection_auth_user";
const LOCAL_LOGIN_USERNAME = "admin";
const LOCAL_LOGIN_PASSWORD = "traffic123";

export default function App() {
  const [stats, setStats] = useState<Stats>({
    total_violations: 0,
    helmet_violations: 0,
    red_light_violations: 0,
    wrong_side_violations: 0,
  });
  const [violations, setViolations] = useState<Violation[]>([]);
  const [detectors, setDetectors] = useState<Record<ViolationType, DetectorStatus>>({
    helmet: { detector_type: "helmet", running: false, pid: null },
    red_light: { detector_type: "red_light", running: false, pid: null },
    wrong_side: { detector_type: "wrong_side", running: false, pid: null },
  });
  const [liveStatus, setLiveStatus] = useState("Checking session");
  const [lastUpdated, setLastUpdated] = useState("Waiting for first sync");
  const [refreshInFlight, setRefreshInFlight] = useState(false);
  const [startingDetector, setStartingDetector] = useState<ViolationType | "">("");
  const [stoppingDetector, setStoppingDetector] = useState<ViolationType | "">("");
  const [authUser, setAuthUser] = useState<string | null>(null);
  const [authChecked, setAuthChecked] = useState(true);
  const refreshLock = useRef(false);

  const refreshDashboard = useCallback(async () => {
    if (!authUser || refreshLock.current) {
      return;
    }

    refreshLock.current = true;
    setRefreshInFlight(true);
    try {
      const [statsPayload, violationsPayload, detectorsPayload] = await Promise.all([
        fetchStats(),
        fetchViolations(),
        fetchDetectors(),
      ]);
      setStats(statsPayload);
      setViolations(violationsPayload);
      setDetectors(buildDetectorMap(detectorsPayload.detectors));
      setLiveStatus("Live sync active");
      setLastUpdated(
        `Last updated ${new Date().toLocaleTimeString("en-IN", {
          timeZone: "Asia/Kolkata",
        })} IST`
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : "Refresh failed";
      setLiveStatus("Backend disconnected");
      setLastUpdated(message);
    } finally {
      refreshLock.current = false;
      setRefreshInFlight(false);
    }
  }, [authUser]);

  const startDetector = useCallback(
    async (type: ViolationType) => {
      setStartingDetector(type);
      try {
        const payload = await startDetectorRequest(type);
        setDetectors((current) => ({
          ...current,
          [type]: payload,
        }));
        await refreshDashboard();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not start detector.";
        window.alert(message);
      } finally {
        setStartingDetector("");
      }
    },
    [refreshDashboard]
  );

  const stopDetector = useCallback(
    async (type: ViolationType) => {
      setStoppingDetector(type);
      try {
        const payload = await stopDetectorRequest(type);
        setDetectors((current) => ({
          ...current,
          [type]: payload,
        }));
        await refreshDashboard();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not stop detector.";
        window.alert(message);
      } finally {
        setStoppingDetector("");
      }
    },
    [refreshDashboard]
  );

  const deleteViolation = useCallback(
    async (id: number) => {
      const confirmed = window.confirm("Delete this violation record?");
      if (!confirmed) {
        return;
      }

      try {
        await deleteViolationRequest(id);
        await refreshDashboard();
      } catch (error) {
        const message = error instanceof Error ? error.message : "Could not delete the violation.";
        window.alert(message);
      }
    },
    [refreshDashboard]
  );

  const logout = useCallback(async () => {
    window.localStorage.removeItem(LOCAL_AUTH_USER_KEY);
    setAuthUser(null);
    setDetectors(buildDetectorMap([]));
    setViolations([]);
    setStats({
      total_violations: 0,
      helmet_violations: 0,
      red_light_violations: 0,
      wrong_side_violations: 0,
    });
    setLiveStatus("Signed out");
    setLastUpdated("Session closed");
  }, []);

  const handleLogin = useCallback(async (username: string, password: string) => {
    if (username !== LOCAL_LOGIN_USERNAME || password !== LOCAL_LOGIN_PASSWORD) {
      throw new Error("The username or password is incorrect.");
    }

    window.localStorage.setItem(LOCAL_AUTH_USER_KEY, username);
    setAuthUser(username);
    setLiveStatus("Live sync active");
    setLastUpdated("Session opened");
  }, []);

  useEffect(() => {
    const storedUser = window.localStorage.getItem(LOCAL_AUTH_USER_KEY);
    if (storedUser) {
      setAuthUser(storedUser);
      setLiveStatus("Live sync active");
    } else {
      setLiveStatus("Login required");
    }
  }, []);

  useEffect(() => {
    if (!authUser) {
      return;
    }

    void refreshDashboard();
    const handle = window.setInterval(() => {
      void refreshDashboard();
    }, 4000);
    return () => window.clearInterval(handle);
  }, [authUser, refreshDashboard]);

  const contextValue = useMemo<DashboardContextValue>(
    () => ({
      stats,
      violations,
      detectors,
      liveStatus,
      lastUpdated,
      refreshInFlight,
      startingDetector,
      stoppingDetector,
      authUser,
      refreshDashboard,
      startDetector,
      stopDetector,
      logout,
      deleteViolation,
    }),
    [
      stats,
      violations,
      detectors,
      liveStatus,
      lastUpdated,
      refreshInFlight,
      startingDetector,
      stoppingDetector,
      authUser,
      refreshDashboard,
      startDetector,
      stopDetector,
      logout,
      deleteViolation,
    ]
  );

  if (!authChecked) {
    return <div className="app-loading">Checking secure session...</div>;
  }

  if (!authUser) {
    return <LoginPage onLogin={handleLogin} />;
  }

  return (
    <HashRouter>
      <Routes>
        <Route element={<AppShell context={contextValue} />}>
          <Route index element={<Navigate to="/dashboard" replace />} />
          <Route path="/dashboard" element={<DashboardPage />} />
          <Route path="/violations" element={<ViolationsPage />} />
          <Route path="/live-detection" element={<LiveDetectionPage />} />
        </Route>
      </Routes>
    </HashRouter>
  );
}
