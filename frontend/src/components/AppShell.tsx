import { BellDot, LayoutDashboard, LogOut, RefreshCcw, ShieldAlert, Video } from "lucide-react";
import { NavLink, Outlet, useOutletContext } from "react-router-dom";
import type { DashboardContextValue } from "../lib/types";

const navItems = [
  { to: "/dashboard", label: "Dashboard", icon: LayoutDashboard },
  { to: "/violations", label: "Violations", icon: ShieldAlert },
  { to: "/live-detection", label: "Live Operations", icon: Video },
];

export function useDashboardContext() {
  return useOutletContext<DashboardContextValue>();
}

export default function AppShell({ context }: { context: DashboardContextValue }) {

  return (
    <div className="app-shell">
      <aside className="sidebar">
        <div className="brand-block">
          <div className="brand-mark">TA</div>
          <div>
            <strong>TrafficAI</strong>
            <span>Operations Console</span>
          </div>
        </div>

        <nav className="sidebar-nav">
          <p className="sidebar-label">Navigation</p>
          {navItems.map((item) => (
            <NavLink
              key={item.to}
              to={item.to}
              className={({ isActive }) => `nav-button${isActive ? " active" : ""}`}
            >
              <span className="nav-icon">
                <item.icon size={16} />
              </span>
              <span>{item.label}</span>
            </NavLink>
          ))}
        </nav>

        <div className="sidebar-footer">
          <div className={`sync-chip${context.liveStatus === "Live sync active" ? " live" : ""}`}>
            <BellDot size={14} />
            <span>{context.liveStatus}</span>
          </div>
          <span>{context.lastUpdated}</span>
        </div>
      </aside>

      <div className="main-shell">
        <header className="topbar">
          <div>
            <strong>Traffic Violation Monitoring</strong>
            <span>Supervise detectors, control live runs, and review captured evidence from one workspace.</span>
          </div>
          <div className="topbar-actions">
            <div className="user-chip">{context.authUser}</div>
            <button className="secondary-button" onClick={context.refreshDashboard} disabled={context.refreshInFlight}>
              <RefreshCcw size={16} />
              <span>{context.refreshInFlight ? "Refreshing..." : "Refresh"}</span>
            </button>
            <button className="danger-button" onClick={() => void context.logout()}>
              <LogOut size={16} />
              <span>Logout</span>
            </button>
          </div>
        </header>

        <main className="content-shell">
          <Outlet context={context} />
        </main>
      </div>
    </div>
  );
}
