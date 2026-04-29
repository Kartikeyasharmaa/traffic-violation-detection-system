import type { AuthSessionResponse, DetectorListResponse, DetectorStatus, Stats, Violation, ViolationType } from "./types";

async function readJson<T>(response: Response): Promise<T> {
  if (!response.ok) {
    let detail = `Request failed with status ${response.status}`;
    try {
      const payload = await response.json();
      detail = payload?.detail || detail;
    } catch {
      return Promise.reject(new Error(detail));
    }
    return Promise.reject(new Error(detail));
  }

  return response.json() as Promise<T>;
}

export async function fetchStats(): Promise<Stats> {
  const response = await fetch("/stats", { credentials: "same-origin" });
  return readJson<Stats>(response);
}

export async function fetchViolations(): Promise<Violation[]> {
  const response = await fetch("/violations?sort=desc", { credentials: "same-origin" });
  return readJson<Violation[]>(response);
}

export async function fetchDetectors(): Promise<DetectorListResponse> {
  const response = await fetch("/detectors", { credentials: "same-origin" });
  return readJson<DetectorListResponse>(response);
}

export async function startDetectorRequest(type: ViolationType): Promise<DetectorStatus> {
  const response = await fetch(`/detectors/${encodeURIComponent(type)}/start`, {
    method: "POST",
    credentials: "same-origin",
  });
  return readJson<DetectorStatus>(response);
}

export async function stopDetectorRequest(type: ViolationType): Promise<DetectorStatus> {
  const response = await fetch(`/detectors/${encodeURIComponent(type)}/stop`, {
    method: "POST",
    credentials: "same-origin",
  });
  return readJson<DetectorStatus>(response);
}

export async function deleteViolationRequest(id: number): Promise<void> {
  const response = await fetch(`/violations/${id}`, { method: "DELETE", credentials: "same-origin" });
  await readJson<{ message: string; id: number }>(response);
}

export async function loginRequest(username: string, password: string): Promise<AuthSessionResponse> {
  const response = await fetch("/auth/login", {
    method: "POST",
    credentials: "same-origin",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ username, password }),
  });
  return readJson<AuthSessionResponse>(response);
}

export async function logoutRequest(): Promise<void> {
  const response = await fetch("/auth/logout", {
    method: "POST",
    credentials: "same-origin",
  });
  await readJson<{ authenticated: boolean }>(response);
}

export async function fetchAuthMe(): Promise<AuthSessionResponse> {
  const response = await fetch("/auth/me", { credentials: "same-origin" });
  return readJson<AuthSessionResponse>(response);
}
