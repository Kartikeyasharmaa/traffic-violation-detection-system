const elements = {
    totalViolations: document.getElementById("totalViolations"),
    helmetViolations: document.getElementById("helmetViolations"),
    redLightViolations: document.getElementById("redLightViolations"),
    wrongSideViolations: document.getElementById("wrongSideViolations"),
    typeFilter: document.getElementById("typeFilter"),
    refreshButton: document.getElementById("refreshButton"),
    tableBody: document.getElementById("violationsTableBody"),
    liveStatus: document.getElementById("liveStatus"),
    lastUpdated: document.getElementById("lastUpdated"),
    startHelmetButton: document.getElementById("startHelmetButton"),
    startRedLightButton: document.getElementById("startRedLightButton"),
    startWrongSideButton: document.getElementById("startWrongSideButton"),
    helmetDetectorStatus: document.getElementById("helmetDetectorStatus"),
    redLightDetectorStatus: document.getElementById("redLightDetectorStatus"),
    wrongSideDetectorStatus: document.getElementById("wrongSideDetectorStatus"),
};

let refreshInFlight = false;
let autoRefreshHandle = null;
const statValues = new Map();
const detectorButtons = {
    helmet: elements.startHelmetButton,
    red_light: elements.startRedLightButton,
    wrong_side: elements.startWrongSideButton,
};
const detectorStatusPills = {
    helmet: elements.helmetDetectorStatus,
    red_light: elements.redLightDetectorStatus,
    wrong_side: elements.wrongSideDetectorStatus,
};

function formatViolationType(value) {
    return value
        .split("_")
        .map((part) => part.charAt(0).toUpperCase() + part.slice(1))
        .join(" ");
}

function parseBackendTimestamp(value) {
    if (!value) {
        return null;
    }

    const normalized = /(?:Z|[+-]\d{2}:\d{2})$/.test(value) ? value : `${value}Z`;
    const date = new Date(normalized);
    return Number.isNaN(date.getTime()) ? null : date;
}

function formatTimestamp(value) {
    const date = parseBackendTimestamp(value);
    return date
        ? date.toLocaleString("en-IN", {
              timeZone: "Asia/Kolkata",
              dateStyle: "medium",
              timeStyle: "medium",
          }) + " IST"
        : value;
}

function animateCount(element, nextValue) {
    const safeTarget = Number(nextValue ?? 0);
    const previousValue = statValues.get(element.id) ?? 0;
    if (previousValue === safeTarget) {
        element.textContent = String(safeTarget);
        return;
    }

    const start = performance.now();
    const duration = 650;

    function tick(now) {
        const progress = Math.min((now - start) / duration, 1);
        const eased = 1 - Math.pow(1 - progress, 3);
        const value = Math.round(previousValue + (safeTarget - previousValue) * eased);
        element.textContent = String(value);

        if (progress < 1) {
            window.requestAnimationFrame(tick);
            return;
        }

        statValues.set(element.id, safeTarget);
        element.textContent = String(safeTarget);
    }

    window.requestAnimationFrame(tick);
}

function renderStats(stats) {
    animateCount(elements.totalViolations, stats.total_violations ?? 0);
    animateCount(elements.helmetViolations, stats.helmet_violations ?? 0);
    animateCount(elements.redLightViolations, stats.red_light_violations ?? 0);
    animateCount(elements.wrongSideViolations, stats.wrong_side_violations ?? 0);
}

function renderViolations(items) {
    if (!items.length) {
        elements.tableBody.innerHTML = '<tr><td colspan="5" class="empty-state">No violations found for the selected filter.</td></tr>';
        return;
    }

    const rows = items
        .sort((left, right) => {
            const rightDate = parseBackendTimestamp(right.timestamp);
            const leftDate = parseBackendTimestamp(left.timestamp);
            return (rightDate?.getTime() ?? 0) - (leftDate?.getTime() ?? 0);
        })
        .map(
            (item, index) => `
                <tr class="record-row" style="--row-delay: ${Math.min(index, 8) * 0.05}s;">
                    <td>
                        ${
                            item.image_url
                                ? `
                                    <a class="evidence-link" href="${item.image_url}" target="_blank" rel="noopener noreferrer">
                                        <img
                                            class="evidence-thumb"
                                            src="${item.image_url}"
                                            alt="${item.violation_type} evidence"
                                            onerror="this.closest('a').replaceWith(Object.assign(document.createElement('span'), {className: 'empty-state', textContent: 'Image unavailable'}))"
                                        >
                                        <span class="evidence-open">Open Evidence</span>
                                    </a>
                                `
                                : '<span class="empty-state">Image unavailable</span>'
                        }
                    </td>
                    <td>
                        <div class="plate-block">
                            <span class="plate-label">Plate Number</span>
                            <span class="plate-value">${item.number_plate || "UNKNOWN"}</span>
                            <span class="vehicle-meta">Record ID #${item.id}</span>
                        </div>
                    </td>
                    <td><span class="pill ${item.violation_type}">${formatViolationType(item.violation_type)}</span></td>
                    <td><span class="timestamp-value">${formatTimestamp(item.timestamp)}</span></td>
                    <td><button class="delete-button" data-id="${item.id}">Delete</button></td>
                </tr>
            `
        )
        .join("");

    elements.tableBody.innerHTML = rows;
    document.querySelectorAll(".delete-button").forEach((button) => {
        button.addEventListener("click", () => deleteViolation(button.dataset.id));
    });
}

async function fetchStats() {
    const response = await fetch("/stats");
    if (!response.ok) {
        throw new Error("Failed to load statistics.");
    }
    return response.json();
}

async function fetchViolations() {
    const selectedType = elements.typeFilter.value;
    const query = selectedType === "all" ? "" : `?type=${encodeURIComponent(selectedType)}`;
    const response = await fetch(`/violations${query}`);
    if (!response.ok) {
        throw new Error("Failed to load violations.");
    }
    return response.json();
}

async function fetchDetectors() {
    const response = await fetch("/detectors");
    if (!response.ok) {
        throw new Error("Failed to load detector status.");
    }
    return response.json();
}

function renderDetectorStatuses(payload) {
    const detectors = payload?.detectors ?? [];
    const mapped = new Map(detectors.map((item) => [item.detector_type, item]));

    ["helmet", "red_light", "wrong_side"].forEach((type) => {
        const item = mapped.get(type);
        const button = detectorButtons[type];
        const pill = detectorStatusPills[type];
        if (!button || !pill) {
            return;
        }

        const running = Boolean(item?.running);
        button.disabled = running;
        button.textContent =
            type === "helmet"
                ? running ? "Helmet Running" : "Start Helmet Video"
                : type === "red_light"
                ? running ? "Red Light Running" : "Start Red Light Video"
                : running ? "Wrong Side Running" : "Start Wrong Side Video";

        pill.textContent = running ? `Running${item?.pid ? ` | PID ${item.pid}` : ""}` : "Idle";
        pill.className = `detector-status ${running ? "running" : "idle"}`;
    });
}

async function startDetector(type) {
    const button = detectorButtons[type];
    if (button) {
        button.disabled = true;
        button.textContent = "Starting...";
    }

    try {
        const response = await fetch(`/detectors/${encodeURIComponent(type)}/start`, { method: "POST" });
        const payload = await response.json();
        if (!response.ok) {
            throw new Error(payload.detail || `Could not start ${type} detector.`);
        }

        renderDetectorStatuses({ detectors: [payload] });
        await refreshDashboard();
    } catch (error) {
        window.alert(error.message);
    } finally {
        await syncDetectorStatuses();
    }
}

async function deleteViolation(id) {
    const confirmed = window.confirm("Delete this violation record?");
    if (!confirmed) {
        return;
    }

    const response = await fetch(`/violations/${id}`, { method: "DELETE" });
    if (!response.ok) {
        window.alert("Could not delete the violation.");
        return;
    }

    await refreshDashboard();
}

async function refreshDashboard() {
    if (refreshInFlight) {
        return;
    }

    refreshInFlight = true;

    try {
        const [stats, violations, detectors] = await Promise.all([fetchStats(), fetchViolations(), fetchDetectors()]);
        renderStats(stats);
        renderViolations(violations);
        renderDetectorStatuses(detectors);
        elements.liveStatus.textContent = "Live sync active";
        elements.lastUpdated.textContent = `Last updated ${new Date().toLocaleTimeString("en-IN", {
            timeZone: "Asia/Kolkata",
        })} IST`;
    } catch (error) {
        if (!elements.tableBody.children.length) {
            elements.tableBody.innerHTML = `<tr><td colspan="5" class="empty-state">${error.message}</td></tr>`;
        }
        elements.liveStatus.textContent = "Backend disconnected";
        elements.lastUpdated.textContent = error.message;
    } finally {
        refreshInFlight = false;
    }
}

async function syncDetectorStatuses() {
    try {
        const detectors = await fetchDetectors();
        renderDetectorStatuses(detectors);
    } catch (error) {
        Object.values(detectorStatusPills).forEach((pill) => {
            if (!pill) {
                return;
            }
            pill.textContent = "Unavailable";
            pill.className = "detector-status idle";
        });
    }
}

elements.typeFilter.addEventListener("change", refreshDashboard);
elements.refreshButton.addEventListener("click", refreshDashboard);
elements.startHelmetButton?.addEventListener("click", () => startDetector("helmet"));
elements.startRedLightButton?.addEventListener("click", () => startDetector("red_light"));
elements.startWrongSideButton?.addEventListener("click", () => startDetector("wrong_side"));

window.addEventListener("load", () => {
    document.body.classList.add("app-ready");
});

refreshDashboard();
autoRefreshHandle = window.setInterval(refreshDashboard, 4000);
