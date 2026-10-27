"""Physics model of a LEO CubeSat: orbit/eclipse geometry, power and thermal subsystems.

Default parameters approximate ESA OPS-SAT (3U CubeSat, ~515 km sun-synchronous orbit).
Eclipse uses the cylindrical Earth-shadow model; power uses an energy-balance battery;
thermal uses a single-node lumped-capacitance model with radiative cooling.
"""
from dataclasses import dataclass, field

import numpy as np

MU = 3.986004418e14       # Earth gravitational parameter [m^3/s^2]
R_E = 6_371_000.0         # mean Earth radius [m]
SOLAR = 1361.0            # solar constant [W/m^2]
SIGMA = 5.670374419e-8    # Stefan-Boltzmann [W/m^2/K^4]
EARTH_IR = 237.0          # mean Earth OLR [W/m^2]
OMEGA_E = 7.2921159e-5    # Earth rotation rate [rad/s]

# downlink rates are illustrative values for a CubeSat S-band link
GROUND_STATIONS = [
    {"name": "ESOC-Darmstadt", "lat": 49.87, "lon": 8.62, "rate_mbps": 1.0},
    {"name": "Kiruna", "lat": 67.86, "lon": 20.96, "rate_mbps": 1.0},
    {"name": "Svalbard", "lat": 78.23, "lon": 15.41, "rate_mbps": 2.0},
    {"name": "ISTRAC-Bengaluru", "lat": 13.03, "lon": 77.51, "rate_mbps": 0.5},
]


@dataclass
class OrbitModel:
    altitude_m: float = 515e3
    beta_deg: float = 20.0   # angle between orbit plane and Sun vector

    @property
    def a(self) -> float:
        return R_E + self.altitude_m

    @property
    def period_s(self) -> float:
        return 2 * np.pi * np.sqrt(self.a ** 3 / MU)

    def eclipse_fraction_analytic(self) -> float:
        """Closed-form fraction of the orbit spent in Earth's cylindrical shadow."""
        h, b = self.altitude_m, np.radians(self.beta_deg)
        beta_star = np.arcsin(R_E / self.a)
        if abs(b) >= beta_star:
            return 0.0
        return float(np.arccos(np.sqrt(h ** 2 + 2 * R_E * h) / (self.a * np.cos(b))) / np.pi)

    def position(self, t: np.ndarray) -> np.ndarray:
        """Satellite position [m] in a frame where the Sun lies along +x tilted by beta."""
        nu = 2 * np.pi * t / self.period_s
        return self.a * np.stack([np.cos(nu), np.sin(nu), np.zeros_like(nu)], axis=-1)

    def sun_vector(self) -> np.ndarray:
        b = np.radians(self.beta_deg)
        return np.array([np.cos(b), 0.0, np.sin(b)])

    def sunlit(self, t: np.ndarray) -> np.ndarray:
        r = self.position(t)
        s = self.sun_vector()
        along = r @ s
        perp = np.linalg.norm(r - np.outer(along, s), axis=1)
        return ~((along < 0) & (perp < R_E))


@dataclass
class PowerModel:
    panel_area_m2: float = 0.09     # deployable panels of a 3U CubeSat
    efficiency: float = 0.28        # triple-junction GaAs
    load_w: float = 12.0            # average bus + payload consumption
    capacity_wh: float = 40.0
    soc0: float = 0.9

    def solar_power(self, sunlit: np.ndarray, degradation: np.ndarray | float = 0.0) -> np.ndarray:
        return SOLAR * self.panel_area_m2 * self.efficiency * sunlit * (1 - degradation)

    def simulate_soc(self, p_in: np.ndarray, dt_s: float, capacity_scale: np.ndarray | float = 1.0) -> np.ndarray:
        cap = self.capacity_wh * np.broadcast_to(capacity_scale, p_in.shape)
        soc = np.empty_like(p_in)
        e = self.soc0 * cap[0]
        for i, p in enumerate(p_in):
            e = np.clip(e + (p - self.load_w) * dt_s / 3600.0, 0.0, cap[i])
            soc[i] = e / cap[i]
        return soc


@dataclass
class ThermalModel:
    mass_kg: float = 4.0
    c_p: float = 900.0          # aluminium [J/kg/K]
    area_m2: float = 0.14       # radiating surface
    absorptivity: float = 0.6
    emissivity: float = 0.8
    t0_k: float = 290.0

    def simulate(self, sunlit: np.ndarray, q_internal_w: np.ndarray | float, dt_s: float) -> np.ndarray:
        q_int = np.broadcast_to(q_internal_w, sunlit.shape)
        temp = np.empty(sunlit.shape)
        t = self.t0_k
        c = self.mass_kg * self.c_p
        for i in range(len(sunlit)):
            q_in = (self.absorptivity * SOLAR * sunlit[i] * self.area_m2 / 4
                    + self.emissivity * EARTH_IR * self.area_m2 / 4 + q_int[i])
            q_out = self.emissivity * SIGMA * self.area_m2 * t ** 4
            t += (q_in - q_out) * dt_s / c
            temp[i] = t
        return temp


@dataclass
class Faults:
    """Fault injection. Each entry is (start_s, magnitude)."""
    panel_degradation: tuple[float, float] | None = None   # fraction of power lost
    battery_fade: tuple[float, float] | None = None        # fraction of capacity lost
    heater_stuck_on: tuple[float, float] | None = None     # extra watts dissipated
    sensor_stuck: tuple[float, str] | None = None          # (start_s, channel name)


@dataclass
class SatellitePhysics:
    orbit: OrbitModel = field(default_factory=OrbitModel)
    power: PowerModel = field(default_factory=PowerModel)
    thermal: ThermalModel = field(default_factory=ThermalModel)

    def simulate(self, n_orbits: float = 3, dt_s: float = 10.0, faults: Faults | None = None) -> dict:
        faults = faults or Faults()
        t = np.arange(0, n_orbits * self.orbit.period_s, dt_s)

        def step(spec):
            return 0.0 if spec is None else np.where(t >= spec[0], spec[1], 0.0)

        sunlit = self.orbit.sunlit(t)
        p_in = self.power.solar_power(sunlit, step(faults.panel_degradation))
        soc = self.power.simulate_soc(p_in, dt_s, 1 - step(faults.battery_fade))
        temp = self.thermal.simulate(sunlit, self.power.load_w * 0.5 + step(faults.heater_stuck_on), dt_s)

        out = {"t_s": t, "sunlit": sunlit.astype(int), "solar_w": p_in, "soc": soc, "temp_k": temp}
        if faults.sensor_stuck is not None:
            start, ch = faults.sensor_stuck
            idx = np.searchsorted(t, start)
            out[ch] = out[ch].astype(float).copy()
            out[ch][idx:] = out[ch][idx]
        fault_mask = np.zeros_like(t, dtype=bool)
        for spec in (faults.panel_degradation, faults.battery_fade, faults.heater_stuck_on, faults.sensor_stuck):
            if spec is not None:
                fault_mask |= t >= spec[0]
        out["fault"] = fault_mask.astype(int)
        return out

    def ground_passes(self, stations: list[dict] | None = None, hours: float = 24.0, dt_s: float = 10.0,
                      inclination_deg: float = 97.4, min_elevation_deg: float = 10.0,
                      raan_deg: float = 0.0) -> list[dict]:
        """Pass windows from geometry: circular inclined orbit (two-body, no J2 drift) and
        ground stations on a rotating spherical Earth. A pass is a contiguous interval with
        elevation above the mask. Returns start/end/duration/max-elevation per pass."""
        stations = stations or GROUND_STATIONS
        t = np.arange(0.0, hours * 3600.0, dt_s)
        a, i, raan = self.orbit.a, np.radians(inclination_deg), np.radians(raan_deg)
        u = 2 * np.pi * t / self.orbit.period_s
        r_sat = a * np.stack([np.cos(raan) * np.cos(u) - np.sin(raan) * np.sin(u) * np.cos(i),
                              np.sin(raan) * np.cos(u) + np.cos(raan) * np.sin(u) * np.cos(i),
                              np.sin(u) * np.sin(i)], axis=1)
        theta = OMEGA_E * t
        passes = []
        for st in stations:
            lat, lon = np.radians(st["lat"]), np.radians(st["lon"])
            n_hat = np.stack([np.cos(lat) * np.cos(lon + theta), np.cos(lat) * np.sin(lon + theta),
                              np.full_like(t, np.sin(lat))], axis=1)
            rel = r_sat - R_E * n_hat
            el = np.degrees(np.arcsin(np.einsum("ij,ij->i", rel, n_hat) / np.linalg.norm(rel, axis=1)))
            vis = el >= min_elevation_deg
            edges = np.flatnonzero(np.diff(np.concatenate([[0], vis.astype(int), [0]])))
            for s0, s1 in zip(edges[::2], edges[1::2]):
                passes.append({"station": st["name"], "start_s": float(t[s0]), "end_s": float(t[s1 - 1]),
                               "duration_s": float(t[s1 - 1] - t[s0] + dt_s),
                               "max_elev_deg": round(float(el[s0:s1].max()), 2),
                               "rate_mbps": st["rate_mbps"]})
        return sorted(passes, key=lambda p: p["start_s"])
