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

    def ground_passes(self, stations: list[tuple[str, float]], n_orbits: int = 6) -> list[dict]:
        """Simplified pass windows: a station at phase angle phi (deg along the ground track)
        is visible when the satellite is within the half-angle cone set by a 10 deg
        elevation mask. Ground-track drift between orbits is modelled by rotating each
        station's phase by Earth's rotation during one period."""
        T = self.orbit.period_s
        h = self.orbit.altitude_m
        el = np.radians(10)
        # Earth central angle of the visibility circle
        lam = np.arccos(R_E / (R_E + h) * np.cos(el)) - el
        half_window_s = lam / (2 * np.pi) * T
        drift = 360.0 * T / 86164.0          # deg of Earth rotation per orbit
        passes = []
        for k in range(n_orbits):
            for name, phi in stations:
                phase = (phi + k * drift) % 360
                # visibility only on orbits whose ground track comes close enough
                if np.cos(np.radians(phase - 180)) < 0.2:
                    continue
                centre = k * T + phase / 360 * T
                passes.append({"station": name, "orbit": k,
                               "start_s": round(centre - half_window_s, 1),
                               "end_s": round(centre + half_window_s, 1)})
        return sorted(passes, key=lambda p: p["start_s"])
