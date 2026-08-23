# Machine Technical Specification

| Field | Value |
|---|---|
| **Document ID** | TEC-328 |
| **Title** | Body-in-White Conveyor Line 2 (CV-2) – Technical Specification |
| **Equipment** | CV-2, Zones 2A–2C (Stations 42–51), Body Shop |
| **Revision** | Rev 9 |
| **Effective Date** | 2024-12-02 |
| **Owner** | Maintenance Manager, Tom Okafor |
| **Classification** | Internal – Controlled |

---

## 1. General Description

Powered roller/pallet conveyor carrying body-in-white shells through the weld stations. Two parallel lanes (2A, 2B) merge at the Zone 2B transfer point.

## 2. Specifications

| Parameter | Value |
|---|---|
| Line speed (max) | 12 m/min (variable 0–12) |
| Pallet capacity | 36 pallets per lane |
| Pallet size | 2,400 mm x 1,200 mm |
| Max body weight per pallet | 950 kg |
| Drive | 22 kW AC gearmotors, 8 drives |
| PLC | Nexus NX-3000 (Profinet) |
| HMI | Nexus Panel-15 (Zone 2B and 2C) |
| Sensors | Photoelectric PS-200 (entry/exit), proximity PR-100 (lift) |
| E-Stop zones | 3 (2A, 2B, 2C), 18 E-Stop buttons total |

## 3. Sensor Map (Zone 2B)

| Sensor | Location | Function |
|---|---|---|
| PS-200-1 | Zone 2B entry | Pallet present for transfer |
| PS-200-2 | Zone 2B exit | Transfer complete |
| PR-100-1 | Lift section, low | Lift at home |
| PR-100-2 | Lift section, high | Lift raised |

## 4. Interlocks

- Transfer from lane 2A to 2B requires PS-200-1 AND PR-100-2 (lift raised).
- E-Stop in any zone stops all drives in that zone; adjacent zones decelerate to stop.
- LOTO points: main disconnect at Zone 2B + 8 local disconnects.

## 5. Maintenance Intervals

| Task | Interval |
|---|---|
| Sensor cleaning (PS-200/PR-100) | Weekly (weld spatter build-up) |
| Pallet datum pin inspection | Every 2 weeks |
| Drive gearbox oil | Every 6 months |
| Chain tension check | Monthly |
| Full conveyor alignment survey | Annually |

## 6. Known Recurring Issues

- Weld spatter accumulation on PS-200-1 is the most common cause of false jam alarms (see SOP-TEC-228).
- Pallet datum pin wear > 1 mm causes body misalignment at Station 51 and triggers the robot's TCP deviation fault (E-5023).

## 7. Cross-References

- SOP-TEC-228 — Conveyor CV-2 Jam Recovery and E-Stop Reset
- TEC-301 — RoboTech RX-700 Technical Specification
- ESC-402 — Escalation Workflow Matrix

## 8. Revision History

| Rev | Date | Change |
|---|---|---|
| 7 | 2024-06-18 | Initial release |
| 8 | 2024-09-09 | Added sensor map for Zone 2B |
| 9 | 2024-12-02 | Documented recurring spatter issue and pin wear link |

---
*Uncontrolled copy if printed. Verify revision on the document portal before use.*
