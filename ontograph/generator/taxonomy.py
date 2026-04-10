"""
ontograph/generator/taxonomy.py — Aerospace TBox definitions.

Pure Python dataclasses — no rdflib, no LLM.

The AEROSPACE_TAXONOMY singleton is the single source of truth for all class
names, property names, and domain-to-class mappings used by the generator module.

Class hierarchy: component leaf classes are nested directly under their corresponding
subsystem (e.g. SolarPanel rdfs:subClassOf PowerSubsystem). The separate "Component"
root and intermediate category classes (PowerComponent, Antenna, etc.) are intentionally
absent — those intermediate abstractions were never instantiated and produced empty classes.

Four predefined domains (cubesat, uam, rocket, lunar) have explicit subsystem
guidance so the LLM receives a focused vocabulary. Any other domain string is
treated as a "custom domain" and given the full taxonomy vocabulary.
"""

from __future__ import annotations

from dataclasses import dataclass, field

_ALL = ["cubesat", "uam", "rocket", "lunar"]
_CU  = ["cubesat"]
_UA  = ["uam"]
_RO  = ["rocket"]
_LU  = ["lunar"]
_CULU = ["cubesat", "lunar"]
_ROLU = ["rocket", "lunar"]
_RUA  = ["rocket", "uam"]
_GNCD = ["rocket", "uam", "lunar"]


@dataclass
class ClassDef:
    local: str
    label: str
    parent: str | None
    description: str
    domains: list[str] = field(default_factory=list)
    is_component: bool = False
    """True for leaf-level component classes that are instantiated within subsystems."""


@dataclass
class DataPropDef:
    local: str
    label: str
    xsd_type: str       # "decimal" | "string" | "integer"
    unit: str | None
    description: str


@dataclass
class ObjectPropDef:
    local: str
    label: str
    domain_class: str
    range_class: str
    description: str


@dataclass
class AerospaceTaxonomy:
    classes: list[ClassDef]
    data_properties: list[DataPropDef]
    object_properties: list[ObjectPropDef]
    domain_required_subsystems: dict[str, list[str]]
    domain_system_classes: dict[str, list[str]]

    def classes_for_domain(self, domain: str) -> list[ClassDef]:
        """Return classes whose domains list includes the given domain."""
        return [c for c in self.classes if domain in c.domains]

    def get_class(self, local: str) -> ClassDef | None:
        """Look up a class by local name."""
        return next((c for c in self.classes if c.local == local), None)


# Predefined domains that receive focused LLM prompts with explicit subsystem guidance.
# Any other string is a "custom domain" that receives the full taxonomy vocabulary.
PREDEFINED_DOMAINS: set[str] = {"cubesat", "uam", "rocket", "lunar"}


AEROSPACE_TAXONOMY = AerospaceTaxonomy(

    # ── Classes ──────────────────────────────────────────────────────────────
    classes=[

        # ── System roots ─────────────────────────────────────────────────────
        ClassDef(
            "AerospaceSystem", "Aerospace System", None,
            "Abstract root class for all aerospace systems.",
            _ALL,
        ),
        ClassDef(
            "SpacecraftSystem", "Spacecraft System", "AerospaceSystem",
            "Abstract class for space-based systems.",
            ["cubesat", "rocket", "lunar"],
        ),
        ClassDef(
            "AirVehicle", "Air Vehicle", "AerospaceSystem",
            "Abstract class for atmospheric vehicle systems.",
            _UA,
        ),

        # ── Satellite / CubeSat ───────────────────────────────────────────────
        ClassDef(
            "Satellite", "Satellite", "SpacecraftSystem",
            "Abstract class for orbital satellite systems.",
            _CU,
        ),
        ClassDef(
            "NanoSatellite", "Nano-Satellite (CubeSat 1U-3U)", "Satellite",
            "Small CubeSat platform with 1U to 3U form factor.",
            _CU,
        ),
        ClassDef(
            "MicroSatellite", "Micro-Satellite (CubeSat 6U-27U)", "Satellite",
            "Larger CubeSat platform with 6U to 27U form factor.",
            _CU,
        ),

        # ── Launch Vehicle ────────────────────────────────────────────────────
        ClassDef(
            "LaunchVehicle", "Launch Vehicle", "SpacecraftSystem",
            "Abstract class for rocket launch vehicles.",
            _RO,
        ),
        ClassDef(
            "SmallLaunchVehicle", "Small Launch Vehicle", "LaunchVehicle",
            "Launch vehicle for payloads up to ~1000 kg to LEO.",
            _RO,
        ),
        ClassDef(
            "HeavyLiftVehicle", "Heavy-Lift Launch Vehicle", "LaunchVehicle",
            "Launch vehicle for payloads over 20,000 kg to LEO.",
            _RO,
        ),

        # ── Lunar ─────────────────────────────────────────────────────────────
        ClassDef(
            "LunarExplorationSystem", "Lunar Exploration System", "SpacecraftSystem",
            "Abstract class for lunar exploration spacecraft.",
            _LU,
        ),
        ClassDef(
            "LunarLander", "Lunar Lander", "LunarExplorationSystem",
            "Spacecraft designed to land on the lunar surface.",
            _LU,
        ),
        ClassDef(
            "LunarOrbiter", "Lunar Orbiter", "LunarExplorationSystem",
            "Spacecraft designed to orbit the Moon.",
            _LU,
        ),

        # ── UAM ───────────────────────────────────────────────────────────────
        ClassDef(
            "UrbanAirMobility", "Urban Air Mobility Vehicle", "AirVehicle",
            "Abstract class for urban air mobility platforms.",
            _UA,
        ),
        ClassDef(
            "eVTOL", "Electric VTOL", "UrbanAirMobility",
            "Electric vertical takeoff and landing vehicle for UAM.",
            _UA,
        ),

        # ── Subsystem root ────────────────────────────────────────────────────
        ClassDef(
            "Subsystem", "Subsystem", None,
            "Abstract root for all spacecraft or vehicle subsystems.",
            _ALL,
        ),
        ClassDef(
            "PowerSubsystem", "Power Subsystem", "Subsystem",
            "Electrical power generation, storage, and distribution.",
            _ALL,
        ),
        ClassDef(
            "PropulsionSubsystem", "Propulsion Subsystem", "Subsystem",
            "Abstract class for propulsion systems.",
            _ALL,
        ),
        ClassDef(
            "ChemicalPropulsionSystem", "Chemical Propulsion System", "PropulsionSubsystem",
            "Chemical thruster-based propulsion (monoprop or biprop).",
            _ROLU,
        ),
        ClassDef(
            "ElectricPropulsionSystem", "Electric Propulsion System", "PropulsionSubsystem",
            "Ion/Hall thruster or electric motor based propulsion.",
            ["cubesat", "uam"],
        ),
        ClassDef(
            "CommunicationSubsystem", "Communication Subsystem", "Subsystem",
            "RF communication, antennas, and data link systems.",
            _ALL,
        ),
        ClassDef(
            "AttitudeControlSubsystem", "Attitude Control Subsystem (ADCS)", "Subsystem",
            "Attitude determination and control using reaction wheels and sensors.",
            _CULU,
        ),
        ClassDef(
            "ThermalManagementSubsystem", "Thermal Management Subsystem", "Subsystem",
            "Thermal control: passive/active management of system temperature.",
            _ALL,
        ),
        ClassDef(
            "StructuralSubsystem", "Structural Subsystem", "Subsystem",
            "Primary and secondary structure, mechanisms, and separation systems.",
            _ALL,
        ),
        ClassDef(
            "PayloadSubsystem", "Payload Subsystem", "Subsystem",
            "Mission-specific instruments, cameras, or passenger systems.",
            _ALL,
        ),
        ClassDef(
            "CommandDataHandlingSubsystem", "Command and Data Handling Subsystem", "Subsystem",
            "On-board computing, data storage, and command execution.",
            _CULU,
        ),
        ClassDef(
            "GuidanceNavigationControlSubsystem",
            "Guidance Navigation and Control Subsystem", "Subsystem",
            "GNC: trajectory computation, navigation, and vehicle control.",
            _GNCD,
        ),

        # ── Power components (under PowerSubsystem) ───────────────────────────
        ClassDef(
            "SolarPanel", "Solar Panel", "PowerSubsystem",
            "Photovoltaic panel for solar energy collection.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "BatteryPack", "Battery Pack", "PowerSubsystem",
            "Rechargeable electrochemical energy storage unit.",
            _ALL, is_component=True,
        ),
        ClassDef(
            "PowerRegulator", "Power Regulator / EPS", "PowerSubsystem",
            "Voltage/current regulation and power distribution unit.",
            _ALL, is_component=True,
        ),
        ClassDef(
            "FuelCell", "Fuel Cell", "PowerSubsystem",
            "Electrochemical power source for extended-duration missions.",
            ["uam", "lunar"], is_component=True,
        ),

        # ── Chemical propulsion components ────────────────────────────────────
        ClassDef(
            "MonopropellantThruster", "Monopropellant Thruster", "ChemicalPropulsionSystem",
            "Single-propellant thruster, typically hydrazine.",
            _ROLU, is_component=True,
        ),
        ClassDef(
            "BipropellantThruster", "Bipropellant Thruster", "ChemicalPropulsionSystem",
            "Dual-propellant thruster (e.g., LOX/LH2, NTO/MMH).",
            _RO, is_component=True,
        ),

        # ── Electric propulsion components ────────────────────────────────────
        ClassDef(
            "IonEngine", "Ion Engine", "ElectricPropulsionSystem",
            "Gridded ion propulsion engine for efficient low-thrust manoeuvres.",
            _CU, is_component=True,
        ),
        ClassDef(
            "HallThruster", "Hall-Effect Thruster", "ElectricPropulsionSystem",
            "Hall-effect thruster for medium-thrust electric propulsion.",
            _CU, is_component=True,
        ),
        ClassDef(
            "ColdGasThruster", "Cold Gas Thruster", "ElectricPropulsionSystem",
            "Simple attitude control thruster using compressed inert gas.",
            _CU, is_component=True,
        ),
        ClassDef(
            "ElectricMotor", "Electric Motor", "ElectricPropulsionSystem",
            "Electric rotary motor for propeller or rotor drive.",
            _UA, is_component=True,
        ),

        # ── Communication components (under CommunicationSubsystem) ───────────
        ClassDef(
            "PatchAntenna", "Patch Antenna", "CommunicationSubsystem",
            "Low-profile microstrip antenna for compact platforms.",
            _CU, is_component=True,
        ),
        ClassDef(
            "HelicalAntenna", "Helical Antenna", "CommunicationSubsystem",
            "Helical antenna for circularly polarized UHF/VHF transmission.",
            _CU, is_component=True,
        ),
        ClassDef(
            "ParabolicDishAntenna", "Parabolic Dish Antenna", "CommunicationSubsystem",
            "High-gain dish antenna for deep-space or long-range communication.",
            _ROLU, is_component=True,
        ),
        ClassDef(
            "UHFTransceiver", "UHF Transceiver", "CommunicationSubsystem",
            "Ultra High Frequency transceiver (300 MHz – 3 GHz).",
            _CU, is_component=True,
        ),
        ClassDef(
            "SBandTransceiver", "S-Band Transceiver", "CommunicationSubsystem",
            "S-Band transceiver (2–4 GHz) for spacecraft telemetry and command.",
            _ALL, is_component=True,
        ),
        ClassDef(
            "XBandTransceiver", "X-Band Transceiver", "CommunicationSubsystem",
            "X-Band transceiver (8–12 GHz) for high-rate science data downlink.",
            _ROLU, is_component=True,
        ),

        # ── Attitude control components ────────────────────────────────────────
        ClassDef(
            "ReactionWheel", "Reaction Wheel", "AttitudeControlSubsystem",
            "Momentum storage wheel for fine attitude control.",
            _CU, is_component=True,
        ),
        ClassDef(
            "MagnetorquerBar", "Magnetorquer Bar", "AttitudeControlSubsystem",
            "Electromagnetic coil for magnetic torque generation.",
            _CU, is_component=True,
        ),
        ClassDef(
            "StarTracker", "Star Tracker", "AttitudeControlSubsystem",
            "Optical sensor for precise attitude determination via star patterns.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "SunSensor", "Sun Sensor", "AttitudeControlSubsystem",
            "Optical sensor for coarse Sun direction measurement.",
            _CU, is_component=True,
        ),
        ClassDef(
            "IMU", "Inertial Measurement Unit (IMU)", "AttitudeControlSubsystem",
            "Accelerometer and gyroscope package for inertial navigation.",
            _ALL, is_component=True,
        ),
        ClassDef(
            "ControlMomentGyroscope", "Control Moment Gyroscope (CMG)", "AttitudeControlSubsystem",
            "High-torque attitude actuator using spinning flywheels.",
            _LU, is_component=True,
        ),

        # ── Thermal components ────────────────────────────────────────────────
        ClassDef(
            "HeatPipe", "Heat Pipe", "ThermalManagementSubsystem",
            "Passive heat transfer device using working fluid phase changes.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "RadiatorPanel", "Radiator Panel", "ThermalManagementSubsystem",
            "Thermal radiator for rejecting waste heat to space.",
            ["cubesat", "lunar", "rocket"], is_component=True,
        ),
        ClassDef(
            "ThermalInsulationBlanket", "Multi-Layer Insulation (MLI) Blanket",
            "ThermalManagementSubsystem",
            "Multi-layer insulation for passive thermal control.",
            ["cubesat", "lunar", "rocket"], is_component=True,
        ),
        ClassDef(
            "ElectricHeater", "Electric Heater", "ThermalManagementSubsystem",
            "Resistive heater for maintaining minimum operating temperatures.",
            _ALL, is_component=True,
        ),

        # ── Structural components ─────────────────────────────────────────────
        ClassDef(
            "PrimaryStructureFrame", "Primary Structure Frame", "StructuralSubsystem",
            "Main load-bearing structural frame of the system.",
            _ALL, is_component=True,
        ),
        ClassDef(
            "DeployableSolarArray", "Deployable Solar Array", "StructuralSubsystem",
            "Hinged or roll-out solar panel deployment mechanism.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "SeparationMechanism", "Separation Mechanism", "StructuralSubsystem",
            "Pyrotechnic or mechanical device for stage or payload separation.",
            _RO, is_component=True,
        ),

        # ── Payload components ────────────────────────────────────────────────
        ClassDef(
            "OpticalImager", "Optical Imager", "PayloadSubsystem",
            "Camera or telescope for Earth observation or planetary imaging.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "SARPayload", "Synthetic Aperture Radar (SAR) Payload", "PayloadSubsystem",
            "SAR instrument for all-weather surface imaging.",
            _CU, is_component=True,
        ),
        ClassDef(
            "SpectralInstrument", "Spectral Instrument", "PayloadSubsystem",
            "Multi- or hyperspectral sensor for scientific observation.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "PassengerCabin", "Passenger Cabin", "PayloadSubsystem",
            "Pressurised compartment for passenger transport.",
            _UA, is_component=True,
        ),
        ClassDef(
            "CargoModule", "Cargo Module", "PayloadSubsystem",
            "Structural enclosure for cargo or payload delivery.",
            _RUA, is_component=True,
        ),

        # ── Computing components (under CommandDataHandlingSubsystem) ──────────
        ClassDef(
            "OnboardComputer", "On-Board Computer (OBC)", "CommandDataHandlingSubsystem",
            "Main processor for command execution and data handling.",
            _CULU, is_component=True,
        ),
        ClassDef(
            "FlightComputer", "Flight Computer", "CommandDataHandlingSubsystem",
            "Radiation-hardened avionics computer for flight control.",
            _RUA, is_component=True,
        ),
        ClassDef(
            "DataStorageUnit", "Data Storage Unit", "CommandDataHandlingSubsystem",
            "Mass memory or solid-state recorder for mission data.",
            _ALL, is_component=True,
        ),

        # ── GNC components (under GuidanceNavigationControlSubsystem) ─────────
        ClassDef(
            "GPSNavigationUnit", "GPS Navigation Unit", "GuidanceNavigationControlSubsystem",
            "GPS/GNSS receiver and navigation processor for position and velocity.",
            _GNCD, is_component=True,
        ),
        ClassDef(
            "NavigationFilterUnit", "Navigation Filter Unit", "GuidanceNavigationControlSubsystem",
            "Kalman filter computer for state estimation and trajectory navigation.",
            _GNCD, is_component=True,
        ),
    ],

    # ── Data Properties ───────────────────────────────────────────────────────
    data_properties=[
        DataPropDef("massKg",               "Mass",                          "decimal", "kg",    "Total mass of the component, subsystem, or system."),
        DataPropDef("powerW",               "Power Consumption",             "decimal", "W",     "Electrical power consumed during nominal operation."),
        DataPropDef("powerGenerationW",     "Power Generation",              "decimal", "W",     "Electrical power produced (solar panels, fuel cells)."),
        DataPropDef("volumeL",              "Volume",                        "decimal", "L",     "Physical volume occupied."),
        DataPropDef("operatingTempMinC",    "Minimum Operating Temperature", "decimal", "deg C", "Minimum temperature within operating specification."),
        DataPropDef("operatingTempMaxC",    "Maximum Operating Temperature", "decimal", "deg C", "Maximum temperature within operating specification."),
        DataPropDef("dataRateMbps",         "Data Rate",                     "decimal", "Mbps",  "Maximum communication or data transfer rate."),
        DataPropDef("frequencyGHz",         "RF Frequency",                  "decimal", "GHz",   "Operating radio frequency."),
        DataPropDef("thrustN",              "Thrust",                        "decimal", "N",     "Propulsive thrust force produced."),
        DataPropDef("specificImpulseS",     "Specific Impulse (Isp)",        "decimal", "s",     "Propellant efficiency metric."),
        DataPropDef("torqueNm",             "Torque",                        "decimal", "Nm",    "Angular torque produced (reaction wheels, CMGs)."),
        DataPropDef("storageGB",            "Storage Capacity",              "decimal", "GB",    "On-board data storage capacity."),
        DataPropDef("processingSpeedMHz",   "Processing Speed",              "decimal", "MHz",   "Processor clock speed."),
        DataPropDef("liftCapacityKg",       "Lift Capacity",                 "decimal", "kg",    "Maximum payload or passenger mass that can be lifted."),
        DataPropDef("rangeKm",              "Range",                         "decimal", "km",    "Maximum operational range."),
        DataPropDef("altitudeKm",           "Altitude",                      "decimal", "km",    "Operating altitude (orbit altitude or cruise altitude)."),
        DataPropDef("orbitalInclinationDeg","Orbital Inclination",           "decimal", "deg",   "Angle of orbit plane relative to the equator."),
        DataPropDef("lifespanYears",        "Design Lifespan",               "decimal", "yr",    "Expected operational design lifetime."),
        DataPropDef("costUSD",              "Estimated Cost",                "decimal", "USD",   "Estimated unit cost in US dollars."),
        DataPropDef("orbitType",            "Orbit Type",                    "string",  None,    "Orbit classification: LEO, MEO, GEO, Lunar, Suborbital."),
        DataPropDef("propellantType",       "Propellant Type",               "string",  None,    "Propellant used: hydrazine, xenon, LH2/LOX, etc."),
        DataPropDef("missionType",          "Mission Type",                  "string",  None,    "Mission objective: Earth-observation, communication, etc."),
        DataPropDef("manufacturer",         "Manufacturer",                  "string",  None,    "Name of the manufacturer or prime contractor."),
        DataPropDef("componentIdentifier",  "Component Identifier",          "string",  None,    "Model number or part number for identification."),
    ],

    # ── Object Properties ─────────────────────────────────────────────────────
    object_properties=[
        ObjectPropDef(
            "hasSubsystem", "Has Subsystem", "AerospaceSystem", "Subsystem",
            "Associates a system with one of its subsystems.",
        ),
        ObjectPropDef(
            "hasComponent", "Has Component", "Subsystem", "Subsystem",
            "Associates a subsystem with one of its component individuals "
            "(component classes are subclasses of their parent subsystem).",
        ),
        ObjectPropDef(
            "isPartOf", "Is Part Of", "Subsystem", "AerospaceSystem",
            "Inverse of hasSubsystem / hasComponent.",
        ),
        ObjectPropDef(
            "poweredBy", "Powered By", "Subsystem", "PowerSubsystem",
            "Indicates which PowerSubsystem supplies power to a subsystem.",
        ),
    ],

    # ── Domain guidance ───────────────────────────────────────────────────────
    domain_required_subsystems={
        "cubesat": [
            "PowerSubsystem",
            "CommunicationSubsystem",
            "AttitudeControlSubsystem",
            "CommandDataHandlingSubsystem",
            "StructuralSubsystem",
            "PayloadSubsystem",
        ],
        "uam": [
            "PowerSubsystem",
            "ElectricPropulsionSystem",
            "CommunicationSubsystem",
            "GuidanceNavigationControlSubsystem",
            "StructuralSubsystem",
            "PayloadSubsystem",
        ],
        "rocket": [
            "ChemicalPropulsionSystem",
            "GuidanceNavigationControlSubsystem",
            "StructuralSubsystem",
            "ThermalManagementSubsystem",
            "CommunicationSubsystem",
        ],
        "lunar": [
            "PowerSubsystem",
            "ChemicalPropulsionSystem",
            "CommunicationSubsystem",
            "AttitudeControlSubsystem",
            "ThermalManagementSubsystem",
            "PayloadSubsystem",
        ],
    },

    domain_system_classes={
        "cubesat": ["NanoSatellite", "MicroSatellite"],
        "uam":     ["eVTOL"],
        "rocket":  ["SmallLaunchVehicle", "HeavyLiftVehicle"],
        "lunar":   ["LunarLander", "LunarOrbiter"],
    },
)
