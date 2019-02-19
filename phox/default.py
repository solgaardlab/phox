from .mesh import SimMZI

# SIMULATION MZI DEFAULTS
simulation_mzi = SimMZI(
    bend_dim=(10, 5),
    end_bend_dim=(5, 0),
    waveguide_width=2,
    waveguide_thickness=0.35,
    phase_shifter_arm_length=20,
    coupler_end_length=10,
    coupling_spacing=0.5,
    interaction_length=0.01
)

demo_mzi = SimMZI(
    bend_dim=(10, 5),
    end_bend_dim=(5, 0),
    waveguide_width=2,
    waveguide_thickness=0.35,
    phase_shifter_arm_length=20,
    coupler_end_length=20,
    coupling_spacing=0.5,
    interaction_length=0.01
)

semi_compact_demo_mzi = SimMZI(
    bend_dim=(10, 5),
    end_bend_dim=(0.01, 0),
    waveguide_width=2,
    waveguide_thickness=0.35,
    phase_shifter_arm_length=20,
    coupler_end_length=10,
    coupling_spacing=0.5,
    interaction_length=0.01
)

compact_demo_mzi = SimMZI(
    bend_dim=(10, 5),
    end_bend_dim=(0.01, 0),
    waveguide_width=2,
    waveguide_thickness=0.35,
    phase_shifter_arm_length=20,
    coupler_end_length=0,
    coupling_spacing=0.5,
    interaction_length=0.01
)

simulation_single_mzi = SimMZI(
    bend_dim=(10, 10),
    end_bend_dim=(5, 0),
    waveguide_width=3,
    waveguide_thickness=0.25,
    phase_shifter_arm_length=20,
    coupler_end_length=20,
    coupling_spacing=1.5,
    interaction_length=0.01
)
