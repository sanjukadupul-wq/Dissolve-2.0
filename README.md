# Dissolve 2.0 (GPU)

> **DISSOLVE** = **D**egradation **I**n-**S**ilico **SOLVE**r

Dissolve 2.0 is a GPU-accelerated computational framework for predicting the degradation of biodegradable zinc-based implants and porous scaffolds. The software couples multi-species reaction-diffusion transport, oxygen reduction reaction (ORR) kinetics, passive film evolution, and level-set interface tracking within a unified finite-element framework to simulate spatially resolved corrosion and material loss in complex three-dimensional geometries.
Built on DOLFINx 0.10.0, PETSc, CUDA, and libCEED, Dissolve 2.0 represents the next-generation implementation of the original Dissolve solver. The code preserves the validated physics and governing equations of Dissolve 1.0 while introducing modern GPU-enabled numerical infrastructure for significantly improved computational performance, scalability, and flexibility. GPU acceleration is applied to the most computationally intensive components of the reaction-diffusion system, enabling large-scale simulations of patient-specific implants and highly porous lattice structures that would otherwise require substantial CPU resources.
The framework is designed for the study and optimisation of biodegradable Zn-based medical devices, including orthopaedic scaffolds, TPMS structures, lattice implants, and vascular stents. By predicting degradation morphology, mass loss, species distributions, and interface evolution, Dissolve 2.0 provides a platform for computational implant design, virtual prototyping, and degradation-performance assessment.
Dissolve 2.0 was developed by Henaka Ariyarathna (2026) as the successor to the original FreeFEM-based Dissolve framework. The software maintains compatibility with the validated corrosion model developed in Dissolve 1.0 while extending the codebase through GPU acceleration, modular software architecture, advanced meshing workflows, and improved support for high-performance computing environments.
For installation instructions and dependency setup, see **[INSTALL.md](INSTALL.md)**.

## GPU Acceleration Strategy

Dissolve 2.0 adopts a hybrid CPU-GPU computational strategy designed to maximise acceleration while maintaining numerical robustness for strongly coupled corrosion simulations. Rather than forcing all equations onto the GPU, each component of the degradation model is solved on the hardware architecture most suitable for its numerical characteristics.
The reaction-diffusion transport equations governing dissolved oxygen (O₂), chloride ions (Cl⁻), and hydroxide ions (OH⁻) are executed on the GPU when CUDA-enabled PETSc support is available. Matrix assembly, vector operations, and iterative linear solves are performed using PETSc CUDA backends, enabling substantial reductions in computational time for large three-dimensional meshes.
The zinc ion transport and passive film evolution equations remain on the CPU. These systems exhibit significantly stronger conditioning challenges during degradation simulations and benefit from mature CPU-based algebraic multigrid preconditioners, particularly Hypre BoomerAMG, which currently provides greater robustness for these equations than available GPU alternatives.
To further increase device utilisation, Dissolve 2.0 incorporates optional GPU-native acceleration technologies:
- **PETSc CUDA** for sparse matrix assembly and linear algebra operations.
- **libCEED** for matrix-free finite-element operator evaluation.
- **Custom CUDA element assembly kernels** for reducing assembly overhead in large tetrahedral meshes.
- **GPU isosurface extraction** for accelerated degradation morphology visualisation and interface tracking.
This hybrid design enables Dissolve 2.0 to preserve the validated numerical behaviour of Dissolve 1.0 while substantially improving computational performance on modern high-performance computing platforms. The approach is particularly advantageous for simulations involving large biodegradable implants, fine-resolution degradation fronts, and highly porous TPMS or lattice structures where computational cost becomes a limiting factor.

## Repository Structure

The Dissolve 2.0 codebase has been reorganised into a modular architecture to improve maintainability, extensibility, and long-term development. The structure separates domain handling, physical models, numerical solvers, input/output utilities, and GPU-specific functionality into dedicated modules while preserving the validated corrosion physics implemented in Dissolve 1.0.

```text
Dissolve-2.0/
├── dissolve.py
├── config/
├── domain/
├── physics/
├── numerics/
├── gpu/
├── io/
├── analysis/
├── meshing/
└── mesh_adaptive_800k.xdmf
```

### Core Modules

#### `dissolve.py`
Main program entry point that orchestrates model initialisation, time integration, transport solves, level-set evolution, and result generation.

#### `config/`
Contains simulation settings, command-line parameters, and version information. All model constants, numerical controls, and run-time options are managed from this module.

#### `domain/`
Mesh-processing and geometric utilities, including mesh loading, signed-distance field initialisation, and level-set reinitialisation procedures.

#### `physics/`
Implementation of the governing equations describing reaction-diffusion transport, passive film evolution, oxygen reduction reaction (ORR) kinetics, and interface recession during degradation.

#### `numerics/`
Numerical solution algorithms, PETSc solver configuration, time-stepping procedures, and CPU-GPU execution management.

#### `gpu/`
GPU-specific acceleration components developed for Dissolve 2.0, including CUDA-enabled assembly routines, libCEED matrix-free operators, and GPU-based geometry processing workflows.

#### `io/`
Input/output routines for simulation results, visualization files, diagnostic logs, summary statistics, and degradation metrics.

#### `analysis/`
Post-processing and structural assessment tools. Current utilities include finite-element mechanical analysis for evaluating degradation-induced changes in scaffold stiffness and load-bearing performance.

#### `meshing/`
Mesh-generation scripts for implant, scaffold, and stent geometries. These tools provide automated generation of computational meshes suitable for degradation simulations.

### Design Philosophy
The modular design of Dissolve 2.0 enables individual components to be modified or extended without altering the core solver framework. This architecture facilitates future development of additional degradation mechanisms, advanced constitutive models, machine-learning-assisted workflows, alternative electrochemical reactions, and next-generation GPU acceleration strategies while maintaining compatibility with the validated Dissolve corrosion model.

## A Note on Mesh Data

To keep repository size manageable and simplify source-code distribution, large mesh binaries are not included in Dissolve 2.0. While the repository provides the mesh-generation workflows and mesh metadata required to reproduce simulation domains, the corresponding high-resolution mesh files are excluded because they can be tens to hundreds of megabytes in size depending on geometry complexity and mesh density.
The included `mesh_adaptive_800k.xdmf` file serves as a lightweight metadata wrapper that references the underlying mesh data stored in an associated HDF5 file. Users must either:
- Generate new meshes using the scripts provided in the `meshing/` directory.
- Supply their own computational meshes.
- Restore the required HDF5 mesh files if obtained separately from the repository authors.
Mesh generation utilities are provided for representative biodegradable implant geometries, including cylindrical scaffolds and vascular stents. These scripts use Gmsh-based adaptive meshing strategies to create high-quality tetrahedral meshes suitable for reaction-diffusion and level-set simulations.

Example:

```bash
python meshing/mesh_generator_disc_adaptive.py
python meshing/mesh_generator_stent_adaptive.py
```

## Running Dissolve 2.0

Dissolve 2.0 performs fully coupled degradation simulations using reaction-diffusion transport, ORR kinetics, passive film evolution, and level-set interface tracking. Simulations can be executed on a single CPU core, distributed across multiple MPI processes, or accelerated using NVIDIA GPUs through PETSc CUDA and libCEED.
Before running the solver, ensure that all required software dependencies have been installed and configured correctly. Detailed installation instructions are provided in **[INSTALL.md](INSTALL.md)**.

### Basic Simulation

Run a standard degradation simulation using a supplied or user-generated mesh:
```bash
python3 dissolve.py --input_mesh mesh_adaptive_800k.xdmf
```

### Specify Simulation Duration

The total simulation time can be adjusted using the `--sim_duration` argument:
```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --sim_duration 24.0
```

### CPU Execution

To disable GPU acceleration and run entirely on the CPU:
```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --use_gpu 0
```

### Multi-Core and HPC Execution

Dissolve 2.0 supports distributed-memory parallelism through MPI.
Example using four processes:
```bash
mpirun -np 4 python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf
```
For large implant geometries and high-resolution meshes, execution on HPC clusters is strongly recommended.

### GPU-Accelerated Execution

When PETSc is compiled with CUDA support and a compatible NVIDIA GPU is available, GPU acceleration can be enabled automatically:
```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --use_gpu 1
```
The solver will utilize GPU-enabled PETSc backends where appropriate while maintaining numerical robustness for equations that benefit from CPU-based solvers.

### Example: TPMS Scaffold Simulation

```bash
python3 dissolve.py \
    --input_mesh gyroid_scaffold.xdmf \
    --sim_duration 672 \
    --dt_hours 1.0 \
    --use_gpu 1
```

### Simulation Outputs

During execution, Dissolve 2.0 generates a range of outputs for post-processing and analysis, including:
- Time-resolved mass loss data.
- Species concentration fields.
- Passive film distributions.
- Level-set interface evolution.
- Surface recession metrics.
- Diagnostic solver information.
- VTK/XDMF visualization files.
- Simulation summary reports.
Results can be visualized directly in ParaView for analysis of degradation morphology, concentration gradients, and evolving implant geometry.

### Typical Workflow

1. Generate or import a computational mesh.
2. Configure simulation parameters.
3. Launch the degradation simulation.
4. Monitor solver progress and diagnostics.
5. Visualize results in ParaView.
6. Analyse degradation behaviour, mass loss, and geometric evolution.

This workflow enables rapid evaluation of biodegradable Zn-based implants, porous scaffolds, lattice structures, and stent geometries under physiologically relevant degradation conditions.

## Command-Line Reference

All simulation parameters in Dissolve 2.0 can be modified directly from the command line. Parameters control numerical settings, degradation kinetics, mesh handling, GPU acceleration, output generation, and post-processing workflows.
A complete list of available arguments can be displayed using:
```bash
python3 dissolve.py --help
```

### GPU Acceleration

These options control CUDA-enabled execution and GPU-specific functionality.
```text
--use_gpu
    Enable GPU acceleration through PETSc CUDA and libCEED
    Default: 1

--velocity_use_gpu_grad
    Use GPU-based gradient evaluation during interface velocity calculations
    Default: 0
```

### Degradation Kinetics and Transport

These parameters define the electrochemical degradation model, species transport rates, and passive film behaviour.

```text
--kf
    Passive film formation rate (1/hour)
    Default: 125
--kd
    Passive film dissolution rate (mm⁶/(g²·hour))
    Default: 40
--k_orr
    Effective oxygen reduction reaction rate used by the surrogate model (mm/hour)
    Default: 0.015
--k_orr_react
    Physical ORR reaction rate used when
    --velocity_mode physical
    Default: 0.90
--velocity_mode
    Interface recession model:
      surrogate
      physical
    Default: surrogate
--use_orr
    Enable oxygen reduction reaction kinetics
    Default: 1
--diff_zn
    Zinc ion diffusivity (mm²/hour)
    Default: 2.72
--diff_cl
    Chloride ion diffusivity (mm²/hour)
    Default: 7.78
--diff_oh
    Hydroxide ion diffusivity (mm²/hour)
    Default: 20.16
```

### Passive Film Transport Properties

These parameters control mass transport through the degradation product layer.

```text
--film_tortuosity
    Film tortuosity factor
    Default: 2.0
--film_porosity
    Film porosity
    Default: 0.25
--film_constrictivity
    Film constrictivity factor
    Default: 1.0
```

### Domain and Mesh Control

Options associated with geometry initialisation and mesh handling.

```text
--input_mesh
    Input XDMF mesh file
    Default: mesh/sphere.mesh
--analytic_sdf
    Initialise the level-set field using an analytical signed-distance function.
    Available options:
      disc
      stent
--adapt_initial_mesh
    Perform mesh adaptation before simulation begins
    Default: 0
--adapt_mesh_runtime
    Enable mesh adaptation during simulation
    Default: 0
```

### Time Integration

Simulation duration and numerical time-stepping controls.

```text
--dt_hours
    Timestep size (hours)
    Default: 1.0
--sim_duration
    Total simulation duration (hours)
    Default: 672.0
--adaptive_dt
    Enable adaptive timestep control
    Default: 0
--dt_min
    Minimum allowable timestep
    Default: 0.1
--time_step_max
    Maximum allowable timestep
    Default: 4.0
--enable_redistance
    Enable periodic level-set reinitialisation
    Default: 1
```

### Output and Post-Processing

Settings governing simulation output and data generation.

```text
--results_file
    Output file for mass-loss history
    Default: output/result.txt
--write_vtu
    Write VTU/PVD visualisation files
    Default: 1
--emit_vtk
    Alias for --write_vtu
    Default: 1
--vis_each_steps
    Visualisation output frequency
    Default: 1
--write_diagnostics
    Write solver diagnostics and performance statistics
    Default: 1
--mechanics
    Run structural finite-element analysis on degradation snapshots
    Default: 0
```

### Example Usage

Run a one-week GPU-accelerated degradation simulation:
```bash
python3 dissolve.py \
    --input_mesh scaffold.xdmf \
    --sim_duration 168 \
    --use_gpu 1
```

Run a 28-day simulation with adaptive timestepping:
```bash
python3 dissolve.py \
    --input_mesh scaffold.xdmf \
    --sim_duration 672 \
    --adaptive_dt 1
```

Run a coupled degradation-mechanics study:
```bash
python3 dissolve.py \
    --input_mesh scaffold.xdmf \
    --mechanics 1 \
    --use_gpu 1
```

## Repository Contents

Dissolve 2.0 is distributed as a source-code repository intended to provide the complete simulation framework, numerical implementation, and mesh-generation workflows required to reproduce and extend the degradation model. To maintain a manageable repository size and simplify version control, large binary datasets and simulation outputs are not included.
The following items are intentionally excluded:
- High-resolution mesh binaries (`.h5`, `.mesh`, and related large mesh files).
- Generated simulation results and visualization outputs.
- Temporary solver files and checkpoint data.
- Post-processing images and videos generated during simulations.
- Large benchmark datasets used for performance testing.
Users can regenerate computational meshes using the scripts provided in the `meshing/` directory or supply their own meshes in a compatible DOLFINx XDMF/HDF5 format. Simulation outputs will be created automatically during execution and stored in the designated output directories.
The repository therefore contains all source code required to build, run, modify, and extend the Dissolve 2.0 framework while avoiding the unnecessary storage of large generated files that can be reproduced from the provided workflows.

## License

Dissolve 2.0 is released under the **GNU General Public License v3.0 (GPL-3.0)**.
This repository is a derivative and GPU-enabled successor of the original **Dissolve 1.0** framework, which was developed using FreeFEM for simulation of biodegradable Zn-alloy degradation. Dissolve 2.0 preserves the validated physical model, governing equations, degradation mechanisms, and core numerical methodology of the original framework while extending the implementation through a modern DOLFINx-based architecture with optional GPU acceleration via PETSc CUDA and libCEED.
Under the terms of the GPL-3.0 license, users are free to:
- Use the software for academic, research, and educational purposes.
- Modify and extend the source code.
- Distribute original or modified versions of the software.
- Incorporate improvements into future derivative works.
Any redistribution of the software, including modified versions, must comply with the requirements of the GPL-3.0 license and retain appropriate copyright and license notices.
See the full license text in the **LICENSE** file.

### Attribution

If you use Dissolve 2.0 in academic research, publications, or derivative software projects, please acknowledge the original Dissolve framework and cite the relevant accompanying publications when available.

**Dissolve 1.0**
- Henaka Ariyarathna (2026)
- FreeFEM-based reaction-diffusion and level-set framework for biodegradable Zn-alloy degradation.

**Dissolve 2.0**
- Henaka Ariyarathna (2026)
- DOLFINx/PETSc/libCEED GPU implementation and software architecture redesign.

The authors welcome citations, feedback, and contributions that support further development of computational tools for biodegradable implant modelling and optimisation.
