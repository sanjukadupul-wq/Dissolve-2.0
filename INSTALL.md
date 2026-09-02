# Installing Dissolve 2.0 (GPU)

## 1. Introduction

Dissolve 2.0 is a GPU-accelerated computational framework for predicting the degradation of biodegradable zinc-based implants and porous scaffolds. The software combines multi-species reaction-diffusion transport, oxygen reduction reaction (ORR) kinetics, passive film evolution, and level-set interface tracking within a finite-element framework to simulate spatially resolved corrosion and material loss in complex three-dimensional geometries.
The code represents the next-generation implementation of the original Dissolve framework, replacing the FreeFEM-based solver architecture with a modern DOLFINx, PETSc, and CUDA-enabled workflow. While preserving the validated physical model and governing equations of Dissolve 1.0, Dissolve 2.0 introduces a modular software design, GPU acceleration, improved scalability, and enhanced support for high-performance computing environments.
Dissolve 2.0 is intended for researchers, engineers, and developers working in biodegradable metallic biomaterials, computational corrosion modelling, implant design, and virtual prototyping. The framework enables investigation of degradation morphology, mass loss, species transport, passive film behaviour, and implant integrity in complex lattice, TPMS, scaffold, and stent geometries.

### Supported Platforms

Dissolve 2.0 is supported on:
- Ubuntu Linux (recommended)
- Windows via Windows Subsystem for Linux 2 (WSL2)
- HPC clusters running Linux
- NVIDIA CUDA-enabled systems for GPU acceleration
Although execution on CPU-only systems is supported, GPU-equipped systems are recommended for large-scale simulations involving high-resolution meshes and extended degradation periods.

### Software Stack

Dissolve 2.0 is built upon the following open-source scientific computing technologies:
- DOLFINx 0.10.0
- PETSc
- MPI (OpenMPI)
- NVIDIA CUDA
- libCEED
- Python 3
- Gmsh
- ParaView
Together, these tools provide finite-element discretisation, parallel computing, GPU acceleration, mesh generation, and scientific visualisation capabilities required for large-scale degradation simulations.

### Recommended Installation Path

For most users, the recommended installation workflow is:
1. Install WSL2 and Ubuntu 24.04 (Windows users).
2. Install NVIDIA drivers and CUDA support.
3. Create a dedicated Python environment.
4. Install PETSc with CUDA support.
5. Install DOLFINx and associated Python packages.
6. Install libCEED for advanced GPU acceleration.
7. Clone the Dissolve 2.0 repository.
8. Run the verification tests and example simulations.
Following the recommended workflow provides the most stable and reproducible environment for both workstation and high-performance computing deployments.

> **Already set up Dissolve 1.0?** If you followed the [Dissolve 1.0 WSL2 setup guide](https://github.com/sanjukadupul-wq/Dissolve-1.0/blob/main/Src%20Codes/INSTALL.md) and installed Ubuntu 22.04, that same WSL2 environment can be reused here — Ubuntu 22.04 supports the full Dissolve 2.0 stack. Ubuntu 24.04 is the version tested during Dissolve 2.0 development and is recommended for new installations, but is not a strict requirement.

## 2. System Requirements

This section outlines the hardware and software requirements for running Dissolve 2.0. The minimum specifications are suitable for small demonstration cases and method development, while the recommended specifications are intended for production-scale degradation simulations involving complex implant geometries and high-resolution meshes.

### Minimum Requirements

The following configuration is sufficient for code development, testing, and small-scale simulations:

#### Hardware

- 4-core CPU
- 16 GB RAM
- 20 GB available storage
- NVIDIA GPU (optional)

#### Software

- Ubuntu 22.04+ or WSL2
- Python 3.10 or newer
- OpenMPI
- PETSc
- DOLFINx 0.10.0
- Gmsh
- ParaView (for visualization)

#### Suitable Use Cases

- Validation examples
- Small mesh simulations (<100k elements)
- CPU-only execution
- Software development and debugging

---

### Recommended Requirements

The following configuration is recommended for routine research use and large three-dimensional degradation simulations.

#### Hardware

- 8-16 CPU cores
- 32-64 GB RAM
- NVIDIA RTX-series GPU with CUDA support
- 12 GB or greater GPU memory
- SSD storage with at least 100 GB free space

#### Software

- Ubuntu 24.04 LTS
- Python 3.11+
- OpenMPI 4.x or newer
- PETSc with CUDA support
- DOLFINx 0.10.0
- libCEED
- CuPy
- Gmsh 4.13.1
- ParaView

#### Suitable Use Cases

- TPMS scaffold degradation simulations
- Lattice implant analysis
- Stent degradation studies
- GPU-accelerated simulations
- High-resolution meshes (500k-5M elements)
- Parametric and sensitivity studies
- HPC deployments

---

### Tested Configurations

Dissolve 2.0 has been developed and tested using the following software environment:

#### Development Workstation

- Windows 11
- WSL2 (Ubuntu 24.04 LTS)
- Python 3.12
- DOLFINx 0.10.0
- PETSc 3.24+
- OpenMPI
- CUDA 12.x
- libCEED
- CuPy
- Gmsh 4.13.1
- ParaView

#### GPU Hardware

- NVIDIA RTX-series GPUs
- CUDA-enabled NVIDIA drivers
- CUDA Toolkit 12.x

#### HPC Environment

- Linux-based clusters
- Slurm workload manager
- OpenMPI
- PETSc with MPI support
- NVIDIA GPU nodes (optional)
While other configurations may function correctly, users are strongly encouraged to follow the tested software stack described in this guide to ensure compatibility and reproducibility of simulation results.

## 3. Supported Operating Systems

Dissolve 2.0 is designed for Linux-based scientific computing environments and supports workstations, high-performance computing (HPC) systems, and GPU-enabled platforms. Although the software can be executed on a variety of Linux distributions, users are strongly encouraged to follow the recommended environments described below to ensure compatibility with DOLFINx, PETSc, CUDA, and MPI.

### Ubuntu Linux

Ubuntu Linux is the primary development and testing platform for Dissolve 2.0 and is recommended for all standalone workstation installations.

#### Recommended Versions

- Ubuntu 24.04 LTS (recommended)
- Ubuntu 22.04 LTS

#### Advantages

- Native support for DOLFINx and PETSc
- Simplified MPI configuration
- Direct CUDA integration
- Easier dependency management
- Full compatibility with Gmsh and ParaView

#### Recommended Use Cases

- Research workstations
- GPU-enabled desktop systems
- Development and debugging
- Medium to large-scale degradation simulations

Users running Ubuntu on dedicated Linux hardware will generally obtain the best performance and the simplest installation experience.

---

### WSL2 (Recommended for Windows Users)

Windows users are strongly encouraged to use Windows Subsystem for Linux 2 (WSL2) rather than attempting a native Windows installation.

WSL2 provides a full Linux environment integrated within Windows while maintaining compatibility with scientific computing tools commonly used by the DOLFINx and PETSc ecosystems.

#### Advantages

- Access to a complete Ubuntu environment
- Improved OpenMPI compatibility
- Simplified PETSc compilation
- Native support for DOLFINx
- CUDA acceleration through NVIDIA WSL drivers
- Easier package management and reproducibility

#### Recommended Configuration

- Windows 11
- WSL2
- Ubuntu 24.04 LTS
- NVIDIA GPU (optional but recommended)
- CUDA Toolkit 12.x

#### Recommended Use Cases

- Windows-based research workstations
- Development and testing
- GPU-accelerated simulations
- Reproducible computational research

Because many scientific software packages are developed primarily for Linux environments, WSL2 offers a more stable and maintainable long-term platform than native Windows installations.

---

### HPC Clusters

Dissolve 2.0 is designed to support execution on modern high-performance computing (HPC) systems.

The software can leverage:

- MPI-based distributed-memory parallelism
- Multi-core CPU architectures
- GPU-enabled compute nodes
- Job scheduling systems such as Slurm
- Shared research computing facilities

#### Supported HPC Environments

- Linux-based clusters
- Slurm-managed systems
- OpenMPI deployments
- CUDA-enabled GPU partitions
- University and national supercomputing facilities

#### Recommended Use Cases

- High-resolution implant simulations
- Large TPMS scaffold studies
- Porosity optimisation investigations
- Parameter sweeps and sensitivity analyses
- Long-duration degradation simulations
- Multi-million element meshes

For large-scale simulations involving complex implant geometries or extended degradation periods, HPC deployment is strongly recommended to reduce wall-clock execution time and improve overall computational efficiency.

### Platform Recommendation Summary

For the best user experience, the following installation hierarchy is recommended:
1. Native Ubuntu Linux
2. Windows 11 + WSL2 + Ubuntu 24.04
3. Linux-based HPC clusters
While CPU-only systems remain fully supported, GPU-enabled environments provide significantly higher performance for large three-dimensional degradation simulations and are recommended whenever suitable hardware is available.

## 4. Software Dependencies

Dissolve 2.0 is built upon a modern scientific computing stack that combines finite-element analysis, parallel computing, GPU acceleration, mesh generation, and scientific visualization tools. This section summarises the required and optional software components needed to run the framework.

### Core Dependencies

The following packages are required for all Dissolve 2.0 installations, regardless of whether GPU acceleration is used.

#### Python

Python serves as the primary programming environment for Dissolve 2.0, providing access to numerical libraries, finite-element interfaces, data processing tools, and workflow automation.

**Recommended version:**
- Python 3.11 or newer

**Used for:**
- Solver execution
- Command-line interface
- Data processing
- Post-processing and visualization workflows

---

#### DOLFINx

DOLFINx is the finite-element framework used to discretize and solve the governing equations of species transport, passive film evolution, and level-set interface tracking.

**Recommended version:**
- DOLFINx 0.10.0

**Used for:**
- Finite-element assembly
- Function-space management
- Boundary condition handling
- Parallel execution

---

#### PETSc

PETSc provides the scalable linear algebra and solver infrastructure used throughout the Dissolve framework.

**Recommended version:**
- PETSc 3.24 or newer

**Used for:**
- Sparse matrix operations
- Iterative linear solvers
- Preconditioning
- Parallel computation
- GPU-enabled linear algebra

---

#### MPI

MPI enables parallel execution across multiple CPU cores and distributed-memory systems.

**Recommended implementation:**
- OpenMPI 4.x or newer

**Used for:**
- Domain decomposition
- Parallel assembly
- Distributed linear solves
- HPC cluster execution

---

### GPU Dependencies

The following software components are required to enable GPU acceleration.

#### CUDA

CUDA provides the core GPU computing platform used by PETSc, CuPy, and custom GPU kernels.

**Recommended version:**
- CUDA Toolkit 12.x

**Used for:**
- GPU memory management
- Sparse linear algebra
- Device-side computation
- Custom CUDA kernels

**Required for:**
- GPU-accelerated simulations

---

#### libCEED

libCEED provides high-performance matrix-free finite-element operator evaluation and GPU execution capabilities.

**Used for:**
- Matrix-free finite elements
- GPU operator evaluation
- High-order finite-element acceleration
- Reduced memory usage

**Optional but recommended for:**
- Large-scale GPU simulations
- Performance-critical workloads

---

#### CuPy

CuPy is a GPU-accelerated numerical array library with a NumPy-compatible interface.

**Used for:**
- GPU array operations
- Data movement
- Auxiliary GPU computations
- GPU-based preprocessing and post-processing

**Recommended package:**
```bash
cupy-cuda12x
```

## 5. Windows Installation (WSL2)

Dissolve 2.0 is developed and tested primarily within Linux environments. For Windows users, the recommended approach is to use **Windows Subsystem for Linux 2 (WSL2)**, which provides a full Ubuntu environment while maintaining access to Windows applications and NVIDIA GPU acceleration.

Compared with native Windows installations, WSL2 offers improved compatibility with DOLFINx, PETSc, MPI, CUDA, and other scientific computing libraries used throughout Dissolve 2.0.

### Install WSL2

Open **PowerShell** as Administrator and install WSL2:

```powershell
wsl --install
```

Restart the system when prompted.

After rebooting, verify the installation:

```powershell
wsl --status
```

You should see output indicating that WSL is installed and operational.

To view available Linux distributions:

```powershell
wsl --list --online
```

---

### Install Ubuntu

If `wsl --install` above already installed a default Ubuntu distribution, you can skip straight to launching it. To install a specific version instead (e.g. alongside an existing Ubuntu 22.04 instance used for Dissolve 1.0), install Ubuntu 24.04 LTS by name:

```powershell
wsl --install -d Ubuntu-24.04
```

Alternatively, install Ubuntu from the Microsoft Store.

Launch Ubuntu and complete the initial setup:

1. Create a username.
2. Create a password.
3. Wait for package initialization to finish.

Update the system:

```bash
sudo apt update
sudo apt upgrade -y
```

Install common development tools:

```bash
sudo apt install -y \
build-essential \
git \
curl \
wget \
unzip \
software-properties-common
```

These tools are required later when installing PETSc, DOLFINx, MPI, and other Dissolve dependencies.

---

### Verify WSL Installation

Confirm that Ubuntu is running correctly:

```bash
lsb_release -a
```

Expected output should indicate Ubuntu 24.04 LTS (or your installed release).

Verify Linux kernel information:

```bash
uname -r
```

Confirm that WSL2 is being used:

```powershell
wsl -l -v
```

Example:

```text
  NAME            STATE           VERSION
* Ubuntu-24.04    Running         2
```

The **VERSION** column should display:

```text
2
```

If version 1 is shown, convert the distribution:

```powershell
wsl --set-version Ubuntu-24.04 2
```

Verify internet access within Ubuntu:

```bash
ping -c 4 github.com
```

Verify Git:

```bash
git --version
```

Verify Python:

```bash
python3 --version
```

At this stage, your WSL2 environment should be ready for installation of:

- NVIDIA CUDA Toolkit
- OpenMPI
- PETSc
- DOLFINx
- libCEED
- Dissolve 2.0

Once WSL2 and Ubuntu are functioning correctly, proceed to the GPU setup instructions in the next section.

## 6. NVIDIA Driver and CUDA Setup

GPU acceleration in Dissolve 2.0 requires a compatible NVIDIA graphics card, recent NVIDIA drivers, and a CUDA-enabled software environment. This section describes how to configure GPU support for both native Ubuntu and WSL2 installations.

Before proceeding, ensure that WSL2 and Ubuntu have been installed successfully as described in the previous section.

### Installing NVIDIA Drivers

#### Windows (WSL2 Users)

Install the latest NVIDIA driver that supports CUDA and WSL2 directly from NVIDIA:

https://www.nvidia.com/download/

After installation, restart Windows and open PowerShell:

```powershell
nvidia-smi
```

A successful installation should display information about your GPU, driver version, and CUDA compatibility.

Example:

```text
+----------------------------------------------------------+
| NVIDIA-SMI                                               |
| Driver Version: xxx.xx                                   |
| CUDA Version: xx.x                                       |
+----------------------------------------------------------+
```

WSL2 users should not install separate Linux GPU drivers inside Ubuntu. WSL automatically exposes the Windows driver to the Linux environment.

---

#### Native Ubuntu Users

Install the recommended NVIDIA driver:

```bash
ubuntu-drivers devices
```

Install the recommended version:

```bash
sudo ubuntu-drivers autoinstall
```

Reboot the system:

```bash
sudo reboot
```

Verify successful installation:

```bash
nvidia-smi
```

---

### Installing CUDA Toolkit

The CUDA Toolkit provides the compiler, runtime libraries, and development tools required by PETSc CUDA, CuPy, and custom GPU kernels used within Dissolve 2.0.

#### Verify Existing CUDA Installation

Check whether CUDA is already installed:

```bash
nvcc --version
```

If CUDA is not found, proceed with installation.

---

#### Install CUDA Toolkit (Ubuntu)

Ubuntu's default `nvidia-cuda-toolkit` package is convenient but often lags several major versions behind NVIDIA's current releases, and typically will **not** provide CUDA 12.x. To reliably get a current CUDA 12.x toolkit, install from NVIDIA's own CUDA network repository instead:

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/wsl-ubuntu/x86_64/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt update
sudo apt install cuda-toolkit-12-6 -y
```

(For native Ubuntu rather than WSL2, replace `wsl-ubuntu` in the URL with your distribution, e.g. `ubuntu2404`, per the [NVIDIA CUDA download page](https://developer.nvidia.com/cuda-downloads).)

Verify installation:

```bash
nvcc --version
```

Expected output:

```text
Cuda compilation tools, release 12.x
```

If your PETSc build requires a different CUDA release, substitute the matching `cuda-toolkit-12-x` package name above.

---

#### Configure Environment Variables

Add CUDA to your shell environment:

```bash
nano ~/.bashrc
```

Append:

```bash
export CUDA_HOME=/usr/local/cuda
export PATH=$CUDA_HOME/bin:$PATH
export LD_LIBRARY_PATH=$CUDA_HOME/lib64:$LD_LIBRARY_PATH
```

Reload the shell:

```bash
source ~/.bashrc
```

Verify:

```bash
echo $CUDA_HOME
```

---

### Verifying GPU Access

Before installing PETSc and DOLFINx, confirm that Ubuntu can access the GPU correctly.

#### Verify NVIDIA Driver

```bash
nvidia-smi
```

Expected output includes:

- GPU model
- Driver version
- GPU memory information
- Active processes

---

#### Verify CUDA Compiler

```bash
nvcc --version
```

Expected output should report the installed CUDA version.

---

#### Verify GPU Visibility from Python

Install a lightweight CUDA-enabled package:

```bash
pip install cupy-cuda12x
```

Launch Python:

```bash
python3
```

Run:

```python
import cupy as cp
print(cp.cuda.runtime.getDeviceCount())
```

Expected output:

```text
1
```

or greater, depending on the number of available GPUs.

Exit Python:

```python
exit()
```

---

#### Verify WSL GPU Access

WSL users should also check that GPU passthrough is functioning correctly:

```bash
nvidia-smi
```

If GPU information appears inside Ubuntu, WSL CUDA support is operating correctly.

---

### Expected System State

Before proceeding to PETSc installation, the following commands should execute successfully:

```bash
nvidia-smi
```

```bash
nvcc --version
```

```bash
python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Successful completion of these checks confirms that the GPU software stack is functioning correctly and that the system is ready for PETSc CUDA, DOLFINx, and Dissolve 2.0 installation.

## 7. Python Environment Setup

Dissolve 2.0 should be installed within an isolated Python virtual environment. This approach prevents dependency conflicts with system packages and provides a reproducible software environment for research, development, and HPC deployment.

The procedures described below have been tested with Python 3.11 and newer.

### Create Virtual Environment

First, install Python development tools if they are not already available:

```bash
sudo apt update
sudo apt install -y \
python3 \
python3-pip \
python3-venv \
python3-dev
```

Verify the Python installation:

```bash
python3 --version
```

Create a dedicated virtual environment for Dissolve 2.0:

```bash
python3 -m venv dissolve-env
```

Activate the environment:

```bash
source dissolve-env/bin/activate
```

Once activated, your terminal prompt should resemble:

```text
(dissolve-env) user@ubuntu:~$
```

Upgrade pip and build tools:

```bash
pip install --upgrade \
pip \
setuptools \
wheel
```

The virtual environment should be activated whenever installing software packages or running Dissolve 2.0.

---

### Install Required Packages

Install the core Python packages required by Dissolve 2.0:

```bash
pip install \
numpy \
scipy \
matplotlib \
h5py \
meshio \
pyvista \
tqdm \
pandas
```

Install MPI Python bindings:

```bash
pip install mpi4py
```

Install PETSc Python bindings:

```bash
pip install petsc4py
```

Install GPU array support (GPU systems only):

```bash
pip install cupy-cuda12x
```

Install Jupyter Notebook (optional):

```bash
pip install jupyter
```

The exact DOLFINx installation procedure will be covered in a dedicated section later in this guide because it may depend on the PETSc configuration used on your system.

---

### Verify Python Installation

Confirm that the virtual environment is active:

```bash
which python
```

Expected output:

```text
.../dissolve-env/bin/python
```

Verify Python functionality:

```bash
python3 -c "print('Python installation successful')"
```

Verify NumPy:

```bash
python3 -c "import numpy; print(numpy.__version__)"
```

Verify MPI support:

```bash
python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.Get_rank())"
```

Verify PETSc bindings:

```bash
python3 -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
```

For GPU-enabled systems, verify CuPy:

```bash
python3 -c "import cupy as cp; print(cp.cuda.runtime.getDeviceCount())"
```

Expected output:

```text
1
```

or greater, depending on the number of available GPUs.

---

### Recommended Directory Layout

A typical Dissolve development environment may be organised as follows:

```text
~/Projects/
├── dissolve-env/
├── Dissolve-2.0/
├── petsc/
├── libCEED/
└── meshes/
```

This structure keeps software installations, simulation projects, and generated meshes organised and simplifies future upgrades.

---

### Expected System State

Before continuing to PETSc and DOLFINx installation, the following commands should execute without errors:

```bash
python3 --version
```

```bash
pip --version
```

```bash
python3 -c "import numpy"
```

```bash
python3 -c "from mpi4py import MPI"
```

```bash
python3 -c "from petsc4py import PETSc"
```

The successful completion of these checks confirms that the Python environment is correctly configured and ready for installation of PETSc, DOLFINx, libCEED, and the Dissolve 2.0 solver framework.

## 8. PETSc Installation

The Portable, Extensible Toolkit for Scientific Computation (PETSc) provides the scalable linear algebra infrastructure used throughout Dissolve 2.0. PETSc is responsible for sparse matrix assembly, iterative linear solvers, preconditioning, MPI parallelism, and GPU-accelerated computation through CUDA-enabled backends.

For GPU-enabled Dissolve 2.0 installations, PETSc must be compiled with CUDA support. This section describes a recommended source installation suitable for workstation and HPC deployments.

### Download PETSc

Install required build dependencies:

```bash
sudo apt update

sudo apt install -y \
build-essential \
gfortran \
git \
cmake \
pkg-config \
openmpi-bin \
libopenmpi-dev \
python3-dev \
libblas-dev \
liblapack-dev
```

Create a software directory:

```bash
mkdir -p ~/software
cd ~/software
```

Clone the PETSc repository:

```bash
git clone -b release https://gitlab.com/petsc/petsc.git
```

Enter the PETSc source directory:

```bash
cd petsc
```

The installation examples below assume PETSc is located at:

```text
~/software/petsc
```

---

### Configure PETSc with CUDA

Define PETSc environment variables:

```bash
export PETSC_DIR=$HOME/software/petsc
export PETSC_ARCH=arch-linux-cuda
```

Configure PETSc. In addition to CUDA support, the flags below are required so that PETSc builds its Python bindings (`petsc4py`) and shared libraries in a way that DOLFINx and `petsc4py` can link against — omitting them is a common cause of DOLFINx silently ending up linked to a different, non-CUDA PETSc:

```bash
./configure \
--with-debugging=0 \
--with-cuda=1 \
--with-shared-libraries=1 \
--with-petsc4py=1 \
--with-mpi4py=1 \
--download-f2cblaslapack \
--download-hypre \
--download-mumps \
--download-scalapack \
PETSC_ARCH=$PETSC_ARCH
```

#### Configuration Options

```text
--with-cuda=1
```

Enables NVIDIA GPU support.

```text
--with-shared-libraries=1
```

Builds shared (`.so`) libraries, required for `petsc4py` and DOLFINx to link against this PETSc build at runtime.

```text
--with-petsc4py=1
--with-mpi4py=1
```

Builds PETSc's own Python bindings against this specific PETSc/CUDA build, instead of relying on a separately pip-installed `petsc4py` that may point at a different PETSc.

```text
--download-hypre
```

Installs Hypre for algebraic multigrid preconditioning.

```text
--download-mumps
```

Installs the MUMPS sparse direct solver.

```text
--download-scalapack
```

Provides distributed dense linear algebra support.

```text
--with-debugging=0
```

Builds an optimized release version suitable for production simulations.

At the end of configuration PETSc should report:

```text
Configure stage complete
```

and display no fatal errors.

---

### Build PETSc

Compile PETSc:

```bash
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH all
```

This process may take between several minutes and over an hour depending on:

- CPU performance
- internet speed
- downloaded packages
- available system resources

Monitor progress using:

```bash
tail -f configure.log
```

After compilation completes successfully:

```bash
make PETSC_DIR=$PETSC_DIR PETSC_ARCH=$PETSC_ARCH check
```

This executes PETSc verification tests.

---

### Configure Environment Variables

Add PETSc variables permanently:

```bash
nano ~/.bashrc
```

Append:

```bash
export PETSC_DIR=$HOME/software/petsc
export PETSC_ARCH=arch-linux-cuda
```

Reload:

```bash
source ~/.bashrc
```

Confirm:

```bash
echo $PETSC_DIR
```

```bash
echo $PETSC_ARCH
```

Expected:

```text
/home/user/software/petsc
```

```text
arch-linux-cuda
```

---

### Verify PETSc Installation

Check the installation:

```bash
ls $PETSC_DIR
```

Verify PETSc configuration:

```bash
$PETSC_DIR/$PETSC_ARCH/lib/petsc/bin/petscvariables
```

Confirm CUDA support:

```bash
grep CUDA \
$PETSC_DIR/$PETSC_ARCH/lib/petsc/conf/petscvariables
```

You should observe CUDA-related configuration entries.

Verify through Python:

```bash
python3 -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
```

Example output:

```text
(3, 24, 0)
```

Verify MPI compatibility:

```bash
mpirun -np 2 python3 -c "from petsc4py import PETSc; print(PETSc.COMM_WORLD.rank)"
```

Expected output:

```text
0
1
```

Verify GPU availability through PETSc:

```bash
python3 -c "from petsc4py import PETSc; print(PETSc.Options())"
```

No errors should occur during initialization.

---

### Common Installation Locations

Typical installation layout:

```text
~/software/
├── petsc/
├── libCEED/
├── Dissolve-2.0/
└── dissolve-env/
```

This structure keeps scientific software installations organised and simplifies upgrades and troubleshooting.

---

### Expected System State

Before proceeding to DOLFINx installation, the following should work successfully:

```bash
echo $PETSC_DIR
```

```bash
echo $PETSC_ARCH
```

```bash
python3 -c "from petsc4py import PETSc"
```

```bash
mpirun -np 2 python3 -c "from petsc4py import PETSc"
```

At this stage PETSc should be fully operational, MPI-enabled, and configured for CUDA acceleration, providing the linear solver infrastructure required by Dissolve 2.0.

## 9. DOLFINx Installation

DOLFINx is the finite-element framework that forms the computational foundation of Dissolve 2.0. It provides mesh handling, finite-element discretisation, function spaces, variational formulations, parallel execution, and interfaces to PETSc solvers.

Dissolve 2.0 has been developed and tested using **DOLFINx 0.10.0**, and users are strongly encouraged to install this version to ensure compatibility and reproducibility.

> **Why build from source instead of `pip install fenics-dolfinx`:** a plain PyPI wheel install typically bundles or depends on its own PETSc build, rather than linking against the CUDA-enabled PETSc built in Section 8. If you install the wheel after building custom PETSc, the CUDA PETSc build may never actually be used at runtime, silently defeating the GPU setup. Building DOLFINx from source against your existing `$PETSC_DIR`/`$PETSC_ARCH` avoids this. If you only need CPU execution, `pip install fenics-dolfinx` is simpler and sufficient — skip the source build below in that case.

### Install DOLFINx (source build, linked against your PETSc)

Before installing DOLFINx, ensure that:

- Python is installed and functioning correctly.
- PETSc has been successfully built with `--with-petsc4py=1` and `--with-shared-libraries=1` (Section 8).
- MPI is available.
- The Dissolve Python environment is activated.

Activate the virtual environment:

```bash
source ~/dissolve-env/bin/activate
```

Update pip and install build tooling:

```bash
pip install --upgrade pip
pip install --upgrade cmake ninja pkgconfig setuptools wheel
```

Clone the DOLFINx stack (basix, ufl, ffcx, dolfinx) and check out the release matching your target version — check the [DOLFINx releases page](https://github.com/FEniCS/dolfinx/releases) for the companion basix/ufl/ffcx tags that go with `v0.10.0`:

```bash
cd ~/software
git clone https://github.com/FEniCS/basix.git
git clone https://github.com/FEniCS/ufl.git
git clone https://github.com/FEniCS/ffcx.git
git clone https://github.com/FEniCS/dolfinx.git

cd dolfinx && git checkout v0.10.0 && cd ..
```

Build and install Basix (C++ library and Python interface):

```bash
cd basix
cmake -G Ninja -B build-dir -S cpp
cmake --build build-dir
sudo cmake --install build-dir
pip install ./python
cd ..
```

Install UFL and FFCx (pure Python):

```bash
pip install ./ufl
pip install ./ffcx
```

Build and install the DOLFINx C++ layer and Python interface, using the PETSc built in Section 8:

```bash
cd dolfinx
cmake -G Ninja -B build-dir -S cpp \
    -DPETSC_DIR=$PETSC_DIR \
    -DPETSC_ARCH=$PETSC_ARCH
cmake --build build-dir
sudo cmake --install build-dir
pip install ./python
cd ..
```

Install supporting scientific packages if not already present:

```bash
pip install \
numpy \
scipy \
mpi4py \
petsc4py
```

---

### Verify DOLFINx Installation

Verify that DOLFINx can be imported successfully:

```bash
python3 -c "import dolfinx; print(dolfinx.__version__)"
```

Expected output:

```text
0.10.0
```

Verify MPI support:

```bash
python3 -c "from mpi4py import MPI; print(MPI.COMM_WORLD.size)"
```

Expected output:

```text
1
```

Verify PETSc integration:

```bash
python3 -c "from petsc4py import PETSc; print(PETSc.Sys.getVersion())"
```

Verify DOLFINx mesh creation:

```bash
python3
```

```python
from mpi4py import MPI
from dolfinx import mesh

domain = mesh.create_unit_cube(
    MPI.COMM_WORLD,
    10, 10, 10
)

print(domain.topology.dim)
```

Expected output:

```text
3
```

Exit Python:

```python
exit()
```

---

### Verify Parallel Execution

Test MPI communication with DOLFINx:

```bash
mpirun -np 4 python3 -c "
from mpi4py import MPI
print('Rank', MPI.COMM_WORLD.rank)
"
```

Expected output:

```text
Rank 0
Rank 1
Rank 2
Rank 3
```

Verify that DOLFINx functions correctly under MPI:

```bash
mpirun -np 4 python3 -c "
from mpi4py import MPI
from dolfinx import mesh
domain = mesh.create_unit_cube(MPI.COMM_WORLD,10,10,10)
print('Rank', MPI.COMM_WORLD.rank, 'OK')
"
```

All ranks should complete successfully.

---

### Verify XDMF Support

Dissolve 2.0 uses XDMF/HDF5 mesh files extensively.

Check XDMF functionality:

```bash
python3 -c "
from dolfinx.io import XDMFFile
print('XDMF support available')
"
```

Expected output:

```text
XDMF support available
```

---

### Verify a Dissolve Dependency Test

Run a combined dependency check:

```bash
python3 -c "
import dolfinx
import numpy
import mpi4py
import petsc4py
print('Dissolve dependency check successful')
"
```

Expected output:

```text
Dissolve dependency check successful
```

---

### Typical DOLFINx Workflow

Within Dissolve 2.0, DOLFINx is used for:

- Mesh loading and management
- Function-space creation
- Variational form assembly
- Boundary-condition application
- Field interpolation
- Parallel finite-element computations
- PETSc solver integration
- XDMF/HDF5 I/O

These capabilities form the core numerical infrastructure upon which the Dissolve degradation model is built.

---

### Expected System State

Before proceeding to libCEED installation, the following commands should execute successfully:

```bash
python3 -c "import dolfinx"
```

```bash
python3 -c "from dolfinx import mesh"
```

```bash
python3 -c "from dolfinx.io import XDMFFile"
```

```bash
mpirun -np 4 python3 -c "import dolfinx"
```

At this stage, DOLFINx should be fully operational and integrated with Python, MPI, and PETSc, providing the finite-element infrastructure required by Dissolve 2.0.

## 10. libCEED Installation

libCEED provides matrix-free finite-element operator evaluation and GPU-native kernels used by Dissolve 2.0's `gpu/` module. It is optional — Dissolve 2.0 runs correctly without it, falling back to PETSc CUDA assembly — but is recommended for large-scale GPU simulations.

### Download and Build libCEED

```bash
cd ~/software
git clone https://github.com/CEED/libCEED.git
cd libCEED
```

Build libCEED with CUDA support (libCEED auto-detects `nvcc` from `$CUDA_HOME`/`$PATH`):

```bash
make configure CUDA_DIR=$CUDA_HOME
make -j$(nproc)
```

### Install the Python Bindings

```bash
pip install .
```

### Configure Environment Variables

```bash
nano ~/.bashrc
```

Append:

```bash
export CEED_DIR=$HOME/software/libCEED
export LD_LIBRARY_PATH=$CEED_DIR/lib:$LD_LIBRARY_PATH
```

Reload:

```bash
source ~/.bashrc
```

### Verify libCEED Installation

```bash
python3 -c "import libceed; ceed = libceed.Ceed('/gpu/cuda'); print(ceed)"
```

If no CUDA-enabled GPU is available, verify the CPU backend instead:

```bash
python3 -c "import libceed; ceed = libceed.Ceed('/cpu/self'); print(ceed)"
```

### Expected System State

Before proceeding to install Dissolve 2.0, the following should succeed:

```bash
python3 -c "import libceed"
```

```bash
echo $CEED_DIR
```

## 11. Installing Dissolve 2.0

Once all dependencies have been installed successfully, Dissolve 2.0 can be cloned and configured.

### Clone Repository

Clone the repository into your WSL or Linux home directory:

```bash
cd ~

git clone https://github.com/sanjukadupul-wq/Dissolve-2.0.git

cd Dissolve-2.0
```

### Verify Repository Structure

Confirm that the repository was cloned correctly:

```bash
ls
```

The repository should contain:

```text
config/
domain/
physics/
numerics/
gpu/
io/
analysis/
meshing/
dissolve.py
INSTALL.md
README.md
```

### Activate Environment

Activate the Python environment created previously:

```bash
source ~/dissolve-env/bin/activate
```

### Validate Installation

Verify all major dependencies:

```bash
python3 -c "import dolfinx"
python3 -c "from petsc4py import PETSc"
python3 -c "import mpi4py"
```

GPU users should additionally verify:

```bash
python3 -c "import cupy"
nvidia-smi
```

No errors should be reported.

---

## 12. Quick Start

This section demonstrates a typical Dissolve 2.0 workflow.

### Generate a Mesh

Generate a scaffold mesh:

```bash
python meshing/mesh_generator_disc_adaptive.py
```

or generate a stent mesh:

```bash
python meshing/mesh_generator_stent_adaptive.py
```

### Run Your First Simulation

```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --sim_duration 24
```

### CPU Execution

```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --use_gpu 0
```

### GPU Execution

```bash
python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --use_gpu 1
```

### MPI Execution

```bash
mpirun -np 4 python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf
```

---

## 13. Running Simulations

### Basic Usage

The basic execution command is:

```bash
python3 dissolve.py --input_mesh <mesh_file>
```

### Common Parameters

```bash
--sim_duration
--dt_hours
--use_gpu
--adaptive_dt
--enable_redistance
--mechanics
```

Full parameter descriptions are provided in the project README.

### Example: TPMS Scaffold

```bash
python3 dissolve.py \
    --input_mesh gyroid.xdmf \
    --sim_duration 672 \
    --use_gpu 1
```

### Example: Stent

```bash
python3 dissolve.py \
    --input_mesh stent.xdmf \
    --sim_duration 720 \
    --use_gpu 1
```

### Parameter Studies

Example parameter sweep:

```bash
for p in 0.3 0.5 0.7
do
    python3 dissolve.py \
        --input_mesh scaffold.xdmf \
        --film_porosity $p
done
```

### Long-Duration Simulations

For simulations exceeding several weeks of simulated degradation:

```bash
--adaptive_dt 1
--enable_redistance 1
```

are recommended.

---

## 14. Output Files

Dissolve 2.0 automatically generates simulation outputs for visualization and analysis.

### Result Files

Examples:

```text
result.txt
mass_loss.csv
diagnostics.csv
```

### Visualization Files

```text
*.vtu
*.pvd
*.xdmf
```

These can be opened directly in ParaView.

### Diagnostic Files

Diagnostics may include:

- Solver timings
- Nonlinear iterations
- Linear solver convergence
- GPU utilization metrics

### Summary Reports

At the end of execution Dissolve generates:

```text
summary.png
```

and terminal summary statistics.

### Typical Output Structure

```text
output/
├── result.txt
├── diagnostics.csv
├── summary.png
├── degradation.pvd
└── *.vtu
```

---

## 15. Visualization and Post-Processing

### Installing ParaView

Download:

https://www.paraview.org

### Opening Simulation Results

Open:

```text
degradation.pvd
```

or

```text
mesh.xdmf
```

using ParaView.

### Recommended Visualization Fields

Useful fields include:

- Level-set interface
- Oxygen concentration
- Chloride concentration
- Hydroxide concentration
- Zinc concentration
- Passive film evolution

### Publication Figures

Recommended workflow:

1. Load results in ParaView.
2. Apply clipping/filtering.
3. Use high-resolution rendering.
4. Export PNG or TIFF images.

---

## 16. High-Performance Computing (HPC)

### Slurm Example

```bash
#!/bin/bash
#SBATCH --job-name=dissolve
#SBATCH --nodes=1
#SBATCH --ntasks=8
#SBATCH --time=24:00:00

source ~/dissolve-env/bin/activate

mpirun -np 8 python3 dissolve.py \
    --input_mesh mesh_adaptive_800k.xdmf \
    --use_gpu 1
```

Submit:

```bash
sbatch run.slurm
```

### Multi-GPU Execution

GPU-enabled clusters may support:

```bash
mpirun -np 4 python3 dissolve.py
```

with one MPI process per GPU.

### Recommended Resources

| Mesh Size | Recommended Memory |
|-----------|---------------------|
| <100k | 8 GB |
| 100k-500k | 16 GB |
| 500k-2M | 32 GB |
| >2M | 64 GB+ |

---

## 17. Verification and Benchmarking

### Dependency Verification

```bash
python3 -c "import dolfinx"
```

```bash
python3 -c "from petsc4py import PETSc"
```

### MPI Verification

```bash
mpirun -np 4 hostname
```

### GPU Verification

```bash
nvidia-smi
```

### Expected Behaviour

A healthy installation should:

- Detect CUDA.
- Detect MPI.
- Load meshes successfully.
- Complete small test runs without errors.

---

## 18. Updating Dissolve

### Update Repository

```bash
git pull
```

### Update Python Packages

```bash
pip install --upgrade pip
pip list --outdated
```

### Updating Dependencies

When upgrading PETSc or DOLFINx:

1. Backup simulations.
2. Create a new environment.
3. Rebuild and verify dependencies.
4. Re-run benchmark cases.

---

## 19. Troubleshooting

### CUDA Not Detected

Verify:

```bash
nvidia-smi
```

and

```bash
nvcc --version
```

### PETSc Errors

Verify:

```bash
echo $PETSC_DIR
echo $PETSC_ARCH
```

### DOLFINx Import Errors

Verify:

```bash
python3 -c "import dolfinx"
```

### MPI Issues

Verify:

```bash
mpirun -np 2 hostname
```

### Mesh Loading Problems

Check:

- Mesh path is correct.
- Associated HDF5 files exist.
- XDMF references are valid.

### Out-of-Memory Errors

Recommended actions:

- Reduce mesh size.
- Increase RAM allocation.
- Use adaptive timestepping.
- Run on HPC resources.

---

## 20. Frequently Asked Questions

### Is a GPU required?

No. Dissolve 2.0 can run entirely on CPU hardware.

### Can I use my own mesh?

Yes. Any valid DOLFINx-compatible XDMF/HDF5 tetrahedral mesh may be used.

### Does Dissolve support MPI?

Yes. MPI execution is supported through OpenMPI and PETSc.

### Is libCEED required?

No. However, it is strongly recommended for high-performance GPU workloads.

### Which operating system is recommended?

Ubuntu 24.04 LTS, or WSL2 with Ubuntu 24.04.

---

## 21. Citation

If you use Dissolve 2.0 in academic research, please cite the accompanying publication when available.

Example:

```text
Ariyarathna H.
Dissolve 2.0: A GPU-Accelerated Framework for Predicting
Biodegradable Zn-Based Implant Degradation.
2026.
```

Users are also encouraged to acknowledge the original Dissolve framework.

---

## 22. License

Dissolve 2.0 is released under the GNU General Public License Version 3.0 (GPL-3.0).

See:

```text
LICENSE
```

for the complete license text.

---

## 23. Support and Contact

For bug reports, feature requests, and technical questions, please use the GitHub Issues system:

```text
https://github.com/sanjukadupul-wq/Dissolve-2.0/issues
```

When reporting issues, include:

- Operating system
- Python version
- PETSc version
- DOLFINx version
- CUDA version
- Full error message
- Steps required to reproduce the issue

Providing complete information will greatly assist debugging and future development efforts.
