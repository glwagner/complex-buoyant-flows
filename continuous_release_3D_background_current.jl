# # Lock-release gravity current example

using Oceananigans
using Oceananigans.Grids
using Oceananigans.Advection
using Oceananigans.OutputWriters
using Oceananigans.Diagnostics
using CUDA
CUDA.allowscalar(true)

using Printf
using SpecialFunctions

# Physical and numerical parameters

Nx = 1024
Ny = 1024  # x resolution
Nz = 128  # z resolution
Lx = 30
Ly = 30  # domain extent
Lz = 3.0 # vertical domain extent
cross_flow = 0.0 # crossflow

Re = 20000
Pe = Re
iRe = 1/Re
iPe = 1/Pe

topology = (Periodic, Bounded, Bounded)

grid = RegularCartesianGrid(size=(Nx, Ny, Nz), x=(0, Lx), y=(0, Ly), z=(0, Lz), topology=topology)

"""
    MomentumFluxBottomWallModel{X, FT} <: Function

Contains parameters for a "wall model" for unresolved momentum flux
in the `X` direction due to turubulent flow past a no-slip bottom boundary.

Follows

    Abkar, M. and Moin, P., "Large-Eddy Simulation of Thermally Stratified Atmospheric
    Boundary-Layer Flow Using a Minimum Dissipation Model", Boundary-Layer Meterol. (2017).

    https://link.springer.com/content/pdf/10.1007/s10546-017-0288-4.pdf

and in particular, equations (23)-(24) therein. Note: we ignore the "stability corrections".
These are zero for insulating boundary conditions.

    MomentumFluxBottomWallModel{X}(FT=Float64; roughness_length, von_karman_constant=0.41)

Returns a callable object that models unresolved momentum flux in the `X` direction due to
turubulent flow past a no-slip bottom boundary. The `von_karman_constant` is really a drag
coefficient; the name is for historical purposes. `FT` specifies the floating point type of
the parameters.
"""
struct MomentumFluxBottomWallModel{X, FT} <: Function
    roughness_length :: FT
    von_karman_constant :: FT

    function MomentumFluxBottomWallModel{X}(FT=Float64; roughness_length, von_karman_constant=0.41) where X
        return new{X, FT}(roughness_length, von_karman_constant)
    end
end

@inline function (wm::MomentumFluxBottomWallModel{:x})(i, j, grid, clock, model_fields)
    u, v, w, b = model_fields
    @inbounds speed = sqrt(u[i, j, 1]^2 + v[i, j, 1]^2)
    roughness_factor = log(grid.Δz / (2 * wm.roughness_length))^2
    return @inbounds - wm.von_karman_constant * speed * u[i, j, 1] / roughness_factor
end

@inline function (wm::MomentumFluxBottomWallModel{:y})(i, j, grid, clock, model_fields)
    u, v, w, b = model_fields
    @inbounds speed = sqrt(u[i, j, 1]^2 + v[i, j, 1]^2)
    roughness_factor = log(grid.Δz / (2 * wm.roughness_length))^2 # works for constant vertical spacing only
    return @inbounds - wm.von_karman_constant * speed * v[i, j, 1] / roughness_factor
end

# We use roughness_length = 0.1 m.
bottom_x_momentum_flux = MomentumFluxBottomWallModel{:x}(roughness_length=0.1)
bottom_y_momentum_flux = MomentumFluxBottomWallModel{:y}(roughness_length=0.1)

ubcs = UVelocityBoundaryConditions(grid,
                                   bottom = BoundaryCondition(Flux, bottom_x_momentum_flux, discrete_form=true))

vbcs = VVelocityBoundaryConditions(grid,
                                   bottom = BoundaryCondition(Flux, bottom_y_momentum_flux, discrete_form=true),
                                   south = BoundaryCondition(NormalFlow, cross_flow),
                                   north = BoundaryCondition(NormalFlow, cross_flow)
                                   )

bbcs = TracerBoundaryConditions(grid, south = BoundaryCondition(Value, 0))

# ## Model instantiation and initial condition

""" Returns the distance to the point (Lx/2, y₀ - a*t, Lz/2) moving at speed `a`. """
@inline distance_to_source(x, y, z, t, a, y₀, Lx, Ly, R) = sqrt((x - y₀ - a*t)^2 + (y - Ly/2)^2 + (z - R)^2)

@inline b_forcing_func(x, y, z, t, p) =
    - 0.5 * (1 - erf((distance_to_source(x, y, z, t, p.a, p.y₀, p.Lx, p.Ly, p.R) - p.R) / p.δᴸ))

# These don't work because ContinuousForcing functions don't compile on the GPU right now:
# b_forcing = Forcing(b_forcing_func, parameters=(a=0.5, y₀=2, R=0.5, δᴸ=0.1))
# u_forcing = Relaxation(rate=1/60, mask=GaussianMask{:x}(center=grid.Lx, width=grid.Lx/10), target=0.1)

# Our work-around is to use DiscreteForcing instead:

@inline b_discrete_forcing_func(i, j, k, grid, clock, model_fields, parameters) =
    @inbounds b_forcing_func(grid.xC[i], grid.yC[j], grid.zC[k], clock.time, parameters)

b_forcing = Forcing(b_discrete_forcing_func, discrete_form=true,
                    parameters=(a=1.0, y₀=2, R=0.5, δᴸ=0.05, Lx=grid.Lx, Ly=grid.Ly, Lz=grid.Lz))

@inline mu(y, Ly, width, mu0) = mu0 * exp(-(y - Ly)^2 / (2 * width^2))

@inline u_forcing_func(i, j, k, grid, clock, model_fields, p) =
    @inbounds mu(grid.yC[j], grid.Ly, p.sponge_width, p.mu0) * (p.cross_flow - model_fields.u[i, j, k])

u_forcing = Forcing(u_forcing_func, discrete_form=true,
                    parameters=(mu0=1/60, sponge_width=grid.Ly/20, cross_flow=cross_flow))

model = IncompressibleModel(
                   grid = grid,
            timestepper = :RungeKutta3,
              advection = WENO5(),
           architecture = GPU(),
    boundary_conditions = (u=ubcs, v=vbcs, b=bbcs),
                closure = AnisotropicMinimumDissipation(ν=iRe, κ=iPe),
                tracers = :b,
               buoyancy = BuoyancyTracer(),
               forcing  = (u=u_forcing, b=b_forcing))

eps_noise = 1e-3
Ξ(y,z) = randn() * z / Lz * (1 - z / Lz) * y / Ly * (1 + y / Ly) # noise

# Set initial condition
b₀(x, y, z) = 0
u₀(x, y, z) = cross_flow + eps_noise * Ξ(y, z) 

set!(model, b=b₀, u=u₀)

### Progress diagnostic function

umax = FieldMaximum(abs, model.velocities.u)
vmax = FieldMaximum(abs, model.velocities.v)
wmax = FieldMaximum(abs, model.velocities.w)

wall_clock = time_ns()

function print_progress(simulation)
    model = simulation.model

    # Print a progress message
    msg = @sprintf("i: %04d, t: %s, Δt: %s, umax = (%.1e, %.1e, %.1e) ms⁻¹, wall time: %s\n",
                   model.clock.iteration,
                   prettytime(model.clock.time),
                   prettytime(wizard.Δt),
                   umax(), vmax(), wmax(),
                   prettytime(1e-9 * (time_ns() - wall_clock))
                  )

    @info msg

    return nothing
end

# Time step wizard based on CFL condition for RK3 (theoretical CFL < √3)
wizard = TimeStepWizard(cfl=1.0, Δt=0.01, max_change=1.2, max_Δt=0.1)

simulation = Simulation(model,
                        iteration_interval = 200,
                        Δt = wizard,
                        stop_time = 20,
                        progress = print_progress)

# Output

simulation.output_writers[:fields] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
                                                      time_interval = 0.2,
                                                             prefix = "continuous_release_3D_background_current",
                                                              force = true)

run!(simulation)
