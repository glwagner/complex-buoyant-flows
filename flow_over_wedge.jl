using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!
using Oceananigans.Architectures: device

using KernelAbstractions: MultiEvent
using Printf
using GLMakie

arch = CPU()
Nz = 64 # Resolution
κ = 1e-3 # Diffusivity and viscosity (Prandtl = 1)
U = 1

underlying_grid = RectilinearGrid(arch,
                                  size = (2Nz, Nz),
                                  x = (-4, 4),
                                  z = (0, 4),
                                  halo = (3, 3),
                                  topology = (Periodic, Flat, Bounded))

const slope = 1/2
@inline wedge(x, y) = slope * min(x, -x) + 1
grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(wedge))

model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            closure = IsotropicDiffusivity(ν=κ, κ=κ),
                            tracers = nothing,
                            coriolis = nothing,
                            buoyancy = nothing)

# Linear stratification
set!(model, u = U)

Δt = 5e-2 * (grid.Lx / grid.Nz) / U
simulation = Simulation(model, Δt = Δt, stop_time = 100)

# wizard = TimeStepWizard(cfl=0.01, max_change=1.2, max_Δt=10*Δt)
# simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(s) =
    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f",
                   100 * s.model.clock.time / s.stop_time,
                   s.model.clock.iteration,
                   s.model.clock.time)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

prefix = "flow_over_wedge_Nz$(Nz)"
simulation.output_writers[:fields] = JLD2OutputWriter(model, model.velocities,
                                                      schedule = TimeInterval(0.1),
                                                      prefix = prefix,
                                                      field_slicer = nothing,
                                                      force = true)

run!(simulation)

@info """
    Simulation complete.
    Runtime: $(prettytime(simulation.run_wall_time))
"""

filepath = prefix * ".jld2"
ut = FieldTimeSeries(filepath, "u", grid=grid)
wt = FieldTimeSeries(filepath, "w", grid=grid)

times = ut.times
Nt = length(times)

ut = [ut[n] for n = 1:Nt]
wt = [wt[n] for n = 1:Nt]

# Preprocess
eventss = []
for n = 1:Nt
    for f in (ut[n], wt[n])
        push!(eventss, mask_immersed_field!(f, NaN))
    end
end

wait(device(CPU()), MultiEvent(Tuple(eventss)))

max_u = 2.0
min_u = 0.5
max_w = 1.0

n = Observable(1)

ui(n) = interior(ut[n])[:, 1, :]
wi(n) = interior(wt[n])[:, 1, :]

fluid_u(n) = filter(isfinite, ui(n)[:])
fluid_w(n) = filter(isfinite, wi(n)[:])

ΣUt = [sum(fluid_u(n)) for n = 1:Nt]

up = @lift ui($n)
wp = @lift wi($n)

fig = Figure(resolution=(1800, 1200))

ax = Axis(fig[1, 1], title="x-velocity")
hm = heatmap!(ax, up, colorrange=(min_u, max_u), colormap=:thermal)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="z-velocity")
hm = heatmap!(ax, wp, colorrange=(-max_w, max_w), colormap=:balance)
cb = Colorbar(fig[2, 2], hm)

title_gen(n) = @sprintf("Flow over wedge at t = %.2f", times[n])
title_str = @lift title_gen($n)
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

display(fig)
