using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Grids: xnode, znode
using KernelAbstractions: MultiEvent
using Printf
using GLMakie
using SpecialFunctions

arch = CPU()
Nx = 256
Nz = 64 # Resolution
κ = 1e-4 # Diffusivity and viscosity (Prandtl = 1)

underlying_grid = RectilinearGrid(arch,
                                  size = (Nx, Nz),
                                  x = (0, 5),
                                  z = (-0.05, 1),
                                  halo = (3, 3),
                                  topology = (Bounded, Flat, Bounded))

@inline bottom_topography(x, y) =  0.0



grid = ImmersedBoundaryGrid(underlying_grid, GridFittedBottom(bottom_topography))



model = NonhydrostaticModel(grid = grid,
                            advection = WENO5(),
                            closure = IsotropicDiffusivity(ν=κ, κ=κ),
                            coriolis = nothing,
                            tracers = :b,
                            buoyancy = BuoyancyTracer())

b₀(x, y, z) =0.5*(erf.((x.-1.0)*10).-1.0)
set!(model, u = 0.0, b = b₀)

Δt = 5e-4
simulation = Simulation(model, Δt = Δt, stop_time = 20)

 wizard = TimeStepWizard(cfl=0.1, max_change=1.1, max_Δt=100*Δt)
 simulation.callbacks[:wizard] = Callback(wizard, IterationInterval(10))

progress(s) =
    @info @sprintf("[%.2f%%], iteration: %d, time: %.3f",
                   100 * s.model.clock.time / s.stop_time,
                   s.model.clock.iteration,
                   s.model.clock.time)

simulation.callbacks[:progress] = Callback(progress, IterationInterval(10))

prefix = "flat_gc_immersed_bottom"
simulation.output_writers[:velocities] = JLD2OutputWriter(model, merge(model.velocities, model.tracers),
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
bt = FieldTimeSeries(filepath, "b", grid=grid)

times = ut.times
Nt = length(times)

ut = [ut[n] for n = 1:Nt]
wt = [wt[n] for n = 1:Nt]
bt = [bt[n] for n = 1:Nt]

# Preprocess
eventss = []
for n = 1:Nt
    for f in (ut[n], wt[n])
        push!(eventss, mask_immersed_field!(f, NaN))
    end
end

wait(device(CPU()), MultiEvent(Tuple(eventss)))

max_u = 0.5
min_u = -0.5
max_w = 0.5
n = Observable(1)

ui(n) = interior(ut[n])[:, 1, :]
wi(n) = interior(wt[n])[:, 1, :]
bi(n) = interior(bt[n])[:, 1, :]

fluid_u(n) = filter(isfinite, ui(n)[:])
fluid_w(n) = filter(isfinite, wi(n)[:])

ΣUt = [sum(fluid_u(n)) for n = 1:Nt]

up = @lift ui($n)
wp = @lift wi($n)
bp = @lift bi($n)

fig = Figure(resolution=(1800, 1800))

ax = Axis(fig[1, 1], title="x-velocity")
hm = heatmap!(ax, up, colorrange=(min_u, max_u), colormap=:thermal)
cb = Colorbar(fig[1, 2], hm)

ax = Axis(fig[2, 1], title="z-velocity")
hm = heatmap!(ax, wp, colorrange=(-max_w, max_w), colormap=:balance)
cb = Colorbar(fig[2, 2], hm)

ax = Axis(fig[3, 1], title="buoyancy")
hm = heatmap!(ax, bp, colorrange=(-1,0), colormap=:balance)
cb = Colorbar(fig[3, 2], hm)

title_gen(n) = @sprintf("Flow over wedge at t = %.2f", times[n])
title_str = @lift title_gen($n)
ax_t = fig[0, :] = Label(fig, title_str)

record(fig, prefix * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end


display(fig)

 f = Figure(resolution = (1000,1000))

ax = Axis(f[1, 1], xlabel = "time", ylabel = "% change",
    title = "Title")
 lines!(ax,times,100*(sumbt.-sumbt[1])./sumbt[1])
 save(prefix * "_mass_change.png",f)
