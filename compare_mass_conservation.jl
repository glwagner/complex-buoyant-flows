using CairoMakie
using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Grids: xnode, znode
using KernelAbstractions: MultiEvent
using Printf
using SpecialFunctions


arch = CPU()

prefixes = ["flat_gc_real_bottom","flat_gc_immersed_bottom","flat_gc_immersed_Re2000"]

f = Figure(resolution = (1000,1000))

ax = Axis(f[1, 1], xlabel = "time", ylabel = "% change",
   title = "Title")
for nmes in prefixes

filepath = nmes * ".jld2"

FieldDataset(filepath; architecture=CPU(), grid=nothing, backend=InMemory(), metadata_paths=["metadata"])
ut = FieldTimeSeries(filepath, "u", grid=nothing)
wt = FieldTimeSeries(filepath, "w", grid=nothing)
bt = FieldTimeSeries(filepath, "b", grid=nothing)

times = ut.times
Nt = length(times)

ut = [ut[n] for n = 1:Nt]
wt = [wt[n] for n = 1:Nt]
bt = [bt[n] for n = 1:Nt]

sumbt = [sum(bt[n]) for n=1:Nt]

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

lines!(ax,times,100*(sumbt.-sumbt[1])./sumbt[1],label=nmes)
display(f)
end
axislegend()
save("compare_leakage.png",f)
