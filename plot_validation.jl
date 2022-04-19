using Oceananigans
using Oceananigans.ImmersedBoundaries: ImmersedBoundaryGrid, GridFittedBottom, mask_immersed_field!
using Oceananigans.Architectures: device
using Oceananigans.Grids: xnode, znode
using KernelAbstractions: MultiEvent
using Printf
using GLMakie
using SpecialFunctions
using Statistics
using Interpolations
using ElectronDisplay
using Statistics
using JLD2

arch = CPU()

prefixes = pwd();

filepath = prefixes * "/immersed_bump_lock_release.jld2";


bt = FieldTimeSeries(filepath, "b", grid=nothing);
x = bt.grid.xᶜᵃᵃ
z = bt.grid.zᵃᵃᶜ

x = x.parent[4:end-3]
z = z.parent[4:end-3]


times = bt.times
Nt = length(times)

bt = [bt[n] for n = 1:Nt];


n = Observable(1)
bi(n) = interior(bt[n])[:,1, :]
bp = @lift bi($n)

mass = [sum(interior(bt[k])[:,1, :]) for k = 1:Nt];



figure = Figure(
                resolution = (2000, 1000),
                fontsize = 12)

ax = Axis(figure[1,1], aspect = DataAspect())


surface!(ax,x,z,bp; shading=false, colormap = :deep)
display(figure)
#cb = Colorbar(figure[1, 2], hm)

title_gen(n) = @sprintf("lock release at t = %.2f", times[n])
title_str = @lift title_gen($n)
#ax_t = figure[0, :] = Label(figure, title_str)

record(figure, prefixes * ".mp4", 1:Nt, framerate=8) do nt
    n[] = nt
end

f = Figure(resolution = (2000, 1000),
fontsize = 24)
ax = Axis(f[1,1], title="relative mass change in %")
lines!(times,100*(mass.-mass[1])./mass[1])

#co = contourf(xx,yy,bflat, levels = 20)
 #Colorbar(f[1, 2], co)
