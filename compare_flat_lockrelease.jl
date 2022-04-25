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


f = Figure(resolution = (800, 600))
ax = Axis(f[1,1], xlabel = "time", ylabel = "Front position")
labels = ["immersed no-slip", "real no-slip", "immersed free-slip"];
nmes = ["immersed_flat_lock_release";"real_flat_lock_release";"slip_flat_lock_release"];

for i = 1:length(nmes)
     filepath = prefixes * "/" * nmes[i] *".jld2";

     bts = FieldTimeSeries(filepath, "b", grid=nothing);
     xs = bts.grid.xᶜᵃᵃ
#z = bt.grid.zᵃᵃᶜ

x = xs.parent[4:end-3]
#z = z.parent[4:end-3]


times = bts.times
Nt = length(times)

bt = [bts[n] for n = 1:Nt];
bmax = [minimum(bt[n][:,1,:],dims = 2) for n = 1:Nt]
imax = [maximum(findall(bmax[n][:].<-0.05)) for n = 1:Nt]
Xfr = [x[imax[n]] for n =1:Nt]



lines!(ax,times,Xfr,label=labels[i])

end
axislegend()
save("Xfr.png", f) # output size = 800 x 600 pixels
