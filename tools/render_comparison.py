"""Render actual sampled fields; the GPU panel uses imported oracle coefficients."""
import argparse
import json
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LightSource
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
import numpy as np
from skimage.measure import marching_cubes
from tools.make_fixture import read_output


def render(directory, output):
    directory=Path(directory)
    manifest=json.loads((directory/"manifest.json").read_text())
    report=json.loads((directory/"field_report.json").read_text())
    n=manifest["grid"]["resolution"]
    ref=read_output(directory/"oracle.txt")
    iso=ref["iso"][0,0]
    a=ref["q"][:n**3,0].reshape(n,n,n)
    b=np.load(directory/"warp_evaluation.npy")[:n**3,0].reshape(n,n,n)
    plt.rcParams.update({"font.family":"DejaVu Sans","text.color":"#20314b","axes.titleweight":"bold"})
    fig=plt.figure(figsize=(13,5.5),facecolor="#f4f6fa")
    for k,(grid,title) in enumerate([(a,"Official PoissonRecon"),(b,"Warp GPU evaluation")]):
        ax=fig.add_subplot(1,3,k+1,projection="3d",computed_zorder=False)
        ax.set_facecolor("#f4f6fa")
        vertices,faces,_,_=marching_cubes(grid,iso)
        # Query grids are in model coordinates.
        vertices=vertices/(n-1)-.5
        tri=vertices[faces]
        normals=np.cross(tri[:,1]-tri[:,0],tri[:,2]-tri[:,0])
        normals/=np.maximum(np.linalg.norm(normals,axis=1)[:,None],1e-15)
        light=np.array([-.3,-.4,1.]);light/=np.linalg.norm(light)
        shade=.35+.65*np.maximum(normals@light,0)
        color=np.array([.12,.56,.65]) if k==0 else np.array([.25,.38,.77])
        colors=np.clip(color[None]*shade[:,None]+.08,0,1)
        mesh=Poly3DCollection(tri,facecolor=colors,edgecolor="none",linewidth=0,rasterized=True)
        ax.add_collection3d(mesh)
        ax.set(xlim=(-.5,.5),ylim=(-.5,.5),zlim=(-.5,.5))
        ax.set_box_aspect((1,1,1));ax.view_init(26,-50);ax.set_axis_off()
        ax.set_title(title,pad=0)
        ax.text2D(.5,-.02,"Reference solve" if k==0 else "Imported reference coefficients",transform=ax.transAxes,ha="center",fontsize=10)
    ax=fig.add_subplot(1,3,3)
    error=np.abs(b-a)
    im=ax.imshow(error[:,n//2,:].T,origin="lower",cmap="magma",vmin=0,vmax=max(error.max(),1e-12),extent=[0,1,0,1])
    ax.set_title("Volume field difference")
    ax.set_xlabel("Normalized cube x");ax.set_ylabel("Normalized cube z")
    fig.colorbar(im,ax=ax,fraction=.045,pad=.03,label="Absolute field error")
    fig.suptitle("Adaptive degree-1 field evaluation • torus",fontsize=18,y=.98)
    fig.text(.5,.07,f"{report['queries']:,} queries  ·  {report['coefficients']:,} coefficients  ·  max |Δf| = {report['max_abs_value_error']:.2e}",ha="center",fontsize=11)
    fig.text(.5,.025,"Evaluation validation only — the Warp reconstruction builder and solver are not complete.",ha="center",fontsize=10,color="#596579")
    fig.subplots_adjust(left=.025,right=.95,bottom=.2,top=.85,wspace=.25)
    output=Path(output);output.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(output,dpi=180,facecolor=fig.get_facecolor())
    plt.close(fig)

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory");p.add_argument("--output",default="docs/results/torus-evaluation.png")
    a=p.parse_args();render(a.directory,a.output)
