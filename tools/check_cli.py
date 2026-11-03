"""Compare the actual official CLI grid with the unchanged library driver."""
import argparse
import json
from pathlib import Path
import subprocess
import numpy as np
from tools.make_fixture import ROOT, read_output, oracle_parameters


def check(directory):
    directory=Path(directory).resolve()
    m=json.loads((directory/"manifest.json").read_text())
    p=m["parameters"]
    if m["precision"]!="float":
        raise ValueError("Official FAST_COMPILE CLI uses float32")
    command=[str(ROOT/".build/PoissonReconCLI"),"--in",str(directory/"input.pts"),"--grid",str(directory/"official.grid"),
             "--out",str(directory/"official.ply"),"--depth",str(p["depth"]),"--fullDepth",str(p["full_depth"]),
             "--baseDepth",str(p["base_depth"]),"--solveDepth",str(p["solve_depth"]),"--kernelDepth",str(p["kernel_depth"]),
             "--pointWeight",str(p["weight"]),"--scale",str(p["scale"]),"--samplesPerNode",str(p["samples_per_node"]),"--iters",str(p["iters"]),
             "--cgAccuracy",str(p["cg_accuracy"]),"--baseVCycles",str(p["base_v_cycles"]),"--parallel","1","--alignDir","0","--verbose"]
    if p["exact"]:
        command += ["--exact"]
    with (directory/"official.log").open("w") as f:
        subprocess.run(command,stdout=f,stderr=subprocess.STDOUT,check=True)
    with (directory/"official.grid").open("rb") as f:
        assert f.readline()==b"G3\n"
        assert f.readline()==b"1 FLOAT\n"
        shape=tuple(map(int,f.readline().split()))
        for _ in range(4):
            f.readline() # Header transform rounded to 6 decimals; use full-precision oracle transform.
        grid=np.frombuffer(f.read(),dtype=np.float32)
    assert len(grid)==np.prod(shape)
    axes=[(np.arange(n)+.5)/n for n in shape]
    unit=np.stack(np.meshgrid(*axes,indexing="ij"),axis=-1).reshape(-1,3,order="F")
    transform=read_output(directory/"oracle.txt")["transform"].reshape(4,4)
    queries=(np.c_[unit,np.ones(len(unit))]@transform)[:,:3].astype(np.float32)
    np.savetxt(directory/"cli_queries.txt",queries,fmt="%.17g")
    subprocess.run([str(ROOT/".build/oracle"),str(directory/"input.pts"),str(directory/"cli_queries.txt"),str(directory/"cli_oracle.txt"),
                    *oracle_parameters(p)],check=True)
    values=read_output(directory/"cli_oracle.txt")["q"][:,0]
    np.testing.assert_allclose(values,grid,atol=1e-5,rtol=1e-4)
    report={"test":"official CLI grid versus unchanged library oracle", "grid_values":len(grid),
            "max_abs_value_error":float(np.max(np.abs(values-grid))),"command":command}
    (directory/"cli_report.json").write_text(json.dumps(report,indent=2)+"\n")
    print(json.dumps(report,indent=2))

if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("directory")
    check(p.parse_args().directory)
