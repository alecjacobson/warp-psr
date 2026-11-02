"""Compare GPU evaluation of oracle coefficients, NOT GPU reconstruction."""
import argparse
import json
from pathlib import Path
import numpy as np
import warp as wp
from reference.adapter import load_field
from tools.make_fixture import read_output


def check(directory, device="cuda:0"):
    directory = Path(directory)
    manifest = json.loads((directory/"manifest.json").read_text())
    double = manifest["precision"] == "double"
    dtype, npdtype = (wp.float64,np.float64) if double else (wp.float32,np.float32)
    vec = wp.types.vector(length=3, dtype=dtype)
    with wp.ScopedDevice(device):
        field = load_field(directory/"oracle_instrumented.txt", dtype=dtype, device=device)
        q = wp.array(np.loadtxt(directory/"queries.txt").astype(npdtype), dtype=vec, device=device)
        values,gradients,valid = field.evaluate(q)
        assert valid.numpy().all()
        actual = np.column_stack([values.numpy(), gradients.numpy()])
        expected = read_output(directory/"oracle.txt")["q"]
        # Require unscaled cube-gradient agreement, stronger than any cell-width-scaled test.
        a,r = (1e-8,1e-6) if double else (1e-5,1e-4)
        np.testing.assert_allclose(actual[:,0],expected[:,0],atol=a,rtol=r)
        # Normalize model gradients to cube coordinates.
        linear = np.array(field.linear).reshape(3,3)
        normalized_actual = actual[:,1:] @ np.linalg.inv(linear)
        normalized_expected = expected[:,1:] @ np.linalg.inv(linear)
        np.testing.assert_allclose(normalized_actual, normalized_expected,atol=a,rtol=r)
        error = actual-expected
        report = {"test":"GPU evaluation of imported reference coefficients only", "device":device,
                  "precision":manifest["precision"], "queries":len(q), "coefficients":len(field.keys),
                  "max_abs_value_error":float(np.max(np.abs(error[:,0]))),
                  "rms_value_error":float(np.sqrt(np.mean(error[:,0]**2))),
                  "max_abs_model_gradient_error":float(np.max(np.abs(error[:,1:]))),
                  "gradient_gate":"unscaled cube gradients (stronger than cell-width-scaled budget)",
                  "reconstruction_compatibility":False}
        (directory/"field_report.json").write_text(json.dumps(report,indent=2)+"\n")
        np.save(directory/"warp_evaluation.npy",actual)
        print(json.dumps(report,indent=2))
        return report

if __name__ == "__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("directory")
    p.add_argument("--device",default="cuda:0")
    a=p.parse_args()
    wp.init()
    if a.device.startswith("cuda") and not wp.is_cuda_available():
        raise RuntimeError("CUDA validation requires an available GPU")
    check(a.directory,a.device)
