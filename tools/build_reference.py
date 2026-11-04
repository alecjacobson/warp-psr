"""Build a pinned, offline oracle; no reference code enters the Python package."""
import json
import hashlib
import os
from pathlib import Path
import shutil
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0,str(ROOT))
from tools.instrument_solver import instrument
BUILD = ROOT / ".build"
PINS = json.loads((ROOT / "reference/pins.json").read_text())
FLAGS = ["-std=c++17", "-O2", "-DNDEBUG", "-pthread", "-Wno-invalid offsetof".replace(" ", "-")]


def build():
    BUILD.mkdir(exist_ok=True)
    source = BUILD / "PoissonRecon"
    if not source.exists():
        subprocess.run(["git", "clone", "https://github.com/mkazhdan/PoissonRecon.git", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", PINS["poissonrecon"]], check=True)
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"]):
        raise RuntimeError("Reference checkout must be clean")
    instrumented = BUILD / "instrumented"
    shutil.copytree(source / "Src", instrumented, dirs_exist_ok=True)
    header = instrumented / "Reconstructors.h"
    text = header.read_text()
    # Insert read-only dumps immediately before temporary data is destroyed.
    marker = "\t\t\t\t// Free up the normal info"
    assert text.count(marker) == 2
    hook = r'''
                if(const char* directory = std::getenv("WARP_PSR_ORACLE_DIR")) {
                    std::ofstream dump(std::string(directory)+"/stages.txt");
                    dump << std::setprecision(17);
                    dump << "effective " << params.depth << ' ' << params.fullDepth << ' ' << params.baseDepth << ' ' << params.solveDepth << ' ' << params.kernelDepth << ' ' << params.scale << ' ' << params.samplesPerNode << ' ' << params.pointWeight << ' ' << params.exactInterpolation << ' ' << params.iters << ' ' << params.cgSolverAccuracy << '\n';
                    auto dw=pointDepthAndWeight.value();
                    dump << "density_scale " << dw[0] << ' ' << dw[1] << '\n';
                    for(const auto& s : *samples) {
                        int d,off[Dim]; implicit.tree.depthAndOffset(s.node,d,off);
                        dump << "sample " << d;
                        for(int a=0;a<Dim;a++) dump << ' ' << off[a];
                        for(int a=0;a<Dim;a++) dump << ' ' << s.sample.data[a];
                        dump << ' ' << s.sample.weight << '\n';
                    }
                    implicit.tree.tree().processNodes([&](const auto* n) {
                        int d,off[Dim]; implicit.tree.depthAndOffset(n,d,off);
                        auto normal=(*normalInfo)(n);
                        if(normal) {
                            dump << "normal " << d;
                            for(int a=0;a<Dim;a++) dump << ' ' << off[a];
                            for(int a=0;a<Dim;a++) dump << ' ' << (*normal)[a];
                            dump << '\n';
                        }
                        auto density=(*implicit.density)(n);
                        if(density) {
                            dump << "density " << d;
                            for(int a=0;a<Dim;a++) dump << ' ' << off[a];
                            dump << ' ' << *density << '\n';
                        }
                        auto idx=n->nodeData.nodeIndex;
                        if(idx>=0 && size_t(idx)<constraints.size()) {
                            dump << "load " << d;
                            for(int a=0;a<Dim;a++) dump << ' ' << off[a];
                            dump << ' ' << constraints[idx] << '\n';
                        }
                    });
                }
'''
    sample_marker = "\t\t\t\tdelete sampleNormalAndAuxData;"
    assert text.count(sample_marker) == 2
    text = text.replace(sample_marker, (ROOT/"reference/preprocess_export.inc").read_text()+sample_marker, 1)
    # Additional exports operate on the exact reference data before solving.
    solve_marker = "\t\t\t\t// Solve the linear system"
    assert text.count(solve_marker) == 2
    text = text.replace(solve_marker, (ROOT/"reference/operator_export.inc").read_text()+solve_marker, 1)
    weighted = instrumented / "FEMTree.WeightedSamples.inl"
    weighted_text = weighted.read_text()
    splat_marker = "\tSplat( temp , (Real)dx );"
    assert weighted_text.count(splat_marker) == 2
    weighted_text = weighted_text.replace(splat_marker, (ROOT/"reference/splat_export.inc").read_text()+splat_marker, 1)
    weighted.write_text("#include <fstream>\n#include <iomanip>\n"+weighted_text)
    initialize = instrumented / "FEMTree.Initialize.inl"
    initialize_text = initialize.read_text()
    input_marker = "\t// Add the point data\n"
    assert initialize_text.count(input_marker) == 1
    initialize_text = initialize_text.replace(input_marker,
        (ROOT/"reference/input_export.inc").read_text()+input_marker)
    replacements = {
        "if( !IsValid( p , d... ) ){ badDataCount++ ; continue; }":
        "if( !IsValid( p , d... ) ){ traceInput(1,nullptr); badDataCount++ ; continue; }",
        "if( !leaf ){ outOfBoundPoints++ ; continue; }":
        "if( !leaf ){ traceInput(2,nullptr); outOfBoundPoints++ ; continue; }",
        "if( Process( *leaf , p , d ... ) ) pointCount++;":
        "if( Process( *leaf , p , d ... ) ){ traceInput(0,leaf); pointCount++; } else traceInput(3,leaf);",
    }
    for original, replacement in replacements.items():
        assert initialize_text.count(original) == 1
        initialize_text = initialize_text.replace(original, replacement)
    initialize.write_text("#include <fstream>\n#include <iomanip>\n"+initialize_text)
    header.write_text("#include <fstream>\n#include <iomanip>\n" + text.replace(marker, hook + marker, 1))
    instrument(instrumented)
    solver_sources = ''.join((instrumented/name).read_text() for name in
                             ['FEMTree.System.inl','SparseMatrixInterface.inl'])
    solver_sources += (ROOT/'reference/solver_trace.h').read_text()
    commands = []
    for name, cpp, inc, extra in [
        ("oracle", "oracle.cpp", source / "Src", []),
        ("oracle_instrumented", "oracle.cpp", instrumented, ["-DEXPORT_TREE"]),
        ("oracle_openmp", "oracle.cpp", source / "Src", ["-fopenmp"]),
        ("oracle_instrumented_openmp", "oracle.cpp", instrumented, ["-fopenmp", "-DEXPORT_TREE"]),
        ("basis", "basis.cpp", source / "Src", []),
        ("PoissonReconCLI", "PoissonRecon.cpp", source / "Src", ["-DFAST_COMPILE"]),
    ]:
        cpp_path = inc / cpp if name == "PoissonReconCLI" else ROOT / "reference" / cpp
        command = ["g++", *FLAGS, *extra, "-I"+str(inc), "-I"+str(source), str(cpp_path), "-lz", "-lpng", "-ljpeg", "-o", str(BUILD / name)]
        print("Building", name, flush=True)
        digest = hashlib.sha256((str(command)+cpp_path.read_text()+header.read_text()+weighted.read_text()+initialize.read_text()+solver_sources).encode()).hexdigest()
        stamp = BUILD / (name + ".sha256")
        if not (BUILD/name).exists() or not stamp.exists() or stamp.read_text() != digest:
            subprocess.run(command, check=True)
            stamp.write_text(digest)
        commands.append(command)
    (BUILD / "build.json").write_text(json.dumps({"pins": PINS, "commands": commands,
        "compiler": subprocess.check_output(["g++", "--version"], text=True).splitlines()[0]}, indent=2))

if __name__ == "__main__":
    build()
