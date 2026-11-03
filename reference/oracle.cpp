// Offline validation only. Never linked or invoked by production reconstruction.
#include <fstream>
#include <iomanip>
#include <iostream>
#include <memory>
#include <chrono>
#include "Reconstructors.h"
using namespace PoissonRecon;
constexpr unsigned Sig = FEMDegreeAndBType<1, BOUNDARY_NEUMANN>::Signature;

template<class Real> struct Samples : Reconstructor::InputOrientedSampleStream<Real,3> {
    std::vector<Point<Real,3>> points, normals;
    size_t cursor=0;
    Samples(const char* file) {
        std::ifstream in(file); double x,y,z,nx,ny,nz;
        while(in>>x>>y>>z>>nx>>ny>>nz) {
            points.emplace_back(Real(x),Real(y),Real(z));
            normals.emplace_back(Real(nx),Real(ny),Real(nz));
        }
        if(points.empty()) throw std::runtime_error("Empty or unreadable input");
    }
    void reset() override { cursor=0; }
    bool read(Point<Real,3>& p, Point<Real,3>& n) override {
        if(cursor==points.size()) return false;
        p=points[cursor]; n=normals[cursor++]; return true;
    }
};

template<class Real> int run(int argc, char** argv) {
    // input queries output depth full-depth point-weight exact iterations
    Samples<Real> samples(argv[1]);
    Reconstructor::Poisson::SolutionParameters<Real> p;
    p.depth=std::stoi(argv[4]); p.fullDepth=std::stoi(argv[5]);
    p.baseDepth=p.fullDepth; p.solveDepth=p.depth; p.kernelDepth=p.depth>2?p.depth-2:0;
    p.pointWeight=Real(std::stod(argv[6])); p.exactInterpolation=std::stoi(argv[7]);
    p.iters=std::stoi(argv[8]); p.scale=Real(1.1); p.samplesPerNode=Real(1.5);
    p.cgSolverAccuracy=Real(1e-3); p.baseVCycles=1;
    if(argc==17) {
        p.baseDepth=std::stoi(argv[10]); p.solveDepth=std::stoi(argv[11]); p.kernelDepth=std::stoi(argv[12]);
        p.scale=Real(std::stod(argv[13])); p.samplesPerNode=Real(std::stod(argv[14]));
        p.cgSolverAccuracy=Real(std::stod(argv[15])); p.baseVCycles=std::stoi(argv[16]);
    }
    using Solver=Reconstructor::Poisson::Solver<Real,3,IsotropicUIntPack<3,Sig>>;
    auto start=std::chrono::steady_clock::now();
    std::unique_ptr<Reconstructor::Implicit<Real,3,IsotropicUIntPack<3,Sig>>> result(Solver::Solve(samples,p));
    auto stop=std::chrono::steady_clock::now();
    if(const char* file=std::getenv("WARP_PSR_TIMING_FILE")) {
        std::ofstream timing(file);
        timing<<std::setprecision(17)<<std::chrono::duration<double>(stop-start).count()<<'\n';
    }
    std::ofstream out(argv[3]); out<<std::setprecision(17);
    out<<"parallel_async "<<(ThreadPool::ParallelizationType==ThreadPool::ASYNC)<<'\n';
    out<<"threads "<<ThreadPool::NumThreads()<<'\n';
    out<<"schedule "<<int(ThreadPool::Schedule)<<'\n';
    out<<"chunk_size "<<ThreadPool::ChunkSize<<'\n';
    out<<"iso "<<result->isoValue<<'\n';
    out<<"transform";
    for(int i=0;i<4;i++) for(int j=0;j<4;j++) out<<' '<<result->unitCubeToModel(i,j);
    out<<'\n';
    auto eval=result->evaluator();
    std::ifstream queries(argv[2]); double x,y,z;
    while(queries>>x>>y>>z) {
        Point<Real,3> q{Real(x),Real(y),Real(z)};
        auto g=eval.grad(q);
        out<<"q "<<eval(q)<<' '<<g[0]<<' '<<g[1]<<' '<<g[2]<<'\n';
    }
    // Optional offline visualization export. Evaluate the unchanged solved field
    // at a regular model-space grid, without enormous ASCII query/gradient files.
    if(const char* spec=std::getenv("WARP_PSR_GRID_SPEC")) {
        const char* output=std::getenv("WARP_PSR_GRID_OUTPUT");
        if(!output) throw std::runtime_error("Grid export requires WARP_PSR_GRID_OUTPUT");
        std::ifstream gridSpec(spec);
        int resolution; double low[3], high[3];
        if(!(gridSpec>>resolution>>low[0]>>low[1]>>low[2]>>high[0]>>high[1]>>high[2]) || resolution<2)
            throw std::runtime_error("Invalid grid specification");
        std::ofstream binary(output,std::ios::binary);
        if(!binary) throw std::runtime_error("Cannot open grid output");
        std::vector<Real> row(resolution);
        for(int i=0;i<resolution;i++) for(int j=0;j<resolution;j++) {
            for(int k=0;k<resolution;k++) {
                int indices[3]={i,j,k}; Point<Real,3> q;
                for(int a=0;a<3;a++) q[a]=Real(low[a]+(high[a]-low[a])*(double(indices[a])/double(resolution-1)));
                row[k]=eval(q);
            }
            binary.write(reinterpret_cast<const char*>(row.data()),row.size()*sizeof(Real));
        }
        if(!binary) throw std::runtime_error("Grid output write failed");
    }
#ifdef EXPORT_TREE
    result->tree.tree().processNodes([&](const auto* n){
        int d,off[3]; result->tree.depthAndOffset(n,d,off);
        auto idx=n->nodeData.nodeIndex;
        out<<"node "<<d<<' '<<off[0]<<' '<<off[1]<<' '<<off[2]<<' '<<int(n->nodeData.flags)
           <<' '<<GetGhostFlag<3>(n)<<' '<<idx;
        out<<' '<<(idx>=0 && size_t(idx)<result->solution.size()?result->solution[idx]:Real(0))<<'\n';
    });
#endif
    return 0;
}
int main(int argc,char** argv) {
    if(argc!=10 && argc!=17) { std::cerr<<"input queries output depth full-depth point-weight exact iterations float|double [base-depth solve-depth kernel-depth scale samples-per-node cg-accuracy base-v-cycles]\n"; return 2; }
    ThreadPool::ParallelizationType=ThreadPool::NONE;
    Reconstructor::ProfilerMS=0;
    try {
        if(const char* setting=std::getenv("WARP_PSR_REFERENCE_PARALLELISM")) {
            std::string name(setting);
            if(name=="async") ThreadPool::ParallelizationType=ThreadPool::ASYNC;
            else if(name!="none") throw std::runtime_error("Reference parallelism must be none or async");
        }
        return std::string(argv[9])=="double" ? run<double>(argc,argv) : run<float>(argc,argv);
    }
    catch(const std::exception& e) { std::cerr<<e.what()<<'\n'; return 1; }
}
