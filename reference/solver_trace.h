// Read-only, opt-in diagnostics for the instrumented offline oracle.
#pragma once
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <stdexcept>
#include <string>

namespace WarpPSRSolveTrace {
inline std::ofstream stream;
inline size_t sequence=0;

struct Session {
    Session() {
        if(const char* path=std::getenv("WARP_PSR_SOLVE_TRACE")) {
            stream.open(path);
            if(!stream) throw std::runtime_error("Cannot open solver trace");
            stream<<std::setprecision(17)<<"format 1\n";
            sequence=0;
        }
    }
    ~Session() {
        if(stream.is_open()) {
            stream<<"event "<<sequence++<<" end\n";
            stream.close();
        }
    }
};

template<class... Args> void event(const char* name,Args... args) {
    if(!stream.is_open()) return;
    stream<<"event "<<sequence++<<' '<<name;
    ((stream<<' '<<args),...);
    stream<<'\n';
}

template<class Pointer> void vector(const char* name,Pointer p,size_t n) {
    if(!stream.is_open()) return;
    stream<<name;
    for(size_t i=0;i<n;++i) stream<<' '<<p[i];
    stream<<'\n';
}

template<class Matrix> void matrix(const Matrix& m) {
    if(!stream.is_open()) return;
    for(size_t i=0;i<m.rows();++i) {
        stream<<"row "<<i;
        for(size_t j=0;j<m.rowSize(i);++j) stream<<' '<<m[i][j].N<<' '<<m[i][j].Value;
        stream<<'\n';
    }
}

template<class Matrix,class Diagonal,class RHS,class Solution,class Colors>
void gsBegin(int depth,const char* phase,int iteration,long long offset,bool forward,
             const Matrix& m,Diagonal diagonal,RHS rhs,Solution solution,const Colors& colors) {
    if(!stream.is_open()) return;
    event("gs",depth,phase,iteration,offset,int(forward),m.rows());
    matrix(m);
    vector("diagonal",diagonal,m.rows());
    vector("rhs",rhs,m.rows());
    vector("before",solution,m.rows());
    long long lo=0,hi=static_cast<long long>(m.rows());
    for(size_t i=0;i<m.rows();++i) for(size_t j=0;j<m.rowSize(i);++j) {
        long long column=m[i][j].N;
        if(column<lo) lo=column;
        if(column>=hi) hi=column+1;
    }
    stream<<"halo "<<lo<<' '<<hi;
    for(long long i=lo;i<hi;++i) stream<<' '<<solution[i];
    stream<<'\n';
    for(size_t c=0;c<colors.size();++c) {
        stream<<"color "<<c;
        for(size_t i:colors[c]) stream<<' '<<i;
        stream<<'\n';
    }
}
}
