// Direct calls to the pinned reference's 1D basis and exact polynomial integrals.
#include <iostream>
#include <iomanip>
#include "Reconstructors.h"
using namespace PoissonRecon;
constexpr unsigned Sig=FEMDegreeAndBType<1,BOUNDARY_NEUMANN>::Signature;
using B=BSplineEvaluationData<Sig>;
using I=BSplineIntegrationData<Sig,Sig>;
constexpr unsigned NormalSig=FEMDegreeAndBType<2,BOUNDARY_DIRICHLET>::Signature;
using N=BSplineIntegrationData<Sig,NormalSig>;
int main(int argc,char** argv) {
    if(argc!=2 && argc!=3) return 2;
    int d=std::stoi(argv[1]), n=1<<d;
    std::cout<<std::setprecision(17);
    if(argc==3) {
        if(std::string(argv[2])!="--normal") return 2;
        for(int nd=std::max(0,d-1);nd<=d+1;++nd)
            for(int i=0;i<=n;++i) for(int j=0;j<(1<<nd);++j)
                std::cout<<"n "<<d<<' '<<nd<<' '<<i<<' '<<j<<' '
                         <<N::Dot<0,0>(d,i,nd,j)<<' '<<N::Dot<1,0>(d,i,nd,j)<<'\n';
        return 0;
    }
    if(d<1) return 2;
    for(int i=0;i<=n;i++) for(int k=0;k<n*3;k++) {
        double x=(k+0.37)/(3*n);
        std::cout<<"b "<<i<<' '<<x<<' '<<B::Value(d,i,x,0)<<' '<<B::Value(d,i,x,1)<<'\n';
    }
    for(int i=0;i<=n;i++) for(int j=0;j<=n;j++)
        std::cout<<"a "<<i<<' '<<j<<' '<<I::Dot<0,0>(d,i,d,j)<<' '<<I::Dot<1,1>(d,i,d,j)<<'\n';
    for(int i=0;i<=(1<<(d-1));i++) for(int j=0;j<=n;j++)
        std::cout<<"c "<<i<<' '<<j<<' '<<I::Dot<0,0>(d-1,i,d,j)<<' '<<I::Dot<1,1>(d-1,i,d,j)<<'\n';
}
