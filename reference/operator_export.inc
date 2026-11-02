// Included only by the instrumented oracle immediately before solveSystem.
if(const char* directory = std::getenv("WARP_PSR_OPERATORS")) {
    std::ofstream dump(std::string(directory)+"/operators.txt");
    dump << std::setprecision(17);
    typename FEMIntegrator::template System<Sigs, IsotropicUIntPack<Dim,1>> op({0.,1.});
    for(int d=0; d<=int(params.solveDepth); ++d) {
        auto A=implicit.tree.systemMatrix(Sigs(),op,d,std::make_tuple(iInfo));
        auto begin=implicit.tree.nodesBegin(d);
        dump << "level " << d << ' ' << begin << ' ' << A.rows() << '\n';
        for(int i=0; i<int(A.rows()); ++i) {
            dump << "rhs " << d << ' ' << (i+begin) << ' ' << constraints[i+begin] << '\n';
            for(int j=0; j<int(A.rowSize(i)); ++j)
                dump << "A " << d << ' ' << (i+begin) << ' ' << (A[i][j].N+begin) << ' ' << A[i][j].Value << '\n';
        }
        if(d) {
            auto R=implicit.tree.downSampleMatrix(Sigs(),d);
            auto cb=implicit.tree.nodesBegin(d-1);
            for(int i=0; i<int(R.rows()); ++i) for(int j=0; j<int(R.rowSize(i)); ++j)
                dump << "R " << d << ' ' << (i+cb) << ' ' << (R[i][j].N+begin) << ' ' << R[i][j].Value << '\n';
        }
    }
}
