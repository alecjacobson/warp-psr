if(const char* directory = std::getenv("WARP_PSR_ORACLE_DIR")) {
    std::ofstream dump(std::string(directory)+"/preprocess.txt");
    dump << std::setprecision(17);
    for(size_t i=0;i<samples->size();++i) {
        const auto& s=(*samples)[i];
        auto normal=(*sampleNormalAndAuxData)[i].template get<0>();
        int d,off[Dim]; implicit.tree.depthAndOffset(s.node,d,off);
        dump << "sample " << i << ' ' << d;
        for(int a=0;a<Dim;++a) dump << ' ' << off[a];
        for(int a=0;a<Dim;++a) dump << ' ' << s.sample.data[a];
        dump << ' ' << s.sample.weight;
        for(int a=0;a<Dim;++a) dump << ' ' << normal[a];
        dump << '\n';
    }
    implicit.tree.tree().processNodes([&](const auto* node) {
        int d,off[Dim]; implicit.tree.depthAndOffset(node,d,off);
        auto density=(*implicit.density)(node);
        if(density) {
            dump << "density " << d;
            for(int a=0;a<Dim;++a) dump << ' ' << off[a];
            dump << ' ' << *density << '\n';
        }
    });
}
