if(const char* directory = std::getenv("WARP_PSR_ORACLE_DIR")) {
    static std::ofstream dump(std::string(directory)+"/splats.txt");
    dump << std::setprecision(17);
    dump << "splat";
    for(int a=0;a<Dim;++a) dump << ' ' << position[a];
    dump << ' ' << rawDepth << ' ' << weight << ' ' << topDepth << ' ' << dx;
    for(int a=0;a<Dim;++a) dump << ' ' << v[a];
    dump << '\n';
}
