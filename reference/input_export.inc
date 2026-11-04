// Read-only instrumentation in FEMTreeInitializer::Initialize. The subtree
// extractor makes accepted leaf coordinates local to the input unit cube.
std::ofstream inputDump;
if(const char* directory = std::getenv("WARP_PSR_ORACLE_DIR")) {
    inputDump.open(std::string(directory)+"/inputs.txt");
    inputDump << std::setprecision(17);
}
size_t inputIndex = 0;
auto traceInput = [&]( int status , const FEMTreeNode* node ) {
    if(inputDump.is_open()) {
        typename FEMTree< Dim , Real >::LocalDepth depth = -1;
        typename FEMTree< Dim , Real >::LocalOffset offset = {};
        if(node) node->depthAndOffset(depth,offset);
        inputDump << "input " << inputIndex << ' ' << status << ' ' << depth;
        for(int a=0;a<Dim;++a) inputDump << ' ' << offset[a];
        inputDump << '\n';
    }
    ++inputIndex;
};
