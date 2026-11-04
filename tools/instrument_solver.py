"""Read-only solve trace hooks, applied only to a copied pinned reference tree."""
from pathlib import Path


def replace_once(text,old,new):
    assert text.count(old)==1, (old,text.count(old))
    return text.replace(old,new)


def instrument(directory: Path):
    path=directory/'FEMTree.System.inl'
    text=path.read_text()
    marker='\tPointer( T ) _solution = solution();'
    text=replace_once(text,marker,marker+'''
    WarpPSRSolveTrace::Session traceSession;
    WarpPSRSolveTrace::event("solve",_baseDepth,minSolveDepth,maxSolveDepth,solverInfo.baseVCycles,solverInfo.vCycles,solverInfo.cgAccuracy,solution.size());
    WarpPSRSolveTrace::vector("initial",_solution,solution.size());
    WarpPSRSolveTrace::vector("constraints",constraints(),constraints.size());
''')
    for forward,phase in [('true','restrict'),('false','prolong')]:
        original=f'for( int i=0 ; i<iters ; i++ ) M[d].gsIteration( multiColorIndices[d] , D[d] , __B , X[d] , {forward} , true );'
        new=f'''for( int i=0 ; i<iters ; i++ ) {{
            WarpPSRSolveTrace::gsBegin(d,"{phase}",i,nodesBegin(d),{forward},M[d],D[d],__B,X[d],multiColorIndices[d]);
            M[d].gsIteration( multiColorIndices[d] , D[d] , __B , X[d] , {forward} , true );
            WarpPSRSolveTrace::vector("after",X[d],M[d].rows());
        }}'''
        text=replace_once(text,original,new)
    original='\t\t\tSolveCG< SPDFunctor , T , Real >( SPDFunctor( M[d] , addDCTerm ) , M[d].rows() , ( ConstPointer( T ) )__B , nonZeroRows , X[d] , Real( cgAccuracy ) , Dot );'
    text=replace_once(text,original,'''
            WarpPSRSolveTrace::event("root",d,v,nodesBegin(d),int(addDCTerm),M[d].rows());
            WarpPSRSolveTrace::matrix(M[d]);
'''+original)
    original='\t\t\t\t_M[_b].gsIteration( mcIndices[__b] , ( ConstPointer( Real ) )_D[_b] , B , X , coarseToFine , true );'
    text=replace_once(text,original,'''
                WarpPSRSolveTrace::gsBegin(depth,"slice",b,_sNodesBegin(depth,BlockFirst(b)),coarseToFine,_M[_b],_D[_b],B,X,mcIndices[__b]);
'''+original+'''
                WarpPSRSolveTrace::vector("after",X,_M[_b].rows());
''')
    # These snapshots are taken before the existing restriction/prolongation
    # updates. No residual-norm setting or matrix window is changed.
    for operation in ['Restriction','Prolongation']:
        indent='\t'*(4 if operation=='Restriction' else 3)
        marker=f'{indent}Update{operation}( d );\n{indent}sStats.constraintUpdateTime'
        assert text.count(marker)==1
        text=text.replace(marker,f'''            WarpPSRSolveTrace::event("level",d,"{operation.lower()}",v,actualIters,_sNodesBegin(d),_sNodesEnd(d)-_sNodesBegin(d));
            WarpPSRSolveTrace::vector("after",_solution+_sNodesBegin(d),_sNodesEnd(d)-_sNodesBegin(d));
'''+marker)
    path.write_text(text)

    path=directory/'SparseMatrixInterface.inl'
    text=path.read_text()
    start=text.index('template< class SPDFunctor , class T , typename Real , class TDotTFunctor > size_t SolveCG(')
    end=text.index('template< class SPDFunctor , class Preconditioner',start)
    body=text[start:end]
    body=replace_once(body,'\tdelta_0 = delta_new;', '''\tdelta_0 = delta_new;
    WarpPSRSolveTrace::event("cg_initial",dim,iters,eps,delta_0);
    WarpPSRSolveTrace::vector("rhs",b,dim);
    WarpPSRSolveTrace::vector("before",x,dim);
    WarpPSRSolveTrace::vector("residual",r,dim);''')
    body=replace_once(body,'\t\treturn 0;', '''        WarpPSRSolveTrace::event("cg_stop","initial",0,delta_new,eps);
\t\treturn 0;''')
    body=replace_once(body,'\t\tif( !dDotQ ) break;', '''\t\tif( !dDotQ ) {
            WarpPSRSolveTrace::event("cg_breakdown",ii,delta_new,dDotQ);
            break;
        }''')
    marker='\t\tThreadPool::ParallelFor( 0 , dim , [&]( unsigned int , size_t i ){ d[i] = r[i] + (T)( d[i] * beta ); } );'
    body=replace_once(body,marker,marker+'''
        WarpPSRSolveTrace::event("cg_iteration",ii+1,delta_old,delta_new,dDotQ,alpha,beta,eps*delta_0);
        WarpPSRSolveTrace::vector("after",x,dim);
        WarpPSRSolveTrace::vector("residual",r,dim);
        WarpPSRSolveTrace::vector("direction",d,dim);''')
    body=replace_once(body,'\treturn ii;', '''    WarpPSRSolveTrace::event("cg_stop",delta_new<=eps*delta_0 ? "tolerance" : (ii==iters ? "limit" : "breakdown"),ii,delta_new,eps*delta_0);
\treturn ii;''')
    path.write_text(text[:start]+body+text[end:])
