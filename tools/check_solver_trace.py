"""Independently replay small serial-reference solver traces in input precision."""
import argparse
import hashlib
import json
from pathlib import Path
import numpy as np
from tools.make_fixture import read_output


def events(path):
    current=None
    with path.open() as source:
        assert source.readline().strip()=='format 1'
        for line in source:
            parts=line.split()
            if parts[0]=='event':
                if current is not None:yield current
                current={'sequence':int(parts[1]),'kind':parts[2],'args':parts[3:],'data':{}}
            else:
                assert current is not None
                current['data'].setdefault(parts[0],[]).append(parts[1:])
    if current is not None:yield current


def check(directory):
    manifest=json.loads((directory/'manifest.json').read_text())
    path=directory/'solve_trace.txt'
    digest=hashlib.sha256(path.read_bytes()).hexdigest()
    assert digest==manifest['reference_file_sha256']['solve_trace.txt']
    assert manifest['reference_parallelism']=='ThreadPool::NONE (serial)'
    scalar=np.float64 if manifest['precision']=='double' else np.float32
    original=read_output(directory/'oracle.txt')
    instrumented=read_output(directory/'oracle_instrumented.txt')
    for key in ['q','iso','transform']:
        np.testing.assert_array_equal(original[key],instrumented[key])
    gs_count=0
    cg_count=0
    stops=[]
    levels=[]
    final_levels={}
    phases=set()
    root=None
    cg_state=None
    ended=False

    def vector(event,name):
        rows=event['data'][name]
        assert len(rows)==1
        return np.array(rows[0],dtype=scalar)

    def matrix(event):
        result=[]
        for i,row in enumerate(event['data'].get('row',[])):
            assert int(row[0])==i and len(row)%2==1
            result.append([(int(row[j]),scalar(row[j+1])) for j in range(1,len(row),2)])
        return result

    def dot(a,b):
        total=scalar(0)
        for x,y in zip(a,b):total=scalar(total+scalar(x*y))
        return total

    def matvec(m,x,dc):
        result=np.zeros(len(m),dtype=scalar)
        for i,row in enumerate(m):
            for j,value in row:result[i]=scalar(result[i]+scalar(x[j]*value))
        if dc:
            average=scalar(0)
            for value in x:average=scalar(average+value)
            average=scalar(average/scalar(len(m)))
            result+=average
        return result

    for sequence,event in enumerate(events(path)):
        assert event['sequence']==sequence
        kind,args=event['kind'],event['args']
        if kind=='gs':
            depth,phase,iteration,offset,forward,count=args
            count=int(count)
            m=matrix(event)
            assert len(m)==count
            halo=event['data']['halo'][0]
            lo,hi=map(int,halo[:2])
            x=np.array(halo[2:],dtype=scalar)
            assert len(x)==hi-lo
            np.testing.assert_array_equal(x[-lo:count-lo],vector(event,'before'))
            b,diagonal=vector(event,'rhs'),vector(event,'diagonal')
            colors=event['data'].get('color',[])
            assert [int(c[0]) for c in colors]==list(range(len(colors)))
            if not int(forward):colors=colors[::-1]
            for color in colors:
                for row in map(int,color[1:]):
                    residual=b[row]
                    for column,value in m[row]:residual=scalar(residual-scalar(x[column-lo]*value))
                    x[row-lo]=scalar(x[row-lo]+scalar(residual*diagonal[row]))
            np.testing.assert_array_equal(x[-lo:count-lo],vector(event,'after'),
                                          err_msg=f'GS event {sequence}, depth {depth}, phase {phase}')
            gs_count+=1
            phases.add(phase)
        elif kind=='root':
            root=(matrix(event),bool(int(args[3])))
        elif kind=='cg_initial':
            assert root is not None
            dim,limit=int(args[0]),int(args[1])
            eps,delta=float(args[2]),scalar(args[3])
            x=vector(event,'before')
            b=vector(event,'rhs')
            residual=b-matvec(root[0],x,root[1])
            np.testing.assert_array_equal(residual,vector(event,'residual'))
            np.testing.assert_array_equal(dot(residual,residual),delta)
            cg_state=dict(x=x,r=residual,d=residual.copy(),delta=delta,initial=delta,
                          eps=eps,limit=limit,iteration=0)
            assert len(x)==dim
            cg_count+=1
        elif kind=='cg_iteration':
            assert cg_state is not None
            state=cg_state
            iteration=int(args[0]);assert iteration==state['iteration']+1
            q=matvec(root[0],state['d'],root[1])
            denominator=dot(state['d'],q)
            alpha=scalar(state['delta']/denominator)
            old=state['delta']
            state['x']+=state['d']*alpha
            if iteration%50==0:
                state['r']=b-matvec(root[0],state['x'],root[1])
                # Preserve the pinned reference's second update on this branch.
                state['x']+=state['d']*alpha
            else:state['r']-=q*alpha
            delta=dot(state['r'],state['r'])
            beta=scalar(delta/old)
            state['d']=state['r']+state['d']*beta
            np.testing.assert_array_equal(np.array([old,delta,denominator,alpha,beta],dtype=scalar),
                                          np.array(args[1:6],dtype=scalar))
            np.testing.assert_array_equal(state['x'],vector(event,'after'))
            np.testing.assert_array_equal(state['r'],vector(event,'residual'))
            np.testing.assert_array_equal(state['d'],vector(event,'direction'))
            state.update(delta=delta,iteration=iteration)
        elif kind=='cg_stop':
            reason,iteration,delta,threshold=args
            state=cg_state
            assert int(iteration)==state['iteration']
            np.testing.assert_array_equal(scalar(delta),state['delta'])
            expected_threshold=state['eps'] if reason=='initial' else state['eps']*float(state['initial'])
            assert float(threshold)==expected_threshold
            if reason in ['initial','tolerance']:assert float(delta)<=float(threshold)
            elif reason=='limit':assert int(iteration)==state['limit'] and float(delta)>float(threshold)
            else:assert reason=='breakdown'
            stops.append(dict(reason=reason,iterations=int(iteration),residual_squared=float(delta),threshold_squared=float(threshold)))
            cg_state=None
        elif kind=='level':
            levels.append(dict(depth=int(args[0]),phase=args[1],iterations=int(args[3])))
            final_levels[int(args[0])]=(int(args[4]),vector(event,'after'))
        elif kind=='end':ended=True
        elif kind not in ['solve','cg_breakdown']:
            raise ValueError(f'Unknown trace event {kind}')
    assert ended and gs_count and cg_count and len(stops)==cg_count and levels
    nodes=instrumented['node']
    for depth,(offset,coefficients) in final_levels.items():
        selected=nodes[nodes[:,0]==depth]
        selected=selected[(selected[:,6]>=offset)&(selected[:,6]<offset+len(coefficients))]
        indices=selected[:,6].astype(np.int64)-offset
        np.testing.assert_array_equal(coefficients[indices],selected[:,7].astype(scalar))
    return dict(fixture=directory.name,trace_sha256=digest,precision=manifest['precision'],
                instrumentation_max_abs_error=0.0,replay='bitwise exact serial input-precision arithmetic',
                gs_updates=gs_count,gs_phases=sorted(phases),cg_solves=cg_count,cg_stops=stops,levels=levels)


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('fixture',type=Path)
    parser.add_argument('--output',type=Path)
    args=parser.parse_args()
    report=check(args.fixture)
    if args.output:args.output.write_text(json.dumps(report,indent=2)+'\n')
    print(json.dumps(report,indent=2))
