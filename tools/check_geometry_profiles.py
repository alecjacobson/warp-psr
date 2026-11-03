"""Run varied sample distributions with named volume-query coverage."""
import json
import subprocess
import sys
from tools.make_fixture import ROOT


GEOMETRY_CASES = {
    'plane':['--shape','plane'],
    'sheets':['--shape','sheets'],
    'cluster':['--shape','cluster'],
    'disconnected':['--shape','disconnected'],
    'transformed':['--shape','sphere','--input-scale','3.7','--translation','2','-3','1'],
}


def main():
    reports=[]
    for precision in ('float','double'):
        for case,options in GEOMETRY_CASES.items():
            name=f'geometry-{case}-{precision}'
            directory=ROOT/'.build/fixtures'/name
            directory.mkdir(parents=True,exist_ok=True)
            with (directory/'validation.log').open('w') as log:
                commands=[['tools/make_fixture.py','--name',name,'--count','1024','--depth','5',
                           '--full-depth','2','--grid','12','--precision',precision,*options],
                          ['-m','tools.check_field',str(directory)],
                          ['-m','tools.check_reconstruction',str(directory)]]
                status=0
                for command in commands:
                    status=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                    if status:break
            entry={'fixture':name,'passed':status==0}
            if not status:
                entry['manifest']=json.loads((directory/'manifest.json').read_text())
                entry['reconstruction']=json.loads((directory/'reconstruction_report.json').read_text())
            reports.append(entry)
            print(name,'PASS' if not status else f'FAIL (see {directory}/validation.log)',flush=True)
    report={'scope':'Float32/float64 independent reconstruction over varied geometry and named query families',
            'all_passed':all(r['passed'] for r in reports),'profiles':reports}
    (ROOT/'.build/geometry-validation.json').write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
