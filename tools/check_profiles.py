"""Check independently varied public settings against explicit official runs."""
import json
import subprocess
import sys
from tools.make_fixture import ROOT


PROFILE_CASES={
    'base':['--base-depth','1'],
    'solve':['--solve-depth','3'],
    'cluster-solve':['--shape','cluster','--count','1024','--depth','5','--solve-depth','4'],
    'kernel':['--kernel-depth','1'],
    'scale':['--scale','1.25'],
    'density':['--samples-per-node','3.0'],
    'weight':['--weight','0.25'],
    'iterations':['--iters','16'],
    'cycles':['--base-v-cycles','2'],
    'cg':['--cg-accuracy','0.00001'],
    'strict':['--iters','32','--base-v-cycles','2','--cg-accuracy','0.000001'],
}

ZERO_BASE_CASES={
    'base-zero':['--base-depth','0'],
    'base-zero-psr':['--base-depth','0','--weight','0'],
    'base-zero-exact':['--base-depth','0','--exact'],
    'full-zero':['--base-depth','0','--full-depth','0'],
    'base-zero-cycles':['--base-depth','0','--base-v-cycles','3','--cg-accuracy','0.000001'],
    'root-solve':['--base-depth','0','--full-depth','0','--solve-depth','0'],
    'depth-one':['--depth','1','--full-depth','0','--base-depth','0','--kernel-depth','0'],
}
for precision in ('float','double'):
    for case,options in ZERO_BASE_CASES.items():
        PROFILE_CASES[f'{case}-{precision}']=[*options,'--precision',precision,'--operators']


def main():
    reports=[]
    for case,options in PROFILE_CASES.items():
        name='profile-'+case
        directory=ROOT/'.build/fixtures'/name
        directory.mkdir(parents=True,exist_ok=True)
        with (directory/'validation.log').open('w') as log:
            commands=[['tools/make_fixture.py','--name',name,'--grid','12',*options],
                      ['-m','tools.check_reconstruction',str(directory)]]
            status=0
            for command in commands:
                status=subprocess.run([sys.executable,*command],cwd=ROOT,stdout=log,stderr=subprocess.STDOUT).returncode
                if status:break
        entry={'fixture':name,'options':options,'passed':status==0}
        if not status:
            entry['manifest']=json.loads((directory/'manifest.json').read_text())
            entry['reconstruction']=json.loads((directory/'reconstruction_report.json').read_text())
        reports.append(entry)
        print(name,'PASS' if not status else f'FAIL (see {directory}/validation.log)',flush=True)
    report={'scope':'Run compatibility with independently varied explicit parameters; strict is a stronger iteration schedule, not a convergence proof',
            'all_passed':all(r['passed'] for r in reports),'profiles':reports}
    output=ROOT/'.build/profile-validation.json'
    output.write_text(json.dumps(report,indent=2)+'\n')
    return 0 if report['all_passed'] else 1


if __name__=='__main__':
    raise SystemExit(main())
