from pathlib import Path
import os, subprocess, time

def processes():
    result = {}
    for path in Path('/proc').iterdir():
        if not path.name.isdigit():
            continue
        try:
            stat = (path/'stat').read_text()
            end = stat.rindex(')')
            fields = stat[end+2:].split()
            result[int(path.name)] = {'name': stat[stat.index('(')+1:end], 'parent': int(fields[1]),
                                      'ticks': int(fields[11])+int(fields[12])}
        except (OSError, ValueError, IndexError):
            continue
    return result

def descendants(snapshot, seeds):
    allowed = set(seeds)
    while True:
        expanded = allowed | {pid for pid, entry in snapshot.items() if entry['parent'] in allowed}
        if expanded == allowed:
            return allowed
        allowed = expanded

def gpu_processes():
    result = subprocess.check_output(['nvidia-smi','--query-compute-apps=pid,process_name,used_memory',
                                      '--format=csv,noheader'], text=True)
    return [line for line in result.splitlines() if line.strip()]

def measured_process(command, log_path, *, root, env):
    # Schedule against observed load, never against the resulting timing.
    idle_checks=0; waiting=[]; previous_idle=processes(); last_idle=time.monotonic()
    ticks_per_second=os.sysconf('SC_CLK_TCK')
    while idle_checks < 3:
        time.sleep(1.)
        now_idle=time.monotonic(); snapshot_idle=processes()
        allowed_idle=descendants(snapshot_idle,[os.getpid()])
        active_idle=[]
        for pid,entry in snapshot_idle.items():
            if pid in allowed_idle or pid not in previous_idle:continue
            cores=(entry['ticks']-previous_idle[pid]['ticks'])/ticks_per_second/(now_idle-last_idle)
            if cores>.25:active_idle.append({'pid':pid,'name':entry['name'],'cpu_equivalents':cores})
        current_gpu=gpu_processes()
        if active_idle or current_gpu:
            idle_checks=0
            waiting.append({'cpu':active_idle,'gpu':current_gpu})
            if len(waiting)%30==1:print('WAIT FOR IDLE',log_path.name,active_idle,current_gpu,flush=True)
        else:idle_checks+=1
        previous_idle=snapshot_idle;last_idle=now_idle
    initial_gpu = gpu_processes()
    assert not initial_gpu, initial_gpu
    initial = previous = processes()
    start = previous_time = time.monotonic()
    observations, foreign = [], []
    ticks_per_second = os.sysconf('SC_CLK_TCK')
    with log_path.open('w') as log:
        process = subprocess.Popen(command, cwd=root, env=env, stdout=log, stderr=subprocess.STDOUT)
        while True:
            try:
                process.wait(timeout=1.)
            except subprocess.TimeoutExpired:
                pass
            now = time.monotonic()
            snapshot = processes()
            allowed = descendants(snapshot, [os.getpid(), process.pid])
            active = []
            for pid, entry in snapshot.items():
                if pid in allowed or pid not in previous:
                    continue
                cores = (entry['ticks']-previous[pid]['ticks'])/ticks_per_second/(now-previous_time)
                if cores > .25:
                    active.append({'pid':pid, 'name':entry['name'], 'cpu_equivalents':cores})
            gpu = gpu_processes()
            foreign_gpu = [line for line in gpu if int(line.split(',')[0]) not in allowed]
            observations.append({'elapsed_seconds':now-start, 'gpu_processes':gpu, 'foreign_cpu_over_quarter_core':active})
            if active or foreign_gpu:
                foreign.append({'cpu':active, 'gpu':foreign_gpu})
            previous, previous_time = snapshot, now
            if process.poll() is not None:
                break
    if process.returncode:
        raise RuntimeError(f'Benchmark exited {process.returncode}: {log_path}')
    return {'idle_scheduling_observations':waiting, 'gpu_before':initial_gpu, 'samples':observations, 'foreign_work_observed':foreign,
            'scope':'One-second process/GPU sampling; records unrelated CPU processes above 0.25 cores. '
                    'No validation, archive processing or other benchmark runs overlap the measured child.'}
