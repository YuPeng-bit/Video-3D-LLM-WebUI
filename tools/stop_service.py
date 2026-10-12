"""Stop only this WebUI's uvicorn process at the selected port. No force kill."""
import argparse,os,signal,time
from pathlib import Path

def matches_service(cwd, args, root, port):
    if Path(cwd).resolve()!=Path(root).resolve(): return False
    if 'uvicorn' not in args or 'app.server:app' not in args: return False
    return ('--port' in args and args[args.index('--port')+1:args.index('--port')+2]==[str(port)]) or '--port='+str(port) in args

def identity(pid):
    path=Path('/proc')/str(pid)
    try:
        cwd=(path/'cwd').resolve(strict=True)
        args=(path/'cmdline').read_bytes().decode().split('\0')
        started=(path/'stat').read_text().rsplit(')',1)[1].split()[19]
        return cwd,args,started
    except (FileNotFoundError,ProcessLookupError,PermissionError): return None

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument('--port',type=int,default=int(os.getenv('PORT','8099')))
    ap.add_argument('--dry-run',action='store_true')
    args=ap.parse_args()
    root=Path(__file__).resolve().parents[1]
    targets=[]
    for path in Path('/proc').iterdir():
        if not path.name.isdigit(): continue
        pid=int(path.name); state=identity(pid)
        if state and matches_service(state[0],state[1],root,args.port): targets.append((pid,state[2]))
    for pid,started in targets:
        print(('Would stop' if args.dry_run else 'Stopping'),pid,'in',root,'port',args.port)
        current=identity(pid)
        if not args.dry_run and current and current[2]==started:
            try: os.kill(pid,signal.SIGTERM)
            except ProcessLookupError: pass
    if args.dry_run: return
    for _ in range(150):
        remaining=[pid for pid,started in targets if (identity(pid) or (None,None,None))[2]==started]
        if not remaining: return
        time.sleep(.1)
    raise SystemExit('Own WebUI still exiting; no force kill: '+str(remaining))

if __name__=='__main__': main()
