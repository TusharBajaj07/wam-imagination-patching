"""Submit jobs to the DEPLOYED wam-interp app, so they keep running if the Mac sleeps or goes offline.

`modal run --detach` was not enough: on 2026-10-26 the Mac slept (battery) and Modal stopped every running job ~3 min
later. A deployed app's spawned calls do not depend on the client at all.

  modal deploy wam/modal_app.py          # once, and again after any code change
  python wam/launch.py --sim flip_test "--cond futureB --layers 21-30 --file x.jsonl"
  python wam/launch.py --status          # calls submitted from this Mac and their state

Each submitted call id is appended to wam/logs/calls.jsonl. Results land on the wam-interp-out volume as usual;
printed output is in the Modal dashboard (app wam-interp).
"""
import json
import sys
import time
from pathlib import Path

import modal

LOG = Path(__file__).parent / "logs" / "calls.jsonl"


def status():
    for line in LOG.read_text().splitlines():
        c = json.loads(line)
        fc = modal.FunctionCall.from_id(c["id"])
        try:
            fc.get(timeout=0)
            state = "done"
        except TimeoutError:
            state = "running"
        except Exception as exc:  # noqa: BLE001  the remote error, re-raised by .get()
            state = f"FAILED {type(exc).__name__}: {str(exc)[:120]}"
        print(f"{c['t']}  {c['script']:12s} {state:10s} {c['args']}")


def main(argv):
    if argv[:1] == ["--status"]:
        return status()
    sim = argv[0] == "--sim"
    script, args = argv[1 if sim else 0], argv[2 if sim else 1] if len(argv) > (2 if sim else 1) else ""
    fn = modal.Function.from_name("wam-interp", "run_sim" if sim else "run")
    call = fn.spawn(script, args.split())
    rec = {"t": time.strftime("%Y-%m-%d %H:%M"), "id": call.object_id, "script": script, "sim": sim, "args": args}
    LOG.parent.mkdir(exist_ok=True)
    with LOG.open("a") as f:
        f.write(json.dumps(rec) + "\n")
    print("submitted", rec)


if __name__ == "__main__":
    main(sys.argv[1:])
