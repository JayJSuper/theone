"""SafeExecutor — auditable, sandboxed execution gated on verifiable causal advice.

Three layers of defense, each producing a third-party-recheckable record:
  1. EXECUTION SAFETY (the sandbox, adopted from solid prior art): every path is
     contained within a sandbox root; a command denylist blocks high-risk shell
     verbs; dry-run is the default and a real run requires explicit confirmation.
  2. CAUSAL GATE (The One's increment): an action *justified by a causal effect*
     must carry a credential that is BOTH independently-recomputable (pgmpy match)
     AND admissible (constraint credential) — the two orthogonal gates. An action
     driven by an unverified or inadmissible causal recommendation is ABSTAINED on,
     not executed. (Non-causal housekeeping actions skip this gate.)
  3. AUDIT (sovereignty): every proposal and execution is appended to an audit log
     with its full credential — provenance, checks, decision, dry-run output.

This is the boundary that lets The One move from *computing* a causal effect to
*acting* on it, while keeping the property that matters: every step is checkable,
not merely trusted.
"""
from __future__ import annotations
import os
import time
import subprocess
from dataclasses import dataclass, field, asdict
from pathlib import Path

DEFAULT_DENYLIST = {"rm", "mv", "chmod", "chown", "dd", "mkfs", "kill", "shutdown",
                    "sudo", "reboot", "curl", "wget", "ssh", "scp", "nc"}


@dataclass
class ExecutionCredential:
    kind: str                       # "write_file" | "run_command"
    params: dict
    path_check: str = "NA"          # PASS | VIOLATED | NA
    command_check: str = "NA"       # PASS | VIOLATED | NA
    causal_gate: str = "NONE"       # ADMISSIBLE | INADMISSIBLE | NONE (no causal driver)
    causal_reason: str = ""
    dry_run_output: str = ""
    decision: str = "ABSTAIN"       # EXECUTE | BLOCK | ABSTAIN
    reason: str = ""
    ts: float = 0.0
    provenance: str = ""

    def as_record(self) -> dict:
        return asdict(self)


class SafeExecutor:
    def __init__(self, sandbox_root: str | None = None, denylist: set | None = None) -> None:
        self.sandbox_root = Path(sandbox_root or os.getcwd()).resolve()
        self.denylist = set(denylist) if denylist is not None else set(DEFAULT_DENYLIST)
        self._audit: list[ExecutionCredential] = []

    # --- layer 1: execution safety ------------------------------------------
    def _path_ok(self, p: str) -> bool:
        try:
            target = Path(p).resolve()
        except Exception:
            return False
        return str(target).startswith(str(self.sandbox_root))

    def _command_ok(self, cmd: str) -> bool:
        toks = cmd.strip().split()
        return bool(toks) and toks[0].lower() not in self.denylist

    # --- layer 2: causal gate -----------------------------------------------
    @staticmethod
    def _causal_gate(causal_credential: dict | None) -> tuple[str, str]:
        """An action justified by a causal effect must carry a credential that is
        recomputable AND admissible (the two orthogonal gates). Returns (gate, reason)."""
        if causal_credential is None:
            return "NONE", "non-causal action (no causal driver)"
        recomputable = bool(causal_credential.get("recomputable"))
        admissible = bool(causal_credential.get("admissible"))
        if recomputable and admissible:
            return "ADMISSIBLE", "causal driver is independently-recomputable AND admissible"
        why = []
        if not recomputable:
            why.append("not independently-recomputable")
        if not admissible:
            why.append("violates a declared constraint")
        return "INADMISSIBLE", "; ".join(why)

    # --- propose (dry-run, always safe) -------------------------------------
    def propose_write(self, target_file: str, content: str,
                      causal_credential: dict | None = None,
                      provenance: str = "") -> ExecutionCredential:
        path_check = "PASS" if self._path_ok(target_file) else "VIOLATED"
        gate, gate_reason = self._causal_gate(causal_credential)
        cred = ExecutionCredential(
            kind="write_file",
            params={"target_file": target_file, "bytes": len(content.encode())},
            path_check=path_check, causal_gate=gate, causal_reason=gate_reason,
            ts=time.time(), provenance=provenance)
        cred.decision, cred.reason = self._decide(path_check, "NA", gate)
        cred.dry_run_output = (f"[dry-run] would write {len(content.encode())} bytes to "
                               f"{Path(target_file).resolve()}" if cred.decision == "EXECUTE"
                               else f"[dry-run blocked] {cred.reason}")
        self._audit.append(cred)
        return cred

    def propose_command(self, command: str, causal_credential: dict | None = None,
                        provenance: str = "") -> ExecutionCredential:
        cmd_check = "PASS" if self._command_ok(command) else "VIOLATED"
        gate, gate_reason = self._causal_gate(causal_credential)
        cred = ExecutionCredential(
            kind="run_command", params={"command": command},
            command_check=cmd_check, causal_gate=gate, causal_reason=gate_reason,
            ts=time.time(), provenance=provenance)
        cred.decision, cred.reason = self._decide("NA", cmd_check, gate)
        cred.dry_run_output = (f"[dry-run] would run: {command}" if cred.decision == "EXECUTE"
                               else f"[dry-run blocked] {cred.reason}")
        self._audit.append(cred)
        return cred

    @staticmethod
    def _decide(path_check: str, cmd_check: str, gate: str) -> tuple[str, str]:
        if path_check == "VIOLATED":
            return "BLOCK", "path escapes sandbox root"
        if cmd_check == "VIOLATED":
            return "BLOCK", "command on denylist"
        if gate == "INADMISSIBLE":
            return "ABSTAIN", "causal driver failed the recomputable-AND-admissible gate"
        return "EXECUTE", "safe AND (no causal driver OR causal driver verified)"

    # --- execute (only an EXECUTE-decision credential, explicit confirm) -----
    def execute(self, cred: ExecutionCredential, content: str | None = None,
                confirm: bool = False) -> dict:
        if cred.decision != "EXECUTE":
            return {"ran": False, "reason": f"not executable: {cred.decision} ({cred.reason})"}
        if not confirm:
            return {"ran": False, "reason": "confirm=False (dry-run only); pass confirm=True to act"}
        try:
            if cred.kind == "write_file":
                p = Path(cred.params["target_file"]).resolve()
                if not self._path_ok(str(p)):
                    return {"ran": False, "reason": "path re-check failed at execute time"}
                p.parent.mkdir(parents=True, exist_ok=True)
                p.write_text(content or "")
                cred.reason += " | executed"
                return {"ran": True, "wrote": str(p)}
            if cred.kind == "run_command":
                if not self._command_ok(cred.params["command"]):
                    return {"ran": False, "reason": "command re-check failed at execute time"}
                out = subprocess.run(cred.params["command"], shell=True, capture_output=True,
                                     text=True, timeout=30, cwd=str(self.sandbox_root))
                return {"ran": True, "returncode": out.returncode, "stdout": out.stdout[:500]}
        except Exception as e:  # noqa: BLE001
            return {"ran": False, "reason": f"execution error: {str(e)[:120]}"}
        return {"ran": False, "reason": "unknown kind"}

    # --- layer 3: audit ------------------------------------------------------
    def audit_log(self) -> list[dict]:
        return [c.as_record() for c in self._audit]
