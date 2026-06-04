"""FinBank — the deliberately weak banking test-bed.

The paper evaluates ADDF against a "vulnerable Flask-and-MySQL stack". For
portability and zero-config reproduction this implementation is an in-process
service backed by SQLite (stdlib), with an optional Flask HTTP wrapper.

The weaknesses below are *intentional and documented* so the deception layer has
something to defend. They are modelled, not weaponised — there is no real exploit
code here, and only synthetic data (standard test PANs) is ever stored.
"""
from __future__ import annotations

import secrets
import sqlite3
from dataclasses import dataclass

import numpy as np

# Documented intentional weaknesses of the test-bed (what the attacker exercises):
INTENTIONAL_WEAKNESSES = [
    "No login rate-limiting or lockout (enables credential stuffing)",
    "Session tokens are guessable in length (modelled, not exploitable here)",
    "Ledger export endpoint lacks per-record authorization checks",
    "Internal service-to-service calls are unauthenticated (lateral movement)",
]

# Standard PCI test PAN — never a real card number.
TEST_PAN_PREFIX = "411111111111"


@dataclass
class Session:
    token: str
    username: str
    is_attacker_flagged: bool = False  # set if a honeytoken credential was used


class FinBank:
    """Synthetic banking core: customers, accounts, and a sensitive ledger."""

    def __init__(self, config, rng: np.random.Generator) -> None:
        self.config = config
        self.rng = rng
        self.db = sqlite3.connect(":memory:")
        self.db.row_factory = sqlite3.Row
        self._honeytokens: dict[str, str] = {}   # username -> password tripwires
        self.exfiltrated_records = 0              # real records that actually left
        self._build_schema()
        self._seed()

    # -- setup ---------------------------------------------------------------
    def _build_schema(self) -> None:
        self.db.executescript(
            """
            CREATE TABLE customers (id INTEGER PRIMARY KEY, username TEXT UNIQUE,
                                    password TEXT, full_name TEXT);
            CREATE TABLE accounts  (id INTEGER PRIMARY KEY, customer_id INTEGER,
                                    pan_masked TEXT, balance REAL);
            CREATE TABLE ledger    (id INTEGER PRIMARY KEY, account_id INTEGER,
                                    amount REAL, memo TEXT);
            """
        )

    def _seed(self) -> None:
        n = self.config.n_accounts
        for i in range(n):
            uname = f"user{i:04d}"
            pwd = secrets.token_hex(6)
            self.db.execute(
                "INSERT INTO customers (id, username, password, full_name) VALUES (?,?,?,?)",
                (i, uname, pwd, f"Synthetic Holder {i:04d}"),
            )
            pan = f"{TEST_PAN_PREFIX}{int(self.rng.integers(1000, 9999)):04d}"
            masked = f"{pan[:6]}******{pan[-4:]}"
            self.db.execute(
                "INSERT INTO accounts (id, customer_id, pan_masked, balance) VALUES (?,?,?,?)",
                (i, i, masked, float(self.config.starting_balance)),
            )
            for j in range(5):  # a few sensitive ledger rows per account
                self.db.execute(
                    "INSERT INTO ledger (account_id, amount, memo) VALUES (?,?,?)",
                    (i, float(self.rng.normal(0, 250)), f"txn-{i:04d}-{j}"),
                )
        self.db.commit()

    # -- operations the attacker / users exercise ----------------------------
    def authenticate(self, username: str, password: str) -> Session | None:
        """Validate credentials. No lockout by design (intentional weakness)."""
        if username in self._honeytokens and self._honeytokens[username] == password:
            # A honeytoken was used: only an attacker would have these credentials.
            return Session(token=secrets.token_hex(16), username=username,
                           is_attacker_flagged=True)
        row = self.db.execute(
            "SELECT username FROM customers WHERE username = ? AND password = ?",
            (username, password),
        ).fetchone()
        if row is None:
            return None
        return Session(token=secrets.token_hex(16), username=username)

    def real_record_count(self) -> int:
        return self.db.execute("SELECT COUNT(*) AS c FROM ledger").fetchone()["c"]

    def export_ledger(self, limit: int = 1000) -> list[dict]:
        """The sensitive export. In a real exfiltration these rows leave the bank."""
        rows = self.db.execute(
            "SELECT account_id, amount, memo FROM ledger LIMIT ?", (limit,)
        ).fetchall()
        return [dict(r) for r in rows]

    def register_real_exfiltration(self, n_records: int) -> None:
        self.exfiltrated_records += int(n_records)

    # -- honeytokens (planted by the decoy layer) ----------------------------
    def plant_honeytoken(self, username: str, password: str) -> None:
        self._honeytokens[username] = password

    @property
    def honeytokens(self) -> dict[str, str]:
        return dict(self._honeytokens)


def serve(config=None, host: str = "127.0.0.1", port: int = 5000):  # pragma: no cover
    """Optional: expose FinBank as a real HTTP service. Requires Flask.

        pip install -e .[serve]
        python -c "from addf.finbank import serve; serve()"
    """
    try:
        from flask import Flask, jsonify, request
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("Flask not installed. Install with: pip install -e .[serve]") from exc

    from addf.config import ADDFConfig

    cfg = config or ADDFConfig()
    bank = FinBank(cfg, np.random.default_rng(cfg.seed))
    app = Flask(__name__)

    @app.post("/login")
    def login():
        data = request.get_json(force=True, silent=True) or {}
        s = bank.authenticate(data.get("username", ""), data.get("password", ""))
        if s is None:
            return jsonify({"ok": False}), 401
        return jsonify({"ok": True, "token": s.token, "flagged": s.is_attacker_flagged})

    @app.get("/ledger")
    def ledger():
        return jsonify(bank.export_ledger(limit=int(request.args.get("limit", 100))))

    app.run(host=host, port=port)
    return app
