"""Diagnose Neo4j connectivity end-to-end.

Run:
    python -m scripts.check_neo4j

Walks DNS → TCP → Bolt handshake → auth → query, printing the first
failure with the most likely cause. Useful when ingestion fails with
ServiceUnavailable or "Could not connect to Neo4j database".
"""
import socket
import sys
from urllib.parse import urlparse

from src.config import get_config


def _ok(msg: str) -> None:
    print(f"  \033[32mOK\033[0m   {msg}")


def _fail(msg: str, hint: str = "") -> None:
    print(f"  \033[31mFAIL\033[0m {msg}")
    if hint:
        print(f"       → {hint}")


def main() -> int:
    cfg = get_config()
    uri = cfg.neo4j_uri
    user = cfg.neo4j_username
    pw = cfg.neo4j_password
    db = cfg.neo4j_database

    print(f"\nNeo4j URI:      {uri}")
    print(f"Username:       {user}")
    print(f"Database:       {db}")
    print(f"Password:       {'(set)' if pw else '(EMPTY)'}\n")

    parsed = urlparse(uri)
    host = parsed.hostname
    port = parsed.port or 7687
    scheme = parsed.scheme

    if not host:
        _fail("URI has no hostname", "Check NEO4J_URI in .env")
        return 1

    # 1. DNS
    print("1. DNS resolution")
    try:
        ip = socket.gethostbyname(host)
        _ok(f"{host} → {ip}")
    except socket.gaierror as exc:
        _fail(f"cannot resolve {host}: {exc}",
              "Wrong hostname, or no internet. For Aura, check the instance still exists.")
        return 1

    # 2. TCP reachability on Bolt port
    print(f"\n2. TCP connect to {host}:{port}")
    try:
        with socket.create_connection((host, port), timeout=5):
            _ok(f"port {port} is reachable")
    except (socket.timeout, OSError) as exc:
        _fail(f"port {port} unreachable: {exc}",
              "Likely causes: (a) Aura instance paused — resume in console; "
              "(b) outbound 7687 blocked by ISP/firewall — try mobile hotspot or "
              "switch to a local Neo4j (bolt://localhost:7687); "
              "(c) local Neo4j not running — start it with docker/Desktop.")
        return 1

    # 3. Driver-level Bolt handshake + auth
    print("\n3. Bolt handshake + authentication")
    try:
        from neo4j import GraphDatabase
    except ImportError:
        _fail("neo4j driver not installed", "pip install neo4j")
        return 1

    try:
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        driver.verify_connectivity()
        _ok(f"connected via {scheme}")
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "auth" in msg.lower() or "unauthorized" in msg.lower():
            hint = "Bad credentials. Check NEO4J_USERNAME / NEO4J_PASSWORD in .env."
        elif "routing" in msg.lower():
            hint = ("Routing failed. If using neo4j+s:// against a paused Aura "
                    "instance, resume it. If on a network blocking 7687, switch "
                    "to local Neo4j (bolt://localhost:7687).")
        else:
            hint = "Verify URI scheme matches your deployment (neo4j+s:// for Aura, bolt:// for local)."
        _fail(f"{type(exc).__name__}: {msg}", hint)
        return 1

    # 4. Round-trip query against the configured database
    print(f"\n4. Round-trip query on database '{db}'")
    try:
        with driver.session(database=db) as session:
            result = session.run("RETURN 1 AS n").single()
            if result and result["n"] == 1:
                _ok("query returned 1")
            else:
                _fail("query returned unexpected result")
                return 1
    except Exception as exc:
        msg = str(exc)
        hint = ""
        if "database" in msg.lower() and "not" in msg.lower():
            hint = (f"Database '{db}' does not exist. For Aura set NEO4J_DATABASE "
                    "to your instance ID; for local Neo4j use 'neo4j'.")
        _fail(f"{type(exc).__name__}: {msg}", hint)
        return 1
    finally:
        driver.close()

    # 5. Server version
    print("\n5. Server info")
    try:
        driver = GraphDatabase.driver(uri, auth=(user, pw))
        with driver.session(database=db) as session:
            row = session.run(
                "CALL dbms.components() YIELD name, versions, edition "
                "RETURN name, versions[0] AS version, edition"
            ).single()
            if row:
                _ok(f"{row['name']} {row['version']} ({row['edition']})")
        driver.close()
    except Exception as exc:
        # non-fatal
        print(f"  (could not read server version: {exc})")

    print("\nAll checks passed. Neo4j is reachable.\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
