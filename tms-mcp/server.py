"""
TMS MCP Server — HappyRobot Logistics
Exposes the legacy TCP TMS as MCP tools consumable by the HappyRobot platform.

Tools:
  search_loads   — LOAD_QUERY: find open loads by lane / equipment
  get_load       — LOAD_GET:   fetch full detail for one load
  book_load      — LOAD_BOOK:  commit a booking (enforces max_rate ceiling)
  ping           — DEBUG_ECHO: transport / auth health check (bypasses fault injection)

Fault handling
--------------
The TMS non-production environment injects four categories of fault without
signalling them in the response (spec §"Fault behavior"):

  Timeout           — server accepts the connection but never writes a response;
                      the idle timeout (spec: 30 s) eventually closes it.
                      Detected by socket.timeout on recv().
                      Action: raise TMSTransportError → retry.

  Partial response  — server writes some record lines then closes without END.
                      Detected by connection close (empty chunk) before END is seen.
                      Action: raise TMSTransportError → retry.
                      The truncated buffer is discarded; only complete, END-terminated
                      responses are accepted.

  Malformed response — server writes a response that violates framing rules:
                      extra delimiters, unterminated lines, field values that are
                      not valid ASCII, or structured fields that fail format checks
                      (e.g. letters in a numeric field, wrong-length datetime).
                      Detected during _parse() by structural checks and field format
                      validation.  Field widths are NOT checked — the spec says widths
                      come from samples, but the wire is authoritative, and the live
                      TMS returns wider values than the doc samples for high-value
                      loads.
                      Action: raise TMSTransportError → retry.

  Delayed termination — server holds the connection open after a complete response.
                      Detected implicitly: once END (or ERR) is seen the response
                      is returned immediately without waiting for the connection to
                      close, so the delay is invisible to callers.

The retry wrapper (_tms) retries on TMSTransportError but not on TMSError
(application-level errors such as UNKNOWN_LOAD or AUTH_FAILED are deterministic
and retrying them would be incorrect).
"""

import os
import socket
import logging
from typing import Optional

from mcp.server.fastmcp import FastMCP
from mcp.server.transport_security import TransportSecuritySettings

# ── configuration ─────────────────────────────────────────────────────────────

TMS_HOST = os.getenv("TMS_HOST", "tramway.proxy.rlwy.net")
TMS_PORT = int(os.getenv("TMS_PORT", "17159"))
TMS_AUTH = os.getenv("TMS_AUTH", "hr-fde-abioppenheim-2026")

# Socket read timeout.  The spec states the server's idle timeout is 30 s;
# we sit slightly below that so we time out before the server closes the
# connection on us, which would look like a partial response rather than a
# timeout and complicate diagnosis.
SOCKET_TIMEOUT = int(os.getenv("SOCKET_TIMEOUT", "25"))

MAX_RETRIES = int(os.getenv("MAX_RETRIES", "3"))

# Maximum frame size per the protocol spec (bytes, including \r\n terminator).
MAX_FRAME_BYTES = 4096

# Statuses that mean a load is NOT bookable and should be hidden from carriers.
# LOAD_QUERY is documented as the "open board", so we use a denylist of clearly
# reserved states rather than an exact == "OPEN" allowlist.  An allowlist silently
# zeroes results whenever the wire uses any token other than the literal "OPEN"
# (e.g. AVAILABLE / POSTED / NEW), which is the failure that produced empty
# search results.  book_load re-checks status server-side before committing, so
# a stray reserved load slipping through here still cannot be booked.
RESERVED_STATUSES = frozenset({"BOOKED", "PENDING", "COVERED", "RESERVED", "CANCELLED", "EXPIRED"})

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)-8s %(message)s")
log = logging.getLogger("tms-mcp")

mcp = FastMCP(
    "TMS — HappyRobot Logistics",
    transport_security=TransportSecuritySettings(enable_dns_rebinding_protection=False),
)

# ── exception hierarchy ───────────────────────────────────────────────────────

class TMSError(Exception):
    """Application-level error returned by the TMS (ERR| response line).
    These are deterministic; the retry wrapper does NOT retry them."""
    def __init__(self, code: str, message: str):
        self.code = code
        self.message = message
        super().__init__(f"TMS {code}: {message}")


class TMSTransportError(Exception):
    """Transport or framing fault — timeout, partial response, or malformed
    response.  These are retriable per the spec's fault-handling guidance."""


# ── field format registry ─────────────────────────────────────────────────────
# The spec says field widths should be counted from transcript samples, but also
# that "where behavior is not stated, the wire is authoritative."  The live TMS
# returns RATE values wider than the doc samples (e.g. 8 chars for loads ≥$10k),
# so validating padded value lengths produces false positives on valid records.
#
# We validate FORMAT instead — constraints that are invariant regardless of value
# magnitude (a rate is always digits; a state code is always 2 letters; a datetime
# is always 14 digits).  This catches genuinely malformed records (garbage in a
# numeric field, wrong-length datetime) without rejecting valid wide values.

import re as _re

_FIELD_FORMATS: dict[str, str] = {
    # Identifiers
    "LOAD_ID":     r"^LD\d+\s*$",          # LD followed by digits, optional trailing space-pad
    # BOOKING_REF intentionally excluded: spec says it is "server-assigned and opaque;
    # do not parse it."  The live TMS returns alphanumeric tokens (e.g. JPHTIMW8X1HIET4M)
    # that do not match the BR\d+ pattern seen in doc samples — validating the format
    # causes successful bookings to be misclassified as malformed responses and retried.
    # State codes — exactly 2 uppercase letters (no padding)
    "ORIG_STATE":  r"^[A-Z]{2}$",
    "DEST_STATE":  r"^[A-Z]{2}$",
    # ZIP codes — exactly 5 digits (no padding)
    "ORIG_ZIP":    r"^\d{5}$",
    "DEST_ZIP":    r"^\d{5}$",
    # Datetimes — exactly 14 digits (YYYYMMDDHHmmss)
    "PICKUP_DT":   r"^\d{14}$",
    "DELIVERY_DT": r"^\d{14}$",
    "TIMESTAMP":   r"^\d{14}$",
    # Equipment type — uppercase letters and underscores only (space-padded)
    "EQTYPE":      r"^[A-Z_]+\s*$",
    # Numeric fields — one or more digits, optional trailing space-pad
    "RATE":        r"^\d+\s*$",
    "MILES":       r"^\d+\s*$",
    "WEIGHT":      r"^\d+\s*$",
    "PIECES":      r"^\d+\s*$",
    "MAX_BUY":     r"^\d+\s*$",
}

# Required fields for each command's response records, used to detect
# structurally incomplete (partial) records that arrived without END.
_REQUIRED_FIELDS: dict[str, frozenset[str]] = {
    "LOAD_QUERY": frozenset({"LOAD_ID", "ORIG_STATE", "DEST_STATE", "EQTYPE", "RATE", "STATUS"}),
    "LOAD_GET":   frozenset({"LOAD_ID", "ORIG_STATE", "DEST_STATE", "PICKUP_DT", "EQTYPE", "RATE", "STATUS"}),
    "LOAD_BOOK":  frozenset({"LOAD_ID", "BOOKING_REF", "STATUS", "TIMESTAMP"}),
    "DEBUG_ECHO": frozenset({"MSG"}),
}


# ── TMS TCP client ────────────────────────────────────────────────────────────

def _send(command_line: str) -> str:
    """
    Open a fresh TCP connection, send one request line, and read the full
    response.

    Returns the raw response text (all lines including END) on success.

    Raises:
        TMSTransportError: timeout, partial response (no END before close),
                           or frame exceeding MAX_FRAME_BYTES.
        TMSError:          application-level ERR| response (not retriable).
    """
    log.info("TMS → %s", command_line[:120])
    sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    sock.settimeout(SOCKET_TIMEOUT)
    try:
        sock.connect((TMS_HOST, TMS_PORT))
        sock.sendall((command_line + "\r\n").encode("ascii"))

        buf = b""
        while True:
            try:
                chunk = sock.recv(4096)
            except socket.timeout:
                # Timeout fault: server accepted but never responded.
                raise TMSTransportError(
                    f"Socket timeout after {SOCKET_TIMEOUT}s waiting for TMS response"
                )

            if not chunk:
                # Connection closed by server.  Per the spec, a well-formed
                # response always ends with END\r\n before the server closes.
                # If we reach EOF without having seen END or ERR|, this is a
                # partial-response fault.
                text = buf.decode("ascii", errors="replace")
                if _response_is_complete(text):
                    # Delayed-termination case: complete response received,
                    # server just held the connection a bit longer than expected.
                    return text
                raise TMSTransportError(
                    f"TMS closed connection before END terminator "
                    f"(partial response, {len(buf)} bytes buffered)"
                )

            buf += chunk

            # Guard against frames that exceed the protocol's stated maximum.
            if len(buf) > MAX_FRAME_BYTES * 64:
                # 64× headroom for multi-record LOAD_QUERY responses; a single
                # frame is ≤4096 bytes but a full response may contain many.
                raise TMSTransportError(
                    f"Response buffer exceeded safety limit ({len(buf)} bytes)"
                )

            # Check whether a terminal line has arrived so we can return
            # without waiting for the server to close the connection
            # (handles the delayed-termination fault category).
            text = buf.decode("ascii", errors="replace")
            if _response_is_complete(text):
                log.info("TMS ← %d bytes", len(buf))
                return text

    finally:
        sock.close()


def _response_is_complete(text: str) -> bool:
    """Return True if text contains a terminal line (END or ERR|)."""
    for line in text.splitlines():
        stripped = line.strip()
        if stripped == "END" or stripped.startswith("ERR|"):
            return True
    return False


def _parse(raw: str, cmd: str = "") -> list[dict]:
    """
    Parse a complete TMS response into a list of field dicts.

    Raises:
        TMSError:          if the response is an ERR| line.
        TMSTransportError: if any record line is structurally malformed
                           (extra delimiters, non-ASCII bytes, fields that
                           exceed their declared fixed width, or required
                           fields missing from a record).
    """
    records = []
    for line in raw.splitlines():
        line = line.strip()
        if not line or line == "END":
            continue

        if line.startswith("ERR|"):
            # ERR lines have the shape: ERR|CODE:<code>|MSG:<msg>
            # The leading 'ERR' segment has no colon, so we must NOT pass the
            # full line through _kv_parse (which requires every segment to be
            # KEY:VALUE).  Parse only the segments after the sentinel instead.
            parts = _kv_parse_err(line)
            raise TMSError(parts.get("CODE", "UNKNOWN"), parts.get("MSG", line))

        # Structural check: the line must be valid ASCII (non-ASCII bytes
        # indicate a malformed response — the protocol declares ASCII encoding).
        try:
            line.encode("ascii")
        except UnicodeEncodeError:
            raise TMSTransportError(f"Malformed record: non-ASCII content in response line")

        record = _kv_parse(line)
        _validate_field_formats(record, line)

        if cmd and cmd in _REQUIRED_FIELDS:
            _validate_required_fields(record, cmd, line)

        records.append(record)

    return records


def _kv_parse_err(line: str) -> dict:
    """
    Parse an ERR| response line into a dict.

    ERR lines have the shape  ERR|CODE:<code>|MSG:<msg>  where the first
    segment 'ERR' is a bare sentinel with no colon.  We skip it and parse
    only the KEY:VALUE segments that follow.
    """
    result = {}
    segments = line.split("|")
    for part in segments[1:]:   # skip 'ERR' sentinel
        if ":" in part:
            k, _, v = part.partition(":")
            result[k.strip()] = v.strip()
    return result


def _kv_parse(line: str) -> dict:
    """
    Parse a single |KEY:VALUE| line into a dict.

    The spec states values must not contain | or \r\n.  Extra pipe characters
    within a value (extra delimiters) are a malformed-response indicator.
    We detect them by checking that every segment contains exactly one colon
    that splits a non-empty key from its value.
    """
    result = {}
    for part in line.split("|"):
        if not part:
            # Leading, trailing, or consecutive pipes — malformed framing.
            raise TMSTransportError(
                f"Malformed record: unexpected empty segment in '{line[:80]}'"
            )
        if ":" not in part:
            raise TMSTransportError(
                f"Malformed record: segment without KEY:VALUE separator in '{line[:80]}'"
            )
        k, _, v = part.partition(":")
        k = k.strip()
        if not k:
            raise TMSTransportError(
                f"Malformed record: empty key in '{line[:80]}'"
            )
        result[k] = v  # preserve raw padding; callers strip when needed
    return result


def _validate_field_formats(record: dict, raw_line: str) -> None:
    """
    Check that structured fields match their expected format.

    We validate format (digit-only numerics, 2-letter state codes, 14-digit
    datetimes) rather than padded value length.  The spec says widths come from
    samples, but the wire is authoritative — the live TMS returns values wider
    than the doc samples for high-value loads, so width checks produce false
    positives.  Format checks catch genuinely corrupt data (e.g. a letter in a
    numeric field) without rejecting valid wide values.
    """
    for field, pattern in _FIELD_FORMATS.items():
        if field in record:
            raw_value = record[field]
            if not _re.match(pattern, raw_value):
                raise TMSTransportError(
                    f"Malformed record: field {field} value {raw_value!r} does not "
                    f"match expected format '{pattern}' in '{raw_line[:80]}'"
                )


def _validate_required_fields(record: dict, cmd: str, raw_line: str) -> None:
    """
    Verify that all required fields for a command's response are present.

    A record that arrived before the server closed the connection — without
    END — may pass the EOF check if some other condition triggered early
    parsing.  Checking required fields catches structurally incomplete records
    that slipped through.
    """
    required = _REQUIRED_FIELDS.get(cmd, frozenset())
    missing = required - record.keys()
    if missing:
        raise TMSTransportError(
            f"Malformed record: missing required fields {missing} for {cmd} "
            f"in '{raw_line[:80]}'"
        )


def _tms(command_line: str, cmd: str = "") -> list[dict]:
    """
    Retry wrapper around _send + _parse.

    Retries on TMSTransportError (timeout, partial, malformed) up to
    MAX_RETRIES times.  Does NOT retry on TMSError (application errors are
    deterministic — retrying UNKNOWN_LOAD or AUTH_FAILED would be wrong).
    """
    last_err: Exception | None = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            raw = _send(command_line)
            return _parse(raw, cmd)
        except TMSError:
            raise  # application error — do not retry
        except TMSTransportError as exc:
            last_err = exc
            log.warning(
                "TMS transport fault attempt %d/%d [%s]: %s",
                attempt, MAX_RETRIES, cmd or "?", exc,
            )
        except Exception as exc:
            # Catch-all for unexpected socket / OS errors (e.g. connection
            # refused, DNS failure) — treat as transport faults.
            last_err = TMSTransportError(str(exc))
            log.warning(
                "TMS unexpected error attempt %d/%d [%s]: %s",
                attempt, MAX_RETRIES, cmd or "?", exc,
            )

    raise TMSTransportError(
        f"TMS unavailable after {MAX_RETRIES} attempt(s): {last_err}"
    )


# ── formatters ────────────────────────────────────────────────────────────────

def _dt(raw: str) -> Optional[str]:
    raw = raw.strip()
    if len(raw) == 14 and raw.isdigit():
        return (
            f"{raw[:4]}-{raw[4:6]}-{raw[6:8]}"
            f"T{raw[8:10]}:{raw[10:12]}:{raw[12:14]}"
        )
    return raw or None


def _int(raw: str) -> Optional[int]:
    raw = raw.strip()
    return int(raw) if raw.isdigit() else None


def _load_summary(r: dict) -> dict:
    return {
        "load_id":         r.get("LOAD_ID", "").strip(),
        "origin":          _location(r, "ORIG"),
        "destination":     _location(r, "DEST"),
        "pickup_datetime": _dt(r.get("PICKUP_DT", "")),
        "equipment_type":  r.get("EQTYPE", "").strip(),
        "rate":            _int(r.get("RATE", "")),
        "miles":           _int(r.get("MILES", "")),
        "status":          r.get("STATUS", "").strip(),
    }


def _load_detail(r: dict) -> dict:
    """Full load record.  _max_buy is kept for internal ceiling enforcement
    and must be removed before returning data to the agent."""
    return {
        "load_id":           r.get("LOAD_ID", "").strip(),
        "origin":            _location(r, "ORIG"),
        "destination":       _location(r, "DEST"),
        "pickup_datetime":   _dt(r.get("PICKUP_DT", "")),
        "delivery_datetime": _dt(r.get("DELIVERY_DT", "")),
        "equipment_type":    r.get("EQTYPE", "").strip(),
        "loadboard_rate":    _int(r.get("RATE", "")),
        "weight":            _int(r.get("WEIGHT", "")),
        "commodity_type":    r.get("COMMODITY", "").strip(),
        "num_of_pieces":     _int(r.get("PIECES", "")),
        "miles":             _int(r.get("MILES", "")),
        "dimensions":        r.get("DIMS", "").strip(),
        "notes":             r.get("NOTES", "").strip(),
        "status":            r.get("STATUS", "").strip(),
        # Internal only — stripped before returning to agent callers.
        "_max_buy":          _int(r.get("MAX_BUY", "")),
    }


def _location(r: dict, prefix: str) -> str:
    city  = r.get(f"{prefix}_CITY",  "").strip()
    state = r.get(f"{prefix}_STATE", "").strip()
    zip_  = r.get(f"{prefix}_ZIP",   "").strip()
    parts = [p for p in [city, state, zip_] if p]
    return ", ".join(parts)


# ── MCP tools ─────────────────────────────────────────────────────────────────

@mcp.tool()
def search_loads(
    orig_state:  Optional[str] = None,
    dest_state:  Optional[str] = None,
    orig_city:   Optional[str] = None,
    dest_city:   Optional[str] = None,
    eqtype:      Optional[str] = None,
    max_results: int = 10,
) -> dict:
    """
    Search the open load board for available loads.
    Provide at least one filter: orig_state, dest_state, orig_city, dest_city, or eqtype.
    Returns a list of matching loads with key details (origin, destination, rate, equipment).
    """
    params: dict[str, str] = {}
    if orig_state: params["ORIG_STATE"] = orig_state.upper()
    if dest_state: params["DEST_STATE"] = dest_state.upper()
    if orig_city:  params["ORIG_CITY"]  = orig_city
    if dest_city:  params["DEST_CITY"]  = dest_city
    if eqtype:     params["EQTYPE"]     = eqtype.upper().replace(" ", "_")

    if not params:
        return {"error": "At least one search filter is required."}

    param_str = "|".join(f"{k}:{v}" for k, v in params.items())
    cmd_str = f"CMD:LOAD_QUERY|AUTH:{TMS_AUTH}|{param_str}|MAX_RESULTS:{max_results}"

    try:
        records = _tms(cmd_str, cmd="LOAD_QUERY")
    except TMSError as e:
        return {"error": f"TMS error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"error": f"TMS temporarily unavailable: {e}"}

    # Diagnostic: surface the raw STATUS tokens the board actually returns.
    # This is what revealed the empty-result bug — the board returns valid loads
    # whose STATUS is not the literal "OPEN", so an exact allowlist dropped them.
    log.info("search_loads raw statuses: %r", [r.get("STATUS", "").strip() for r in records])

    # Hide only loads the TMS has clearly reserved; keep everything else.
    # LOAD_QUERY is documented as the "open board", so a denylist of reserved
    # states is correct — an exact == "OPEN" allowlist silently zeroes results
    # whenever the wire uses any other available-token (AVAILABLE / POSTED / NEW).
    open_records = [
        r for r in records
        if r.get("STATUS", "").strip().upper() not in RESERVED_STATUSES
    ]
    if len(open_records) < len(records):
        log.info(
            "search_loads: dropped %d reserved record(s) from results",
            len(records) - len(open_records),
        )
    return {"loads": [_load_summary(r) for r in open_records], "count": len(open_records)}


@mcp.tool()
def get_load(load_id: str) -> dict:
    """
    Retrieve full details for a single load by its load ID (e.g. LD0000046112).
    Returns origin, destination, pickup/delivery times, equipment type, rate,
    weight, commodity, dimensions, and any operator notes.
    The maximum allowable rate (max_rate) is never disclosed.
    """
    try:
        records = _tms(f"CMD:LOAD_GET|AUTH:{TMS_AUTH}|LOAD_ID:{load_id}", cmd="LOAD_GET")
    except TMSError as e:
        if e.code == "UNKNOWN_LOAD":
            return {"error": f"Load {load_id} not found."}
        return {"error": f"TMS error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"error": f"TMS temporarily unavailable: {e}"}

    if not records:
        return {"error": f"Load {load_id} not found."}

    detail = _load_detail(records[0])
    detail.pop("_max_buy", None)  # never expose to callers
    return detail


@mcp.tool()
def evaluate_offer(load_id: str, carrier_rate: int, counter_round: int = 1) -> dict:
    """
    Evaluate a carrier's asking rate against the broker's private rate ceiling
    and return a negotiation decision.

    The ceiling (max_rate / MAX_BUY) is fetched and applied entirely server-side
    and is NEVER returned to the caller, so it cannot reach the carrier directly
    or indirectly — the agent only ever sees a discrete decision and, on a
    counter, a concrete price to voice.

    Recall the broker PAYS the carrier: the loadboard rate is the posted headline
    the carrier anchors to, and the ceiling sits *below* it.  Negotiation means
    pulling the carrier DOWN to at or below the ceiling.

    Args:
        load_id:       the load under negotiation (e.g. LD00274)
        carrier_rate:  the rate the carrier is asking for (USD, integer)
        counter_round: which counter round this is (1-3); paces the offer and
                       enforces the three-round walk-away limit

    Returns one of:
        {"decision": "accept",  "agreed_rate": <carrier_rate>}
            Carrier is at or below the ceiling — proceed to book_load at this rate.
        {"decision": "counter", "broker_offer": <int>, "counter_round": <int>}
            Carrier is above the ceiling — voice broker_offer as the price and
            loop.  broker_offer is always at or below the ceiling.
        {"decision": "decline"}
            Counter rounds exhausted with no agreement — close professionally and
            log as a failed negotiation; do not transfer.
    """
    # Fetch the load to retrieve MAX_BUY for ceiling enforcement — same path as
    # book_load, so behaviour stays consistent between negotiation and booking.
    try:
        records = _tms(f"CMD:LOAD_GET|AUTH:{TMS_AUTH}|LOAD_ID:{load_id}", cmd="LOAD_GET")
    except TMSError as e:
        if e.code == "UNKNOWN_LOAD":
            return {"error": f"Load {load_id} not found."}
        return {"error": f"TMS error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"error": f"TMS temporarily unavailable: {e}"}

    if not records:
        return {"error": f"Load {load_id} not found."}

    detail  = _load_detail(records[0])
    max_buy = detail.get("_max_buy")

    # If the ceiling is absent (token not flagged for MAX_BUY per spec), refuse to
    # guess rather than risk margin leakage — mirrors book_load's behaviour.
    if max_buy is None:
        log.error("MAX_BUY absent from LOAD_GET response for load=%s — cannot evaluate", load_id)
        return {"error": "Unable to verify rate ceiling for this load. Please contact dispatch."}

    # Clamp the round into [1, 3] so a stray value can't distort pacing.
    counter_round = max(1, min(3, counter_round))

    # Carrier already at or below the ceiling — take the deal at their number.
    if carrier_rate <= max_buy:
        return {"decision": "accept", "agreed_rate": carrier_rate}

    # Carrier still above the ceiling after the final round — walk away.
    if counter_round >= 3:
        return {"decision": "decline"}

    # Counter BELOW the ceiling early and approach it on the final round
    # (~0.96 / 0.98 / 1.00 of max_buy across rounds 1-3).  This preserves margin
    # instead of conceding the full ceiling on every load, and never offers above
    # it.  The agent voices broker_offer as a normal price; it is never labelled
    # as a maximum, so nothing about the ceiling is disclosed.
    offer = round(max_buy * (0.94 + 0.02 * counter_round))
    return {
        "decision":      "counter",
        "broker_offer":  min(int(offer), max_buy),
        "counter_round": counter_round,
    }


@mcp.tool()
def book_load(load_id: str, mc_number: str, agreed_rate: int) -> dict:
    """
    Book a load at an agreed rate.  Enforces the broker's rate ceiling —
    if agreed_rate exceeds the maximum allowable rate the booking is refused.
    The ceiling value is never disclosed to the caller.
    Returns a booking reference and confirmation on success.
    """
    # Step 1 — fetch load to retrieve MAX_BUY for ceiling enforcement.
    try:
        records = _tms(f"CMD:LOAD_GET|AUTH:{TMS_AUTH}|LOAD_ID:{load_id}", cmd="LOAD_GET")
    except TMSError as e:
        if e.code == "UNKNOWN_LOAD":
            return {"error": f"Load {load_id} not found."}
        return {"error": f"TMS error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"error": f"TMS temporarily unavailable:  {e}"}

    if not records:
        return {"error": f"Load {load_id} not found."}

    detail = _load_detail(records[0])
    max_buy = detail.get("_max_buy")

    # Step 2 — reject loads that are not open for booking.
    # LOAD_GET returns records regardless of STATUS (spec note), so a reserved
    # load will resolve here but fail at LOAD_BOOK with ALREADY_BOOKED.
    # Catch it early to give the agent a clear, actionable message.  Uses the
    # same denylist as search_loads so the two stay consistent.
    load_status = detail.get("status", "").upper()
    if load_status in RESERVED_STATUSES:
        log.warning(
            "Booking rejected — load reserved: load=%s status=%s",
            load_id, load_status,
        )
        return {"error": f"Load {load_id} is not available for booking (status: {load_status}). Please find another load."}

    # Step 3 — enforce rate ceiling.
    # If MAX_BUY is absent from the record (token not flagged for it per spec),
    # we refuse the booking rather than silently skipping the check — accepting
    # an unknown rate would create margin leakage risk.
    if max_buy is None:
        log.error("MAX_BUY absent from LOAD_GET response for load=%s — booking refused", load_id)
        return {"error": "Unable to verify rate ceiling for this load. Please contact dispatch."}

    if agreed_rate > max_buy:
        log.warning(
            "Rate ceiling exceeded: agreed=%d max=%d load=%s",
            agreed_rate, max_buy, load_id,
        )
        return {"error": "The agreed rate exceeds what we can offer on this load. Can you come down?"}

    # Step 4 — commit booking.
    cmd_str = (
        f"CMD:LOAD_BOOK|AUTH:{TMS_AUTH}"
        f"|LOAD_ID:{load_id}|MC_NUM:{mc_number}|AGREED_RATE:{agreed_rate}"
    )
    try:
        booking = _tms(cmd_str, cmd="LOAD_BOOK")
    except TMSError as e:
        if e.code == "ALREADY_BOOKED":
            return {"error": "This load is no longer available."}
        if e.code == "INVALID_RATE":
            return {"error": "Rate rejected by the system. Please check the agreed amount."}
        return {"error": f"TMS error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"error": f"TMS temporarily unavailable: {e}"}

    if not booking:
        return {"error": "TMS returned an empty booking response."}

    b = booking[0]
    booked_status = b.get("STATUS", "").strip()
    if booked_status.upper() != "BOOKED":
        # The live TMS has been observed returning STATUS:PENDING on a successful
        # booking (spec samples show BOOKED).  Log it for observability but do not
        # treat it as an error — the BOOKING_REF is the authoritative confirmation.
        log.warning(
            "LOAD_BOOK returned unexpected status: load=%s status=%s booking_ref=%s",
            load_id, booked_status, b.get("BOOKING_REF", "").strip(),
        )
    return {
        "load_id":     b.get("LOAD_ID", load_id).strip(),
        "booking_ref": b.get("BOOKING_REF", "").strip(),
        "status":      booked_status,
        "timestamp":   b.get("TIMESTAMP", "").strip(),
        "agreed_rate": agreed_rate,
        "mc_number":   mc_number,
    }


@mcp.tool()
def ping() -> dict:
    """
    Health check — confirms transport framing and authentication are working.
    Uses DEBUG_ECHO which bypasses TMS fault injection, so a successful
    response here does NOT guarantee operational commands are fault-free.
    Returns auth status and the number of fields parsed by the server.
    """
    msg = "HEALTHCHECK"
    cmd_str = f"CMD:DEBUG_ECHO|AUTH:{TMS_AUTH}|MSG:{msg}"
    try:
        # DEBUG_ECHO bypasses fault injection — no retry needed, but we parse
        # with cmd="" so no required-field check is applied (ECHO responses
        # have a different shape from operational responses).
        raw = _send(cmd_str)
        records = _parse(raw, cmd="DEBUG_ECHO")
    except TMSError as e:
        return {"ok": False, "error": f"TMS auth error {e.code}: {e.message}"}
    except TMSTransportError as e:
        return {"ok": False, "error": f"Transport error: {e}"}
    except Exception as e:
        return {"ok": False, "error": str(e)}

    if not records:
        return {"ok": False, "error": "Empty response from DEBUG_ECHO"}

    r = records[0]
    return {
        "ok":           r.get("AUTH", "").strip() == "OK",
        "auth":         r.get("AUTH", "").strip(),
        "fields_parsed": _int(r.get("FIELDS_PARSED", "")),
        "echo":         r.get("MSG", "").strip(),
    }


# ── entrypoint ────────────────────────────────────────────────────────────────

app = mcp.streamable_http_app()