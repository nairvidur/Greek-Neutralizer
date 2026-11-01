"""OCC option symbology (OSI): ROOT, YYMMDD, C or P, strike * 1000 in 8 digits."""

from __future__ import annotations

import re
from dataclasses import dataclass
from datetime import date

from greeks_neutralizer.domain import OptionRight
from greeks_neutralizer.errors import ValidationError

_OSI = re.compile(r"^([A-Z0-9.]{1,6})\s*(\d{6})([CP])(\d{8})$")


@dataclass(frozen=True, slots=True)
class OsiSymbol:
    root: str
    expiry: date
    right: OptionRight
    strike: float

    def format(self) -> str:
        strike_int = round(self.strike * 1000)
        return (
            f"{self.root:<6}{self.expiry:%y%m%d}"
            f"{'C' if self.right is OptionRight.CALL else 'P'}{strike_int:08d}"
        )


def parse_osi(symbol: str) -> OsiSymbol:
    """Parse padded or compact OSI. Raises ValidationError rather than guessing."""
    match = _OSI.match(symbol.strip().upper())
    if match is None:
        raise ValidationError(f"not an OSI option symbol: {symbol!r}")
    root, ymd, cp, strike = match.groups()
    try:
        expiry = date(2000 + int(ymd[:2]), int(ymd[2:4]), int(ymd[4:6]))
    except ValueError as exc:
        raise ValidationError(f"invalid expiry in OSI symbol {symbol!r}: {exc}") from None
    return OsiSymbol(
        root=root,
        expiry=expiry,
        right=OptionRight.CALL if cp == "C" else OptionRight.PUT,
        strike=int(strike) / 1000.0,
    )


# Index roots with AM settlement (standard monthly SPX and similar) use a different root from the
# PM-settled weeklies, which is why the root rather than the date decides the expiry time here.
AM_SETTLED_ROOTS = frozenset({"SPX", "NDX", "RUT", "VIX", "DJX"})
PM_SETTLED_INDEX_ROOTS = frozenset({"SPXW", "NDXP", "RUTW", "XSP", "MRUT", "VIXW"})
CASH_SETTLED_ROOTS = AM_SETTLED_ROOTS | PM_SETTLED_INDEX_ROOTS


def is_am_settled(root: str) -> bool:
    return root.upper() in AM_SETTLED_ROOTS


def is_cash_settled_european(root: str) -> bool:
    return root.upper() in CASH_SETTLED_ROOTS
