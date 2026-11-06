"""A historical option market from end-of-day chain files: real quotes and real spreads.

Each day is built from that day's chain file, the rate known that day, and dividends projected from
those already paid, so nothing later is read. Tested in tests/test_chain_market.py.

What is real: option bid and ask, the rate, and past dividends. Spot is implied by put-call
parity in the same quotes, and the underlying's close is a check on it.
What is assumed, because the free source does not carry it:
- Quote sizes: every option shows `displayed_size` a side, which caps a hedge leg at that size.
- Fees: `option_fee` per contract. The stock trades at `stock_half_spread` with no fee.
- The next dividends: each pays the last known amount on SPY's quarterly schedule, the third
  Friday of March, June, September, and December. Amounts are not known in advance.
- Marks for options the source did not quote that day: the model, from that day's calibrated
  surface. Each such mark is counted by the engine and reported.

Settlement is simplified. An expiring option is marked at intrinsic and removed, as if
cash-settled. Early exercise and assignment are not modeled, which matters most for short calls
just before an ex-date. The strangle book rolls 10 days before expiry, so it rarely gets there.
"""

from __future__ import annotations

import csv
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, datetime, time, timedelta
from pathlib import Path

import numpy as np
from scipy.optimize import least_squares

from greeks_neutralizer.backtest.engine import DayMarket
from greeks_neutralizer.backtest.proxy import third_friday
from greeks_neutralizer.data.chain import ChainQuote, ParityForward, calibrate_chain
from greeks_neutralizer.data.quality import QualityPolicy
from greeks_neutralizer.domain import (
    NEW_YORK,
    ExerciseStyle,
    Instrument,
    LinearInstrument,
    OptionContract,
    OptionRight,
    Position,
    Quote,
    SettlementKind,
)
from greeks_neutralizer.errors import GNError, MarketDataError, ValidationError
from greeks_neutralizer.hedging.problem import HedgeCandidate
from greeks_neutralizer.market import (
    CashDividend,
    DiscountCurve,
    MarketSnapshot,
    UnderlyingMarket,
)
from greeks_neutralizer.pricing.black import FloatArray
from greeks_neutralizer.pricing.book import value_options
from greeks_neutralizer.time_model import calendar_years, us_expiry

# A carry yield past this, after projected dividends, says the chain's forwards are not usable.
_MAX_IMPLIED_CARRY = 0.15
# A parity spot this far from the close says the files do not describe the same market.
_MAX_CLOSE_GAP = 0.05
# implied_forward's default strike window around spot, and a margin for the close's gap to it.
_PARITY_WINDOW = 0.05
_PARITY_WINDOW_MARGIN = 0.01


@dataclass(frozen=True, slots=True)
class ChainRow:
    expiry: date
    strike: float
    right: OptionRight
    bid: float
    ask: float


@dataclass(frozen=True, slots=True)
class ChainDay:
    day: date
    close: float
    rows: tuple[ChainRow, ...]


def load_chain_day(path: Path) -> ChainDay:
    """One day's file from scripts/fetch_dolthub_chains.py. Rows without a two-sided, ordered
    market are dropped here, since nothing downstream could use them."""
    rows: list[ChainRow] = []
    closes: set[str] = set()
    days: set[str] = set()
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            days.add(raw["date"])
            closes.add(raw["underlying_close"])
            if raw["bid"] in ("", None) or raw["ask"] in ("", None):
                continue
            bid, ask = float(raw["bid"]), float(raw["ask"])
            if not (np.isfinite(bid) and np.isfinite(ask)) or bid < 0.0 or ask <= 0.0 or ask < bid:
                continue
            right = {"call": OptionRight.CALL, "put": OptionRight.PUT}.get(raw["call_put"].lower())
            if right is None:
                raise MarketDataError(f"{path}: unknown right {raw['call_put']!r}")
            rows.append(
                ChainRow(
                    date.fromisoformat(raw["expiration"]), float(raw["strike"]), right, bid, ask
                )
            )
    if len(days) != 1 or len(closes) != 1:
        raise MarketDataError(f"{path}: expected one date and one close, got {days} and {closes}")
    day = date.fromisoformat(days.pop())
    if path.stem != day.isoformat():
        raise MarketDataError(f"{path}: file name does not match its date {day}")
    close = float(closes.pop())
    if not np.isfinite(close) or close <= 0.0:
        raise MarketDataError(f"{path}: close {close} is not a price")
    return ChainDay(day, close, tuple(rows))


def load_dividends(path: Path) -> tuple[tuple[date, float], ...]:
    out: list[tuple[date, float]] = []
    with path.open(newline="") as handle:
        for raw in csv.DictReader(handle):
            out.append((date.fromisoformat(raw["ex_date"]), float(raw["amount"])))
    return tuple(sorted(out))


def projected_dividends(
    paid: Sequence[tuple[date, float]], day: date, horizon_days: int = 400
) -> tuple[CashDividend, ...]:
    """Dividends after `day` as known on `day`: the last amount paid by then, on each quarterly
    third Friday within the horizon. Paid dividends after `day` are never read."""
    known = [amount for ex, amount in paid if ex <= day]
    if not known:
        return ()
    amount = known[-1]
    out: list[CashDividend] = []
    year = day.year
    while len(out) < 8:
        for month in (3, 6, 9, 12):
            ex = third_friday(year, month)
            if day < ex <= day + timedelta(days=horizon_days):
                out.append(CashDividend(_close_of(ex), amount))
        year += 1
        if date(year, 1, 1) > day + timedelta(days=horizon_days):
            break
    return tuple(out)


def _close_of(day: date) -> datetime:
    return datetime.combine(day, time(16, 0), tzinfo=NEW_YORK)


def rate_on(rates_pct: dict[date, float], day: date) -> float | None:
    """The last rate published on or before `day`, as a decimal. Never a later one."""
    known = [d for d in rates_pct if d <= day]
    return rates_pct[max(known)] / 100.0 if known else None


def implied_spot_and_carry(
    forwards: Sequence[ParityForward],
    dividends: Sequence[CashDividend],
    curve: DiscountCurve,
    as_of: datetime,
) -> tuple[float, float]:
    """Spot and carry yield that reproduce the parity forwards, given the projected dividends.

    The model's forward is (S - PV of dividends to T) exp(-q T) / D, so each expiry gives
    F D exp(q T) + PV = S. With one expiry q is 0. With more, S and q are fitted by least
    squares in log terms. Raises if the fitted carry is implausible, which means the
    forwards disagree.
    """
    if not forwards:
        raise MarketDataError("no parity forwards")
    T = np.array([f.T for f in forwards])
    fd = np.array([f.forward * f.discount for f in forwards])
    pv = np.array(
        [
            sum(
                d.amount * float(curve.discount(calendar_years(as_of, d.ex_date)))
                for d in dividends
                if as_of < d.ex_date <= f.expiry
            )
            for f in forwards
        ]
    )
    q = 0.0
    spot = float(fd[0] + pv[0])
    if len(forwards) > 1:

        def residuals(x: FloatArray) -> FloatArray:
            escrowed = np.maximum(x[0] - pv, 1e-12)
            return np.asarray(np.log(fd) - np.log(escrowed) + x[1] * T, dtype=np.float64)

        fit = least_squares(residuals, np.array([spot, 0.0]), xtol=1e-14, ftol=1e-14)
        if not fit.success or np.any(fit.x[0] - pv <= 0.0):
            raise MarketDataError(f"parity forwards do not fit a spot and carry: {fit.message}")
        spot, q = float(fit.x[0]), float(fit.x[1])
    if not np.isfinite(spot) or spot <= 0.0 or not np.isfinite(q) or abs(q) > _MAX_IMPLIED_CARRY:
        raise MarketDataError(f"parity forwards imply spot {spot:.2f} and carry {q:.3f}")
    return spot, q


@dataclass(frozen=True, slots=True)
class ChainDayMarket(DayMarket):
    """A day of a historical chain. `listed` is every option quoted that day, for the book."""

    listed: tuple[OptionContract, ...] = ()


@dataclass(frozen=True, slots=True)
class MarkAudit:
    """Per held-out contract: (model value - quoted mid, quoted half-spread), in dollars a share."""

    day: date
    strike_holdout: tuple[tuple[float, float], ...]
    expiry_holdout: tuple[tuple[float, float], ...]


@dataclass(frozen=True, slots=True)
class SkippedDay:
    day: date
    reason: str


@dataclass(slots=True)
class ChainMarket:
    """Iterates `ChainDayMarket`s over chain files in date order.

    Spot comes from put-call parity, not the close. The source's chain snapshot and its close
    need not be at the same moment, and on a fast day they differ by more than a percent, so the
    stock hedge trades at the price the options imply. The close is kept as a check.

    A day whose chain cannot be calibrated is skipped and recorded, and the next P&L period spans
    it, since on such a day no hedge could have been priced from that chain either.
    `modeled_quotes` records each time a trade or hedge had to be priced without a real quote.
    """

    symbol: str
    files: tuple[Path, ...]
    rates_pct: dict[date, float]
    dividends_paid: tuple[tuple[date, float], ...]
    spread_multiple: float = 1.0
    displayed_size: int = 500
    option_fee: float = 0.50
    stock_half_spread: float = 0.01
    min_expiry_days: int = 7
    n_hedge_expiries: int = 2
    strike_grid_stdevs: tuple[float, ...] = (-2.0, -1.25, -0.5, 0.5, 1.25, 2.0)
    skipped: list[SkippedDay] = field(default_factory=list)
    modeled_quotes: list[str] = field(default_factory=list)
    # Parity spot over the close, minus one, for each day built.
    close_gaps: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if not self.files:
            raise ValidationError("no chain files")
        if self.spread_multiple <= 0.0:
            raise ValidationError("spread_multiple must be positive")
        if self.displayed_size < 1:
            raise ValidationError("displayed_size must be at least 1")
        self.files = tuple(sorted(self.files, key=lambda p: p.stem))

    def __iter__(self) -> Iterator[ChainDayMarket]:
        for path in self.files:
            chain = load_chain_day(path)
            try:
                yield self.day(chain)
            except GNError as exc:
                self.skipped.append(SkippedDay(chain.day, f"{type(exc).__name__}: {exc}"))

    def _contract(self, row: ChainRow) -> OptionContract:
        return OptionContract(
            self.symbol,
            row.right,
            row.strike,
            us_expiry(row.expiry),
            ExerciseStyle.AMERICAN,
            settlement=SettlementKind.PHYSICAL,
        )

    def _quotes(self, chain: ChainDay, as_of: datetime) -> dict[str, tuple[OptionContract, Quote]]:
        quotes: dict[str, tuple[OptionContract, Quote]] = {}
        for row in chain.rows:
            if row.expiry <= chain.day:
                continue
            contract = self._contract(row)
            size = self.displayed_size
            quotes[contract.key] = (contract, Quote(row.bid, row.ask, size, size, as_of))
        if not quotes:
            raise MarketDataError(f"{chain.day}: no live quotes")
        return quotes

    def _market(
        self, quotes: Sequence[tuple[OptionContract, Quote]], chain: ChainDay, as_of: datetime
    ) -> tuple[UnderlyingMarket, float]:
        """The calibrated market from these quotes, and the parity spot's gap to the close."""
        rate = rate_on(self.rates_pct, chain.day)
        if rate is None:
            raise MarketDataError(f"{chain.day}: no rate published yet")
        calibration = calibrate_chain(
            [ChainQuote(c, q) for c, q in quotes],
            as_of,
            chain.close,
            discount_rate=rate,
            policy=QualityPolicy(max_age=timedelta(days=1), require_size=False),
        )
        dividends = projected_dividends(self.dividends_paid, chain.day)
        curve = DiscountCurve.flat(rate)
        spot, carry = implied_spot_and_carry(calibration.forwards, dividends, curve, as_of)
        gap = spot / chain.close - 1.0
        if abs(gap) > _MAX_CLOSE_GAP:
            raise MarketDataError(
                f"{chain.day}: parity spot {spot:.2f} is {gap:+.1%} from the close {chain.close}"
            )
        market = UnderlyingMarket(self.symbol, spot, curve, calibration.surface, carry, dividends)
        return market, gap

    def day(self, chain: ChainDay) -> ChainDayMarket:
        as_of = _close_of(chain.day)
        quotes = self._quotes(chain, as_of)
        market, gap = self._market(list(quotes.values()), chain, as_of)
        self.close_gaps.append(gap)
        spot = market.spot
        snapshot = MarketSnapshot(as_of, {self.symbol: market})
        linear = LinearInstrument(self.symbol, self.symbol)
        listed = tuple(contract for contract, _ in quotes.values())
        typical_half = _typical_half_spread([q for _, q in quotes.values()])

        def observed_mid(contract: OptionContract) -> float | None:
            found = quotes.get(contract.key)
            return found[1].mid if found else None

        def quote_for(instrument: Instrument) -> HedgeCandidate:
            if isinstance(instrument, LinearInstrument):
                half = self.stock_half_spread * self.spread_multiple
                quote = Quote(spot - half, spot + half, 1_000_000, 1_000_000, as_of)
                return HedgeCandidate(instrument, quote)
            if not isinstance(instrument, OptionContract):
                raise MarketDataError(f"cannot quote {instrument!r}")
            found = quotes.get(instrument.key)
            if found is not None:
                return HedgeCandidate(instrument, self._widened(found[1]), self.option_fee)
            self.modeled_quotes.append(f"{chain.day}: {instrument.key}")
            return HedgeCandidate(
                instrument, self._modeled_quote(instrument, snapshot, typical_half), self.option_fee
            )

        candidates = [quote_for(linear)]
        for expiry in _hedge_expiries(listed, as_of, self.min_expiry_days, self.n_hedge_expiries):
            for contract in _grid(listed, expiry, market, as_of, self.strike_grid_stdevs):
                candidates.append(quote_for(contract))
        return ChainDayMarket(
            as_of, snapshot, tuple(candidates), quote_for, observed_mid, listed=listed
        )

    def mark_audit(self, chain: ChainDay) -> MarkAudit:
        """Out-of-sample error of the marks the backtest uses for contracts it has no quote for.

        Two holdouts, each recalibrated without the held-out quotes and then priced against them:
        alternate strikes within every expiry, outside the strikes parity needs, which is how a
        held contract is marked when its expiry is quoted but its strike is not; and the middle
        expiry, which is how it is marked
        when its expiry is not quoted at all. Only OTM contracts with a mid of at least 0.05 are
        scored, since those are what the books and hedges hold.
        """
        as_of = _close_of(chain.day)
        quotes = list(self._quotes(chain, as_of).values())
        full, _ = self._market(quotes, chain, as_of)
        expiries = sorted({c.expiry for c, _ in quotes})

        def otm(contract: OptionContract, quote: Quote) -> bool:
            forward = full.forward(as_of, contract.expiry)
            above = contract.strike >= forward
            return quote.mid >= 0.05 and (contract.right is OptionRight.CALL) == above

        strike_out: set[str] = set()
        for expiry in expiries:
            strikes = sorted({c.strike for c, _ in quotes if c.expiry == expiry})
            # Interior strikes only, so a mark is never an extrapolation, and only outside the
            # strikes parity uses, so the held-out fit still has its forward.
            held = {
                k
                for k in strikes[1:-1:2]
                if abs(k / full.spot - 1.0) > _PARITY_WINDOW + _PARITY_WINDOW_MARGIN
            }
            strike_out |= {c.key for c, _ in quotes if c.expiry == expiry and c.strike in held}
        expiry_out: set[str] = set()
        if len(expiries) >= 3:
            middle = expiries[len(expiries) // 2]
            expiry_out = {c.key for c, _ in quotes if c.expiry == middle}

        def score(out: set[str]) -> tuple[tuple[float, float], ...]:
            if not out:
                return ()
            kept = [(c, q) for c, q in quotes if c.key not in out]
            market, _ = self._market(kept, chain, as_of)
            snapshot = MarketSnapshot(as_of, {self.symbol: market})
            scored = [(c, q) for c, q in quotes if c.key in out and otm(c, q)]
            if not scored:
                return ()
            prices = value_options([c for c, _ in scored], snapshot).greeks.price
            return tuple(
                (float(price) - q.mid, 0.5 * (q.ask - q.bid))
                for (_, q), price in zip(scored, prices, strict=True)
            )

        return MarkAudit(chain.day, score(strike_out), score(expiry_out))

    def _widened(self, quote: Quote) -> Quote:
        # The bid floor at zero charges a cheap wing less than spread_multiple times its
        # half-spread, so stressed option-hedge costs are a lower bound.
        if self.spread_multiple == 1.0:
            return quote
        half = 0.5 * (quote.ask - quote.bid) * self.spread_multiple
        return Quote(
            max(quote.mid - half, 0.0),
            quote.mid + half,
            quote.bid_size,
            quote.ask_size,
            quote.timestamp,
        )

    def _modeled_quote(
        self, contract: OptionContract, snapshot: MarketSnapshot, typical_half: float
    ) -> Quote:
        as_of = snapshot.as_of
        if contract.expiry <= as_of:
            spot = snapshot.market(contract.underlying).spot
            value = max(contract.right.omega * (spot - contract.strike), 0.0)
        else:
            value = float(value_options([contract], snapshot).greeks.price[0])
        half = typical_half * self.spread_multiple
        return Quote(
            max(value - half, 0.0), value + half, self.displayed_size, self.displayed_size, as_of
        )


def _typical_half_spread(quotes: Sequence[Quote]) -> float:
    """Median half-spread of the day's two-sided quotes, for pricing a contract it lacks."""
    halves = [0.5 * (q.ask - q.bid) for q in quotes if q.bid > 0.0]
    return float(np.median(halves)) if halves else 0.05


def is_monthly(expiry: datetime) -> bool:
    """A standard monthly: the third Friday, or the Thursday before it when Friday is a holiday."""
    third = third_friday(expiry.year, expiry.month)
    return expiry.date() in (third, third - timedelta(days=1))


def _hedge_expiries(
    listed: Sequence[OptionContract], as_of: datetime, min_days: int, count: int
) -> list[datetime]:
    """Listed monthlies first, as the proxy's grid uses monthlies, then the nearest others."""
    expiries = sorted(
        {c.expiry for c in listed if (c.expiry - as_of).days >= min_days},
        key=lambda e: (not is_monthly(e), e),
    )
    return sorted(expiries[:count])


def _grid(
    listed: Sequence[OptionContract],
    expiry: datetime,
    market: UnderlyingMarket,
    as_of: datetime,
    stdevs: Sequence[float],
) -> list[OptionContract]:
    """For each grid point, the listed OTM contract nearest that many stdevs from the forward."""
    tau = calendar_years(as_of, expiry)
    forward = market.forward(as_of, expiry)
    atm = float(market.surface.vol(np.array([0.0]), np.array([tau]))[0])
    stdev = atm * np.sqrt(tau)
    out: list[OptionContract] = []
    for z in stdevs:
        right = OptionRight.PUT if z < 0.0 else OptionRight.CALL
        pool = [c for c in listed if c.expiry == expiry and c.right is right]
        if not pool:
            continue
        target = forward * np.exp(z * stdev)
        best = min(pool, key=lambda c: (abs(c.strike - target), c.strike))
        if best not in out:
            out.append(best)
    return out


def nearest_listed(
    day: ChainDayMarket, expiry: datetime, right: OptionRight, strike: float
) -> OptionContract:
    pool = [c for c in day.listed if c.expiry == expiry and c.right is right]
    if not pool:
        raise MarketDataError(f"no {right.value} listed at {expiry.date()}")
    return min(pool, key=lambda c: (abs(c.strike - strike), c.strike))


def _roll_expiry(day: ChainDayMarket, roll_days: int) -> datetime:
    """The first listed monthly more than roll_days + 5 days out, as the proxy books choose, or
    the first listed expiry past that when no monthly is listed."""
    past = sorted(
        {c.expiry for c in day.listed if (c.expiry.date() - day.as_of.date()).days > roll_days + 5}
    )
    if not past:
        raise MarketDataError(f"{day.as_of.date()}: no listed expiry past {roll_days + 5} days")
    monthly = [e for e in past if is_monthly(e)]
    return monthly[0] if monthly else past[0]


def _roll(
    day: DayMarket, holdings: Mapping[str, Position], roll_days: int
) -> list[Position] | None:
    """None while every live option has more than roll_days left. Otherwise the live options to
    close, which is empty when nothing is live, as on the first day or after an expiry."""
    live = [
        p
        for p in holdings.values()
        if isinstance(p.instrument, OptionContract) and p.instrument.expiry > day.as_of
    ]
    if live and all(
        calendar_years(day.as_of, p.instrument.expiry) * 365.0 > roll_days
        for p in live
        if isinstance(p.instrument, OptionContract)
    ):
        return None
    return live


def _atm_stdev(day: DayMarket, symbol: str, expiry: datetime) -> tuple[float, float]:
    market = day.snapshot.market(symbol)
    tau = calendar_years(day.as_of, expiry)
    atm = float(market.surface.vol(np.array([0.0]), np.array([tau]))[0])
    return market.forward(day.as_of, expiry), atm * float(np.sqrt(tau))


@dataclass(slots=True)
class ChainShortStrangle:
    """The proxy's short strangle on listed contracts: about +/-1 stdev, rolled at `roll_days`.

    Trades only contracts quoted that day, at the listed strike nearest each target.
    """

    symbol: str
    lots: int = 100
    roll_days: int = 10
    stdevs: float = 1.0

    def trades(self, day: DayMarket, holdings: Mapping[str, Position]) -> Sequence[Position]:
        if not isinstance(day, ChainDayMarket):
            raise ValidationError("ChainShortStrangle needs a ChainDayMarket")
        live = _roll(day, holdings, self.roll_days)
        if live is None:
            return ()
        expiry = _roll_expiry(day, self.roll_days)
        forward, stdev = _atm_stdev(day, self.symbol, expiry)
        put = nearest_listed(day, expiry, OptionRight.PUT, forward * np.exp(-self.stdevs * stdev))
        call = nearest_listed(day, expiry, OptionRight.CALL, forward * np.exp(self.stdevs * stdev))
        return [
            *(Position(p.instrument, -p.quantity) for p in live),
            Position(put, -self.lots),
            Position(call, -self.lots),
        ]


@dataclass(slots=True)
class ChainPutOverlay:
    """The proxy's overlay on listed contracts: long shares with a monthly protective put."""

    symbol: str
    notional_shares: int = 100_000
    put_stdevs: float = 1.5
    roll_days: int = 10

    def trades(self, day: DayMarket, holdings: Mapping[str, Position]) -> Sequence[Position]:
        if not isinstance(day, ChainDayMarket):
            raise ValidationError("ChainPutOverlay needs a ChainDayMarket")
        trades: list[Position] = []
        linear = LinearInstrument(self.symbol, self.symbol)
        if linear.key not in holdings:
            trades.append(Position(linear, self.notional_shares))
        live = _roll(day, holdings, self.roll_days)
        if live is None:
            return trades
        expiry = _roll_expiry(day, self.roll_days)
        forward, stdev = _atm_stdev(day, self.symbol, expiry)
        put = nearest_listed(
            day, expiry, OptionRight.PUT, forward * np.exp(-self.put_stdevs * stdev)
        )
        lots = max(self.notional_shares // 100, 1)
        return [*trades, *(Position(p.instrument, -p.quantity) for p in live), Position(put, lots)]


__all__ = [
    "ChainDay",
    "ChainDayMarket",
    "ChainMarket",
    "ChainPutOverlay",
    "ChainRow",
    "ChainShortStrangle",
    "MarkAudit",
    "SkippedDay",
    "implied_spot_and_carry",
    "is_monthly",
    "load_chain_day",
    "load_dividends",
    "nearest_listed",
    "projected_dividends",
    "rate_on",
]
