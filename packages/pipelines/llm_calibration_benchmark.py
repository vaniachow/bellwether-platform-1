"""
LLM Calibration Benchmark Against Prediction Markets

Evaluates how well LLMs estimate probabilities by comparing their estimates
to cross-platform matched market prices (Bellwether VWAP).

Usage:
    python llm_calibration_benchmark.py              # Run full benchmark
    python llm_calibration_benchmark.py --dry-run    # Print prompts, no API calls
    python llm_calibration_benchmark.py --model gpt-4o-mini  # Use a different model
"""

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from config import BASE_DIR, get_openai_api_key, atomic_write_json
from category_utils import old_to_new_category, CATEGORY_DISPLAY_NAMES, CATEGORY_COLORS

# ── Constants ──────────────────────────────────────────────────────────────────

ACTIVE_MARKETS_PATH = BASE_DIR / "packages" / "docs" / "data" / "active_markets.json"
OUTPUT_PATH = BASE_DIR / "packages" / "docs" / "data" / "llm_calibration.json"

MAX_CONCURRENT = 30
MAX_RETRIES = 3
RETRY_BASE_DELAY = 2.0
INTER_REQUEST_DELAY = 0.05  # 50ms stagger

SYSTEM_PROMPT = (
    "You are a probability estimator. Given a prediction market question, "
    "estimate the probability that the event occurs. Respond with ONLY a "
    "number between 0 and 100 representing your probability estimate as a "
    "percentage. Do not include any other text, explanation, or the % sign."
)

USER_PROMPT_TEMPLATE = (
    "What is the probability that the following event occurs?\n\n"
    "Event: \"{label}\"\n\n"
    "Today's date: {today}\n\n"
    "Respond with ONLY a number between 0 and 100."
)


def safe_print(msg: str):
    """Print with fallback for Windows encoding issues."""
    try:
        print(msg)
    except UnicodeEncodeError:
        print(msg.encode("ascii", errors="replace").decode("ascii"))


# ── Data Loading ───────────────────────────────────────────────────────────────


def build_best_label(m: dict) -> str:
    """Build the most informative label for prompting.

    Many matched markets have generic labels like "Who will win the
    governorship in California?" that are identical across candidates.
    We prefer the Kalshi question (usually more specific) or the Polymarket
    question, falling back to the generic label.  We also try to extract
    candidate info from the ticker when the label is ambiguous.
    """
    # Prefer the platform-specific questions (usually more specific)
    k_q = (m.get("k_question") or "").strip()
    pm_q = (m.get("pm_question") or "").strip()
    label = (m.get("label") or "").strip()

    # Pick the longest non-empty question as the most descriptive
    best = max([k_q, pm_q, label], key=len) if any([k_q, pm_q, label]) else label

    # If the best label is generic (e.g. "Who will win the governorship...")
    # try to extract candidate info from the ticker
    ticker = m.get("ticker", "") or m.get("key", "")
    if best == label and ("Who will win" in best or "will win" in best.lower()):
        # Extract candidate slug from ticker like BWR-BIANCO-WIN-GOV_CA-...
        parts = ticker.replace("BWR-", "").split("-")
        if len(parts) >= 2:
            candidate = parts[0].replace("_", " ").title()
            # Only add if it looks like a name (not a generic code)
            if len(candidate) > 2 and candidate.upper() != candidate:
                best = f"Will {candidate} win? ({best})"

    return best


def load_matched_markets() -> list[dict]:
    """Load cross-platform matched markets from active_markets.json."""
    with open(ACTIVE_MARKETS_PATH, "r", encoding="utf-8") as f:
        data = json.load(f)

    markets = []
    for m in data["markets"]:
        if not m.get("has_both"):
            continue
        pm_price = m.get("pm_price")
        k_price = m.get("k_price")
        if pm_price is None or k_price is None:
            continue

        # Normalize category
        raw_cat = m.get("category", "MISC")
        cat_code = old_to_new_category(raw_cat)
        cat_display = CATEGORY_DISPLAY_NAMES.get(cat_code, raw_cat)

        # Volume-weighted average price (true VWAP)
        pm_vol = m.get("pm_volume") or 0
        k_vol = m.get("k_volume") or 0
        total_vol = pm_vol + k_vol
        if total_vol > 0:
            vwap = (pm_price * pm_vol + k_price * k_vol) / total_vol
        else:
            vwap = (pm_price + k_price) / 2.0

        best_label = build_best_label(m)

        markets.append({
            "ticker": m.get("ticker", m.get("key", "")),
            "label": best_label,
            "original_label": m.get("label", ""),
            "category": cat_code,
            "category_display": cat_display,
            "pm_price": pm_price,
            "k_price": k_price,
            "market_price": round(vwap, 6),
            "total_volume": m.get("total_volume", 0),
            "pm_url": m.get("pm_url"),
            "k_url": m.get("k_url"),
        })

    # Sort by volume descending
    markets.sort(key=lambda x: x["total_volume"], reverse=True)
    return markets


# ── LLM Querying ──────────────────────────────────────────────────────────────


def parse_llm_response(text: str) -> float | None:
    """Parse a probability estimate from LLM response text."""
    text = text.strip()
    # Try to extract a number (handles "45", "45%", "0.45", "45.5%")
    match = re.search(r"(\d+(?:\.\d+)?)\s*%?", text)
    if not match:
        return None
    value = float(match.group(1))
    # If value is between 0 and 1 (exclusive of boundaries that could be percentages)
    # treat as already a probability
    if 0 < value < 1:
        return value
    # Otherwise treat as percentage
    if 0 <= value <= 100:
        return value / 100.0
    return None


async def query_single_market(
    client,
    market: dict,
    idx: int,
    total: int,
    model: str,
    semaphore: asyncio.Semaphore,
    today: str,
) -> dict:
    """Query LLM for a single market's probability estimate."""
    prompt = USER_PROMPT_TEMPLATE.format(label=market["label"], today=today)

    async with semaphore:
        await asyncio.sleep(INTER_REQUEST_DELAY * (idx % 5))
        last_error = None

        for attempt in range(MAX_RETRIES + 1):
            try:
                response = await client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": prompt},
                    ],
                    temperature=0,
                    max_tokens=10,
                )
                text = response.choices[0].message.content
                estimate = parse_llm_response(text)

                if estimate is not None:
                    if (idx + 1) % 20 == 0 or idx == total - 1:
                        safe_print(f"  [{idx + 1}/{total}] {market['label'][:60]}... -> {estimate:.1%}")
                    return {**market, "llm_estimate": round(estimate, 6), "raw_response": text}
                else:
                    safe_print(f"  [{idx + 1}/{total}] PARSE ERROR: '{text}' for {market['ticker']}")
                    return {**market, "llm_estimate": None, "raw_response": text, "error": "parse_error"}

            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                is_retryable = any(k in error_str for k in ["429", "rate limit", "timeout", "500", "502", "503", "connection"])

                if is_retryable and attempt < MAX_RETRIES:
                    delay = RETRY_BASE_DELAY * (2 ** attempt)
                    safe_print(f"  [{idx + 1}/{total}] Retry in {delay:.0f}s (attempt {attempt + 1})...")
                    await asyncio.sleep(delay)
                    continue

                safe_print(f"  [{idx + 1}/{total}] ERROR: {e}")
                return {**market, "llm_estimate": None, "error": str(last_error)}


async def run_benchmark(markets: list[dict], model: str) -> list[dict]:
    """Query LLM for all markets concurrently."""
    from openai import AsyncOpenAI

    client = AsyncOpenAI(api_key=get_openai_api_key())
    semaphore = asyncio.Semaphore(MAX_CONCURRENT)
    today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
    total = len(markets)

    safe_print(f"\nQuerying {model} for {total} markets (max {MAX_CONCURRENT} concurrent)...")

    tasks = [
        query_single_market(client, m, i, total, model, semaphore, today)
        for i, m in enumerate(markets)
    ]

    results = []
    for coro in asyncio.as_completed(tasks):
        result = await coro
        results.append(result)

    # Sort back by volume
    results.sort(key=lambda x: x.get("total_volume", 0), reverse=True)
    return results


# ── Metrics Computation ───────────────────────────────────────────────────────


def compute_metrics(results: list[dict]) -> dict:
    """Compute calibration metrics from benchmark results."""
    # Filter to successful estimates
    valid = [r for r in results if r.get("llm_estimate") is not None]
    if not valid:
        return {"error": "No valid estimates"}

    market_prices = np.array([r["market_price"] for r in valid])
    llm_estimates = np.array([r["llm_estimate"] for r in valid])
    errors = llm_estimates - market_prices
    volumes = np.array([r["total_volume"] for r in valid])

    # ── Overall metrics ──
    overall = {
        "n_markets": len(valid),
        "n_errors": len(results) - len(valid),
        "mae": round(float(np.mean(np.abs(errors))), 4),
        "rmse": round(float(np.sqrt(np.mean(errors ** 2))), 4),
        "bias": round(float(np.mean(errors)), 4),
        "median_abs_error": round(float(np.median(np.abs(errors))), 4),
        "correlation": round(float(np.corrcoef(market_prices, llm_estimates)[0, 1]), 4)
            if len(valid) > 1 else None,
    }

    # ── By category ──
    by_category = []
    categories = sorted(set(r["category"] for r in valid))
    for cat in categories:
        cat_results = [r for r in valid if r["category"] == cat]
        cat_prices = np.array([r["market_price"] for r in cat_results])
        cat_llm = np.array([r["llm_estimate"] for r in cat_results])
        cat_errors = cat_llm - cat_prices
        cat_display = CATEGORY_DISPLAY_NAMES.get(cat, cat)
        by_category.append({
            "category": cat,
            "category_display": cat_display,
            "color": CATEGORY_COLORS.get(cat, "#6b7280"),
            "n": len(cat_results),
            "mae": round(float(np.mean(np.abs(cat_errors))), 4),
            "rmse": round(float(np.sqrt(np.mean(cat_errors ** 2))), 4),
            "bias": round(float(np.mean(cat_errors)), 4),
        })
    by_category.sort(key=lambda x: x["n"], reverse=True)

    # ── By liquidity tier ──
    volume_quartiles = np.percentile(volumes, [25, 50, 75])
    tier_labels = ["fragile", "thin", "moderate", "liquid"]
    by_liquidity = []
    for i, label in enumerate(tier_labels):
        if i == 0:
            mask = volumes <= volume_quartiles[0]
        elif i == 1:
            mask = (volumes > volume_quartiles[0]) & (volumes <= volume_quartiles[1])
        elif i == 2:
            mask = (volumes > volume_quartiles[1]) & (volumes <= volume_quartiles[2])
        else:
            mask = volumes > volume_quartiles[2]

        tier_errors = errors[mask]
        if len(tier_errors) == 0:
            continue
        by_liquidity.append({
            "tier": label,
            "n": int(mask.sum()),
            "volume_range": [round(float(volumes[mask].min()), 0), round(float(volumes[mask].max()), 0)],
            "mae": round(float(np.mean(np.abs(tier_errors))), 4),
            "rmse": round(float(np.sqrt(np.mean(tier_errors ** 2))), 4),
            "bias": round(float(np.mean(tier_errors)), 4),
        })

    # ── Calibration curve (10 bins by market price) ──
    bin_edges = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
    calibration_curve = []
    for j in range(len(bin_edges) - 1):
        lo, hi = bin_edges[j], bin_edges[j + 1]
        if j < len(bin_edges) - 2:
            mask = (market_prices >= lo) & (market_prices < hi)
        else:
            mask = (market_prices >= lo) & (market_prices <= hi)
        if mask.sum() == 0:
            continue
        calibration_curve.append({
            "bin_lo": lo,
            "bin_hi": hi,
            "bin_midpoint": round((lo + hi) / 2, 2),
            "mean_market": round(float(market_prices[mask].mean()), 4),
            "mean_llm": round(float(llm_estimates[mask].mean()), 4),
            "mae": round(float(np.mean(np.abs(errors[mask]))), 4),
            "n": int(mask.sum()),
        })

    # ── Market-level results (for scatter plot / table) ──
    market_results = []
    for r in valid:
        market_results.append({
            "ticker": r["ticker"],
            "label": r.get("original_label", r["label"]),
            "prompt_label": r["label"],
            "category": r["category"],
            "category_display": r["category_display"],
            "market_price": r["market_price"],
            "llm_estimate": r["llm_estimate"],
            "error": round(r["llm_estimate"] - r["market_price"], 4),
            "abs_error": round(abs(r["llm_estimate"] - r["market_price"]), 4),
            "volume": round(r["total_volume"], 0),
        })

    return {
        "overall": overall,
        "by_category": by_category,
        "by_liquidity": by_liquidity,
        "calibration_curve": calibration_curve,
        "markets": market_results,
    }


# ── Main ──────────────────────────────────────────────────────────────────────


def main():
    parser = argparse.ArgumentParser(description="LLM Calibration Benchmark")
    parser.add_argument("--dry-run", action="store_true", help="Print prompts without calling API")
    parser.add_argument("--model", default="gpt-4o", help="OpenAI model to use (default: gpt-4o)")
    args = parser.parse_args()

    print("=" * 70)
    print("LLM CALIBRATION BENCHMARK")
    print("=" * 70)

    # Load markets
    safe_print(f"\nLoading matched markets from {ACTIVE_MARKETS_PATH.name}...")
    markets = load_matched_markets()
    safe_print(f"  Found {len(markets)} cross-platform matched markets")

    # Category breakdown
    cat_counts = Counter(m["category_display"] for m in markets)
    for cat, n in cat_counts.most_common():
        safe_print(f"    {cat}: {n}")

    if args.dry_run:
        print(f"\n-- DRY RUN: showing sample prompts --")
        today = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        for m in markets[:10]:
            prompt = USER_PROMPT_TEMPLATE.format(label=m["label"], today=today)
            safe_print(f"\n  Market:  {m['ticker']}")
            safe_print(f"  Label:   {m['original_label'][:70]}")
            safe_print(f"  VWAP:    {m['market_price']:.2%}")
            safe_print(f"  Volume:  ${m['total_volume']:,.0f}")
            safe_print(f"  Prompt:  {m['label'][:80]}")
        safe_print(f"\n  ... and {len(markets) - 10} more markets")
        safe_print(f"\n  Estimated API cost: ~${len(markets) * 0.003:.2f} (gpt-4o)")
        return

    # Run benchmark
    results = asyncio.run(run_benchmark(markets, args.model))

    # Compute metrics
    print("\nComputing calibration metrics...")
    metrics = compute_metrics(results)

    # Build output
    output = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        **metrics,
    }

    # Write output
    atomic_write_json(OUTPUT_PATH, output, ensure_ascii=False, indent=2)
    safe_print(f"\nResults written to {OUTPUT_PATH}")

    # Print summary
    o = metrics["overall"]
    print(f"\n{'=' * 70}")
    safe_print(f"RESULTS: {args.model} vs Bellwether VWAP ({o['n_markets']} markets)")
    print(f"{'=' * 70}")
    safe_print(f"  MAE:         {o['mae']:.4f} ({o['mae']:.1%})")
    safe_print(f"  RMSE:        {o['rmse']:.4f}")
    safe_print(f"  Bias:        {o['bias']:+.4f} ({'overconfident' if o['bias'] > 0 else 'underconfident'})")
    safe_print(f"  Correlation: {o['correlation']:.4f}")
    safe_print(f"\nBy Category:")
    for c in metrics["by_category"]:
        safe_print(f"  {c['category_display']:25s}  n={c['n']:3d}  MAE={c['mae']:.4f}  bias={c['bias']:+.4f}")
    safe_print(f"\nBy Liquidity:")
    for t in metrics["by_liquidity"]:
        safe_print(f"  {t['tier']:10s}  n={t['n']:3d}  MAE={t['mae']:.4f}  bias={t['bias']:+.4f}")
    safe_print(f"\nCalibration Curve:")
    for b in metrics["calibration_curve"]:
        safe_print(f"  [{b['bin_lo']:.0%}-{b['bin_hi']:.0%}]  market={b['mean_market']:.3f}  llm={b['mean_llm']:.3f}  n={b['n']}")


if __name__ == "__main__":
    main()
