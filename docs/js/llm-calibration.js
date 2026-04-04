/**
 * LLM Calibration Benchmark Dashboard
 *
 * Loads llm_calibration.json and renders Plotly charts + stat cards.
 */

(function () {
  'use strict';

  // ── Helpers ───────────────────────────────────────────────────────────────

  async function fetchJSON(url) {
    const res = await fetch(url);
    if (!res.ok) throw new Error(`Failed to load ${url}: ${res.status}`);
    return res.json();
  }

  function pct(v, decimals) {
    if (v == null) return '---';
    decimals = decimals != null ? decimals : 1;
    return (v * 100).toFixed(decimals) + '%';
  }

  function fmtVolume(v) {
    if (v >= 1e6) return '$' + (v / 1e6).toFixed(1) + 'M';
    if (v >= 1e3) return '$' + (v / 1e3).toFixed(0) + 'K';
    return '$' + v.toFixed(0);
  }

  var PLOTLY_LAYOUT_DEFAULTS = {
    font: { family: 'DM Sans, sans-serif', size: 13, color: '#2C2C2C' },
    paper_bgcolor: 'transparent',
    plot_bgcolor: 'transparent',
    margin: { t: 20, r: 30, b: 60, l: 70 },
    hoverlabel: { font: { family: 'DM Sans, sans-serif', size: 12 } },
  };

  var PLOTLY_CONFIG = {
    displayModeBar: false,
    responsive: true,
  };

  // ── Stat Cards ────────────────────────────────────────────────────────────

  function renderStats(data) {
    var o = data.overall;

    document.getElementById('n-markets').textContent = o.n_markets;
    document.getElementById('model-name').textContent = data.model || '---';
    document.getElementById('generated-at').textContent = data.generated_at
      ? new Date(data.generated_at).toLocaleDateString('en-US', {
          year: 'numeric', month: 'short', day: 'numeric',
        })
      : '---';

    document.getElementById('stat-mae').textContent = pct(o.mae);
    document.getElementById('stat-rmse').textContent = pct(o.rmse);
    document.getElementById('stat-bias').textContent = (o.bias >= 0 ? '+' : '') + pct(o.bias);
    document.getElementById('stat-corr').textContent =
      o.correlation != null ? o.correlation.toFixed(3) : '---';

    var biasDetail = document.getElementById('stat-bias-detail');
    biasDetail.textContent = o.bias > 0.005
      ? 'LLM overestimates'
      : o.bias < -0.005
        ? 'LLM underestimates'
        : 'Near-zero bias';
  }

  // ── Calibration Curve ─────────────────────────────────────────────────────

  function renderCalibrationCurve(data) {
    var curve = data.calibration_curve;
    if (!curve || !curve.length) return;

    var x = curve.map(function (b) { return b.mean_market; });
    var y = curve.map(function (b) { return b.mean_llm; });
    var text = curve.map(function (b) {
      return 'Bin: ' + pct(b.bin_lo, 0) + '-' + pct(b.bin_hi, 0) +
        '<br>Market avg: ' + pct(b.mean_market) +
        '<br>LLM avg: ' + pct(b.mean_llm) +
        '<br>n=' + b.n;
    });

    var traces = [
      // Perfect calibration diagonal
      {
        x: [0, 1], y: [0, 1],
        mode: 'lines',
        line: { color: '#E8E8E8', width: 2, dash: 'dash' },
        name: 'Perfect calibration',
        hoverinfo: 'skip',
      },
      // Actual calibration
      {
        x: x, y: y,
        mode: 'lines+markers',
        marker: { size: 10, color: '#4A90D9' },
        line: { color: '#4A90D9', width: 3 },
        name: data.model || 'LLM',
        text: text,
        hoverinfo: 'text',
      },
    ];

    var layout = Object.assign({}, PLOTLY_LAYOUT_DEFAULTS, {
      xaxis: { title: 'Market Price (VWAP)', range: [0, 1], dtick: 0.2, tickformat: '.0%' },
      yaxis: { title: 'LLM Estimate', range: [0, 1], dtick: 0.2, tickformat: '.0%' },
      showlegend: true,
      legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)' },
      height: 420,
    });

    Plotly.newPlot('chart-calibration', traces, layout, PLOTLY_CONFIG);
  }

  // ── Category Bar Chart ────────────────────────────────────────────────────

  function renderCategoryChart(data) {
    var cats = data.by_category;
    if (!cats || !cats.length) return;

    var traces = [{
      x: cats.map(function (c) { return c.category_display; }),
      y: cats.map(function (c) { return c.mae; }),
      type: 'bar',
      marker: {
        color: cats.map(function (c) { return c.color || '#4A90D9'; }),
        line: { width: 0 },
      },
      text: cats.map(function (c) {
        return c.category_display + '<br>MAE: ' + pct(c.mae) +
          '<br>Bias: ' + (c.bias >= 0 ? '+' : '') + pct(c.bias) +
          '<br>n=' + c.n;
      }),
      hoverinfo: 'text',
    }];

    var layout = Object.assign({}, PLOTLY_LAYOUT_DEFAULTS, {
      yaxis: { title: 'Mean Absolute Error', tickformat: '.0%' },
      xaxis: { tickangle: -40, automargin: true },
      margin: { t: 20, r: 20, b: 120, l: 70 },
      height: 400,
    });

    Plotly.newPlot('chart-category', traces, layout, PLOTLY_CONFIG);
  }

  // ── Liquidity Bar Chart ───────────────────────────────────────────────────

  function renderLiquidityChart(data) {
    var tiers = data.by_liquidity;
    if (!tiers || !tiers.length) return;

    var tierColors = {
      fragile: '#D94A4A',
      thin: '#D4950A',
      moderate: '#4A90D9',
      liquid: '#3A8A5C',
    };

    var traces = [{
      x: tiers.map(function (t) { return t.tier.charAt(0).toUpperCase() + t.tier.slice(1); }),
      y: tiers.map(function (t) { return t.mae; }),
      type: 'bar',
      marker: {
        color: tiers.map(function (t) { return tierColors[t.tier] || '#4A90D9'; }),
        line: { width: 0 },
      },
      text: tiers.map(function (t) {
        return t.tier.charAt(0).toUpperCase() + t.tier.slice(1) +
          '<br>MAE: ' + pct(t.mae) +
          '<br>Bias: ' + (t.bias >= 0 ? '+' : '') + pct(t.bias) +
          '<br>n=' + t.n;
      }),
      hoverinfo: 'text',
    }];

    var layout = Object.assign({}, PLOTLY_LAYOUT_DEFAULTS, {
      yaxis: { title: 'Mean Absolute Error', tickformat: '.0%' },
      xaxis: { automargin: true },
      height: 400,
    });

    Plotly.newPlot('chart-liquidity', traces, layout, PLOTLY_CONFIG);
  }

  // ── Scatter Plot ──────────────────────────────────────────────────────────

  function renderScatter(data) {
    var markets = data.markets;
    if (!markets || !markets.length) return;

    // Group by category for colored traces
    var catMap = {};
    markets.forEach(function (m) {
      var key = m.category_display || 'Other';
      if (!catMap[key]) catMap[key] = { x: [], y: [], text: [], sizes: [], color: null };
      catMap[key].x.push(m.market_price);
      catMap[key].y.push(m.llm_estimate);
      catMap[key].text.push(
        m.label + '<br>Category: ' + m.category_display +
        '<br>Market: ' + pct(m.market_price) +
        '<br>LLM: ' + pct(m.llm_estimate) +
        '<br>Error: ' + (m.error >= 0 ? '+' : '') + pct(m.error) +
        '<br>Volume: ' + fmtVolume(m.volume)
      );
      catMap[key].sizes.push(Math.max(5, Math.min(25, Math.sqrt(m.volume / 50000))));
    });

    // Find category colors from by_category
    var colorLookup = {};
    if (data.by_category) {
      data.by_category.forEach(function (c) { colorLookup[c.category_display] = c.color; });
    }

    var traces = [
      // Diagonal
      {
        x: [0, 1], y: [0, 1],
        mode: 'lines',
        line: { color: '#E8E8E8', width: 2, dash: 'dash' },
        name: 'Perfect',
        hoverinfo: 'skip',
        showlegend: false,
      },
    ];

    Object.keys(catMap).forEach(function (cat) {
      var d = catMap[cat];
      traces.push({
        x: d.x, y: d.y,
        mode: 'markers',
        marker: {
          size: d.sizes,
          color: colorLookup[cat] || '#4A90D9',
          opacity: 0.7,
          line: { width: 1, color: 'rgba(255,255,255,0.6)' },
        },
        name: cat,
        text: d.text,
        hoverinfo: 'text',
      });
    });

    var layout = Object.assign({}, PLOTLY_LAYOUT_DEFAULTS, {
      xaxis: { title: 'Market Price (VWAP)', range: [-0.02, 1.02], tickformat: '.0%' },
      yaxis: { title: 'LLM Estimate', range: [-0.02, 1.02], tickformat: '.0%' },
      showlegend: true,
      legend: { x: 0.02, y: 0.98, bgcolor: 'rgba(255,255,255,0.8)', font: { size: 11 } },
      height: 500,
    });

    Plotly.newPlot('chart-scatter', traces, layout, PLOTLY_CONFIG);
  }

  // ── Sortable Market Table ─────────────────────────────────────────────────

  function renderTable(data) {
    var markets = data.markets;
    if (!markets || !markets.length) return;

    var tbody = document.getElementById('market-table-body');
    var currentSort = { key: 'abs_error', asc: false };

    // Find category colors
    var colorLookup = {};
    if (data.by_category) {
      data.by_category.forEach(function (c) { colorLookup[c.category_display] = c.color; });
    }

    function render() {
      var sorted = markets.slice().sort(function (a, b) {
        var va = a[currentSort.key], vb = b[currentSort.key];
        if (typeof va === 'string') {
          return currentSort.asc ? va.localeCompare(vb) : vb.localeCompare(va);
        }
        return currentSort.asc ? va - vb : vb - va;
      });

      tbody.innerHTML = sorted.map(function (m) {
        var color = colorLookup[m.category_display] || '#6b7280';
        var errClass = m.error > 0 ? 'error-positive' : m.error < 0 ? 'error-negative' : '';
        return '<tr>' +
          '<td class="label-cell">' + m.label + '</td>' +
          '<td><span class="cat-badge" style="background:' + color + '15;color:' + color + '">' +
            m.category_display + '</span></td>' +
          '<td>' + pct(m.market_price) + '</td>' +
          '<td>' + pct(m.llm_estimate) + '</td>' +
          '<td class="' + errClass + '">' + (m.error >= 0 ? '+' : '') + pct(m.error) + '</td>' +
          '<td>' + pct(m.abs_error) + '</td>' +
          '<td>' + fmtVolume(m.volume) + '</td>' +
          '</tr>';
      }).join('');
    }

    // Sort click handlers
    document.querySelectorAll('#market-table th[data-sort]').forEach(function (th) {
      th.addEventListener('click', function () {
        var key = th.getAttribute('data-sort');
        if (currentSort.key === key) {
          currentSort.asc = !currentSort.asc;
        } else {
          currentSort.key = key;
          currentSort.asc = key === 'label' || key === 'category_display';
        }
        // Update sort arrows
        document.querySelectorAll('#market-table th').forEach(function (h) {
          h.classList.remove('sorted');
        });
        th.classList.add('sorted');
        th.querySelector('.sort-arrow').innerHTML = currentSort.asc ? '&#9650;' : '&#9660;';
        render();
      });
    });

    render();
  }

  // ── Init ──────────────────────────────────────────────────────────────────

  fetchJSON('data/llm_calibration.json')
    .then(function (data) {
      renderStats(data);
      renderCalibrationCurve(data);
      renderCategoryChart(data);
      renderLiquidityChart(data);
      renderScatter(data);
      renderTable(data);
    })
    .catch(function (err) {
      console.error('Failed to load calibration data:', err);
      document.querySelector('.page-header .subtitle').textContent =
        'Error loading benchmark data. Run the benchmark pipeline first: ' +
        'python packages/pipelines/llm_calibration_benchmark.py';
    });

})();
