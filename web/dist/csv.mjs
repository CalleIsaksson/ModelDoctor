import {inspectReport} from './inspection.mjs';
export const CURVES = ['train_loss', 'val_loss', 'train_acc', 'val_acc'];
export const SIGNALS = ['large_weight_count', 'acc_gap_too_big', 'loss_oscillation', 'dying_relu',
  'gradient_vanish', 'gradient_explode', 'decrease_acc_count', 'increase_loss_count',
  'cons_mean_weight_count', 'cons_std_weight_count', 'nan_weight_count', 'nan_gradients_count',
  'saturated_activation', 'mean_gradient', 'gradient_std', 'gradient_max', 'gradient_min',
  'gradient_median', 'adjusted_lr', 'mean_activation', 'std_activation', 'mean_grad', 'std_grad'];
export const REQUIRED_COLUMNS = ['epoch', ...CURVES, ...SIGNALS];
export const MAX_BYTES = 5 * 1024 * 1024;

export function parseCsv(text, requiredColumns = REQUIRED_COLUMNS) {
  text = text.replace(/^\uFEFF/, '');
  const rows = [];
  let row = [], cell = '', quoted = false, afterQuote = false;
  const endCell = () => { row.push(cell); cell = ''; afterQuote = false; };
  const endRow = () => { endCell(); if (row.length > 1 || row[0] !== '') rows.push(row); row = []; };
  for (let i = 0; i < text.length; i++) {
    const char = text[i];
    if (quoted) {
      if (char === '"') {
        if (text[i + 1] === '"') { cell += '"'; i++; }
        else { quoted = false; afterQuote = true; }
      } else cell += char;
    } else if (char === ',') endCell();
    else if (char === '\r' || char === '\n') { if (char === '\r' && text[i + 1] === '\n') i++; endRow(); }
    else if (char === '"' && cell === '' && !afterQuote) quoted = true;
    else if (char === '"' || afterQuote) throw new Error('CSV-filen innehåller felaktiga citationstecken.');
    else cell += char;
  }
  if (quoted) throw new Error('CSV-filen har ett citat som inte avslutas.');
  if (cell !== '' || row.length || afterQuote) endRow();
  if (rows.length < 2) throw new Error('CSV-filen behöver kolumnnamn och minst en epok.');
  const headers = rows.shift();
  if (new Set(headers).size !== headers.length) throw new Error('Kolumnnamnen måste vara unika.');
  const missing = requiredColumns.filter(name => !headers.includes(name));
  if (missing.length) throw new Error(`Filen saknar ${missing.length} kolumner: ${missing.slice(0, 5).join(', ')}${missing.length > 5 ? '…' : ''}. Se CSV-formatet.`);
  return rows.map((cells, index) => {
    if (cells.length !== headers.length) throw new Error(`Rad ${index + 2} har fel antal kolumner.`);
    return Object.fromEntries(headers.map((name, i) => [name, cells[i]]));
  });
}

// Import convenience only. The trainer and API still receive the existing observed schema.
export function normalizeUploadCsv(text) {
  if (new TextEncoder().encode(text).length > MAX_BYTES) throw new Error('Filen får vara högst 5 MB.');
  const rows = parseCsv(text, ['epoch', 'train_loss']);
  if (rows.length > 10000) throw new Error('Loggen får innehålla högst 10 000 epoker.');
  const providedColumns = Object.keys(rows[0]);
  const missingColumns = REQUIRED_COLUMNS.filter(name => !providedColumns.includes(name));
  if (!missingColumns.length) return {csv:text,providedColumns,missingColumns};
  const escape = value => /[",\r\n]/.test(value) ? `"${value.replaceAll('"','""')}"` : value;
  const csv = `${REQUIRED_COLUMNS.join(',')}\n${rows.map(row=>REQUIRED_COLUMNS.map(name=>escape(row[name] ?? '')).join(',')).join('\n')}\n`;
  return {csv,providedColumns,missingColumns};
}

export function parseNumber(value) {
  let text = String(value).trim().toLowerCase();
  if (text === 'true' || text === 'false') return Number(text === 'true');
  if (text.startsWith('tf.tensor(')) return parseNumber(text.slice(10).split(',')[0]);
  if (!text) return null;
  if (/^[+-]?(nan|inf|infinity)$/.test(text)) return null;
  if (!/^[+-]?(\d+(\.\d*)?|\.\d+)(e[+-]?\d+)?$/.test(text)) throw new Error(`Ogiltigt mätvärde: ${text.slice(0, 30)}.`);
  const number = Number(text);
  return Number.isFinite(number) ? number : null;
}

export function analyzeCsv(text, architecture = 'unknown') {
  if (new TextEncoder().encode(text).length > MAX_BYTES) throw new Error('CSV-filen får vara högst 5 MB.');
  if (!['FNN', 'CNN', 'RNN', 'unknown'].includes(architecture)) throw new Error('Välj en giltig arkitektur.');
  const rows = parseCsv(text);
  if (rows.length > 10000) throw new Error('CSV-filen får innehålla högst 10 000 epoker.');
  const epochs = rows.map(row => parseNumber(row.epoch));
  if (epochs.some((epoch, i) => epoch === null || !Number.isInteger(epoch) || epoch < 0 || (i > 0 && epochs[i - 1] >= epoch)))
    throw new Error('Epoknumren måste vara heltal från 0 och öka för varje rad.');
  const curves = Object.fromEntries(CURVES.map(name => [name, rows.map(row => parseNumber(row[name]))]));
  const signals = Object.fromEntries(['mean_gradient', 'gradient_max', 'adjusted_lr'].map(name => [name, rows.map(row => parseNumber(row[name]))]));
  const all = rows.flatMap(row => [...CURVES, ...SIGNALS].map(name => parseNumber(row[name])));
  const missing = all.filter(value => value === null).length;
  const report = {api_version: 1, architecture, epochs, curves, signals, num_epochs: rows.length,
    data_quality: {missing_values: missing, total_values: all.length, missing_rate: missing / all.length},
    final: Object.fromEntries(CURVES.map(name => [name, curves[name].at(-1)])),
    diagnosis: null, mode: 'preview', model: null,
    warnings: ['Källans train_acc/val_acc kan vara ett regressionsmått och visas som registrerade värden.']};
  report.inspection = inspectReport(report);
  return report;
}
