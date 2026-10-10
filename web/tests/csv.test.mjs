import assert from 'node:assert/strict';
import {readFileSync} from 'node:fs';
import {test} from 'node:test';
import {analyzeCsv, parseNumber, REQUIRED_COLUMNS} from '../dist/csv.mjs';

const fixture = readFileSync(new URL('../dist/examples/rnn_training_log.csv', import.meta.url), 'utf8');
const missingLog = `${REQUIRED_COLUMNS.join(',')}\n0,${Array(27).fill('nan').join(',')}\n1,${Array(27).fill('nan').join(',')}`;

test('original demo CSV is usable without copying its published target into predictions', () => {
  const report = analyzeCsv(fixture, 'RNN');
  assert.equal(report.num_epochs, 50);
  assert.equal(report.diagnosis, null);
  assert.equal(report.mode, 'preview');
  assert.equal(report.final.train_loss, report.curves.train_loss.at(-1));
});
test('nulls, booleans and Tensor wrappers preserve measurement semantics', () => {
  assert.equal(parseNumber('inf'), null);
  assert.equal(parseNumber(''), null);
  assert.equal(parseNumber('True'), 1);
  assert.equal(parseNumber('tf.Tensor(False, shape=(), dtype=bool)'), 0);
  const report = analyzeCsv(missingLog);
  assert.deepEqual(report.curves.train_loss, [null, null]);
  assert.equal(report.data_quality.missing_rate, 1);
});
test('interleaved epochs, missing columns and malformed quotes are rejected', () => {
  assert.throws(() => analyzeCsv(missingLog.replace('\n1,', '\n0,')), /Epoknumren/);
  assert.throws(() => analyzeCsv('epoch,train_loss\n0,1'), /saknar/);
  assert.throws(() => analyzeCsv('"unterminated'), /inte avslutas/);
});
test('labels and identifiers in extra columns cannot become a prediction', () => {
  const csv = missingLog.split('\n').map((line, i) => `${line},${i === 0 ? 'diagnosis' : 'weight_fault'}`).join('\n');
  assert.equal(analyzeCsv(csv).diagnosis, null);
});
