import assert from 'node:assert/strict';
import {test} from 'node:test';
import {readFileSync} from 'node:fs';
import {analyzeCsv,normalizeUploadCsv,parseCsv,REQUIRED_COLUMNS} from '../dist/csv.mjs';
import {DEMOS,makeDemo} from '../dist/demo.mjs';
import {assessTraining} from '../dist/assessment.mjs';

test('simple upload keeps observations and missing telemetry distinct',()=>{
  const imported=normalizeUploadCsv('epoch,train_loss,val_loss\n0,1.2,1.3\n1,0.9,\n');
  const report=analyzeCsv(imported.csv);
  assert.equal(parseCsv(imported.csv).length,2);
  assert.deepEqual(report.curves.train_loss,[1.2,.9]);assert.deepEqual(report.curves.val_loss,[1.3,null]);
  assert.deepEqual(report.signals.gradient_max,[null,null]);assert.equal(report.diagnosis,null);
  assert.ok(imported.missingColumns.includes('gradient_max'));
});
test('minimum upload works without validation while rejected structures stay rejected',()=>{
  const report=analyzeCsv(normalizeUploadCsv('epoch,train_loss\n0,1\n1,.5\n').csv);
  assert.equal(assessTraining(report).id,'missing_validation');
  for(const text of ['epoch,val_loss\n0,1','epoch,epoch,train_loss\n0,1,2','epoch,train_loss\n0,1,extra'])assert.throws(()=>normalizeUploadCsv(text));
  assert.throws(()=>analyzeCsv(normalizeUploadCsv('epoch,train_loss\n1,1\n0,.5').csv),/Epoknumren/);
});
test('import drops annotations and retains real telemetry without inventing values',()=>{
  const imported=normalizeUploadCsv('epoch,train_loss,val_loss,gradient_max,diagnosis\n0,1,.9,0,loss_fault\n1,.8,.7,,loss_fault');
  assert.ok(!imported.csv.includes('loss_fault'));
  const report=analyzeCsv(imported.csv);assert.deepEqual(report.signals.gradient_max,[0,null]);assert.equal(report.diagnosis,null);
  const full=readFileSync(new URL('../dist/examples/rnn_training_log.csv',import.meta.url),'utf8');
  assert.equal(normalizeUploadCsv(full).csv,full); // Original published provenance stays intact.
});
test('demo scenarios yield readable pattern assessments and no classifier target',()=>{
  for(const demo of DEMOS){const sample=makeDemo(demo.id),report=analyzeCsv(sample.csv),assessment=assessTraining(report);
    assert.equal(assessment.id,demo.id);assert.equal(assessment.source,'logged_patterns');assert.equal(report.diagnosis,null);
    assert.equal(sample.illustrative,true);assert.equal(sample.category,undefined);
    assert.ok(!['Hyperparametrar','Lossfunktion'].includes(assessment.title));
    assert.deepEqual(report.signals.mean_gradient,Array(24).fill(null));
  }
});
test('learned category never replaces the observed assessment',()=>{
  const report=analyzeCsv(makeDemo('diverging').csv),expected=assessTraining(report);
  assert.deepEqual(assessTraining({...report,mode:'llama',diagnosis:'hyperparameter_fault'}),expected);
  assert.match(expected.summary,/kan tyda/);assert.match(expected.scope,/not a learned diagnosis/);
});
test('sparse endpoints and baseline metadata cannot certify improvement',()=>{
  const short=analyzeCsv(normalizeUploadCsv('epoch,train_loss,val_loss\n0,1,1\n1,.5,.5').csv);
  assert.equal(assessTraining({...short,diagnosis:'no_injected_fault',mode:'llama'}).id,'inconclusive');
});
