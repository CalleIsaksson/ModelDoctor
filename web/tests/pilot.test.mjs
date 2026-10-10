import assert from 'node:assert/strict';
import {test} from 'node:test';
import {analyzeCsv, REQUIRED_COLUMNS} from '../dist/csv.mjs';
import {comparison} from '../dist/workspace.mjs';

function log(train, validation) {
  return `${REQUIRED_COLUMNS.join(',')}\n${train.map((value,i)=>[i,value,validation[i],...Array(25).fill('')].join(',')).join('\n')}`;
}

test('observed validation regression offers an investigation with evidence, no classifier label',()=>{
  const report = analyzeCsv(log([1,.8,.6,.5,.4,.3],[1,.5,.4,.5,.6,.7]));
  assert.deepEqual(report.inspection.best_validation,{epoch:2,value:.4});
  const finding = report.inspection.findings.find(item=>item.id === 'validation_regression');
  assert.equal(finding.evidence.final_loss,.7);
  assert.equal(report.diagnosis,null);
});
test('missing validation produces an instrumentation check, and gaps never become zeros',()=>{
  const report = analyzeCsv(log([1,.8,.6],[null,null,null]));
  assert.equal(report.inspection.best_validation,null);
  assert.equal(report.inspection.findings[0].id,'missing_validation');
  assert.ok(!report.inspection.findings.some(item=>item.id === 'zero_logged_gradient'));
});
test('comparison warns about incompatible data and lists changed settings without a causal claim',()=>{
  const a={project:'p',dataset:'v1',architecture:'CNN',metric_name:'accuracy',config:{lr:.01,batch:32},report:analyzeCsv(log([1,.8],[1,.9]))};
  const b={...a,dataset:'v2',config:{lr:.001,batch:32},report:analyzeCsv(log([1,.7],[1,.8]))};
  const result=comparison(a,b);
  assert.equal(result.warnings.length,1);
  assert.deepEqual(result.settings,[{key:'lr',a:.01,b:.001}]);
  assert.ok(Math.abs(result.best_delta + .1) < 1e-12);
});
