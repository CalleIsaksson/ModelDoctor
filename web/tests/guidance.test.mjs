import assert from 'node:assert/strict';
import {test} from 'node:test';
import {analyzeCsv,REQUIRED_COLUMNS} from '../dist/csv.mjs';
import {reviewPlan,categoryResult} from '../dist/guidance.mjs';
const log=(validation=[1,.5,.4,.5,.6,.7])=>analyzeCsv(`${REQUIRED_COLUMNS.join(',')}\n${validation.map((value,i)=>[i,1/(i+1),value,...Array(25).fill('')].join(',')).join('\n')}`);

test('uploaded observations give an actionable plan without inheriting a demo label',()=>{
  const report=log(),before=JSON.stringify(report),plan=reviewPlan(report);
  assert.equal(plan.category,null);assert.equal(plan.area,null);
  assert.equal(plan.checks[0].id,'review_best_epoch');assert.equal(plan.checks[0].source,'observed');
  assert.match(plan.checks[0].how,/checkpoint/);assert.match(plan.checks[0].verify,/sluttestet/);
  assert.equal(JSON.stringify(report),before);assert.equal(report.diagnosis,null);
});
test('published category and model prediction retain independent sources',()=>{
  const report=log();assert.equal(categoryResult(report,'loss_fault').source,'published_example');
  assert.deepEqual(categoryResult({...report,mode:'llama',diagnosis:'weight_fault'},'loss_fault'),{area:'weight_fault',source:'model_category',label:'Modellförslag'});
  assert.equal(categoryResult({...report,mode:'preview',diagnosis:'weight_fault'}),null);
  assert.equal(categoryResult({...report,mode:'llama',diagnosis:'unrecognized'}),null);
});
test('task-specific loss guide is conditional and preserves soft-target support',()=>{
  const report=log();const plan=reviewPlan(report,{area:'loss_fault',task:'multiclass'});
  const check=plan.checks.find(item=>item.id==='loss_contract');
  assert.equal(check.source,'user_selected');assert.match(check.how,/Om du använder/);
  assert.match(check.how,/klassindex eller klassfördelningar/);assert.equal(check.links[0].title,'PyTorch: CrossEntropyLoss');
  assert.equal(plan.category,null);assert.equal(report.diagnosis,null);
  assert.match(reviewPlan(report,{area:'loss_fault',task:'binary'}).checks.find(item=>item.id==='loss_contract').how,/utan en extra sigmoid/);
  assert.match(reviewPlan(report,{area:'loss_fault',task:'regression'}).checks.find(item=>item.id==='loss_contract').how,/shape/);
});
test('hyperparameter label never prescribes a direction or invented setting',()=>{
  const plan=reviewPlan(log(),{exampleCategory:'hyperparameter_fault'});
  const check=plan.checks.find(item=>item.id==='effective_settings');
  assert.equal(check.source,'published_example');assert.match(check.how,/inte om learning rate är för hög eller låg/);
  assert.ok(plan.checks.some(item=>item.id==='one_change'));
});
test('baseline label and no-rule result never certify healthy training',()=>{
  const plan=reviewPlan(log([1,.8,.7,.6,.5,.4]),{exampleCategory:'no_injected_fault'});
  assert.equal(plan.category.area,'no_injected_fault');assert.equal(plan.checks[0].id,'set_reference');
  assert.match(plan.checks[0].why,/inget specifikt fel/);
});
test('missing validation gives instrumentation guidance before a chosen area',()=>{
  const plan=reviewPlan(log([null,null,null,null,null,null]),{area:'loss_fault'});
  assert.equal(plan.checks[0].id,'add_validation');assert.match(plan.headline,/saknas/);
  assert.ok(plan.checks.some(item=>item.id==='loss_contract'));
});
