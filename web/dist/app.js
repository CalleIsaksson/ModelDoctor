import {analyzeCsv, normalizeUploadCsv, SIGNALS, MAX_BYTES} from './csv.mjs';
import {DEMOS, makeDemo} from './demo.mjs';
import {assessTraining} from './assessment.mjs';
import {BrowserRuns, comparison} from './workspace.mjs';
import {AREAS, categoryResult, reviewPlan} from './guidance.mjs';
import {renderTrainingChart} from './chart.mjs';

const $ = id => document.getElementById(id);
const state = {csv: '', name: '', example: null, report: null, chart: 'loss', examples: [], api: {available: false}, busy: false, version: 0, currentRun:null, runs:[], selected:new Set(), token:'', viewMode:'demo', originalCsv:'', importInfo:null,ownDraft:null};
const exploration = {window:'all',index:null,hidden:new Set(),reviewed:new Set(),expanded:new Set(),playing:false,timer:null,reveal:null};
function resetExploration() {
  stopPlayback(); exploration.reveal=null;
  state.chart='loss'; for(const button of document.querySelectorAll('[data-chart]'))button.setAttribute('aria-selected',String(button.dataset.chart==='loss'));
  document.querySelector('.extra-charts').open=false;
  exploration.window='all'; exploration.index=null; exploration.hidden.clear(); exploration.reviewed.clear(); exploration.expanded.clear();
  $('chart-window').value='all'; $('task-context').value='unknown'; $('review-area').value='auto';
}
const browserRuns = new BrowserRuns();
const API_BASE = (window.MODELDOCTOR_API_BASE || '').replace(/\/$/, '');
const number = value => value === null || value === undefined ? '—' : Math.abs(value) >= 1e5 || (Math.abs(value) < 0.0001 && value !== 0) ? value.toExponential(2) : new Intl.NumberFormat('sv-SE', {maximumFractionDigits: 4}).format(value);
const svgNS = 'http://www.w3.org/2000/svg';
const svgElement = (name, attributes = {}, content) => {
  const element = document.createElementNS(svgNS, name);
  if (name === 'text') element.style.stroke = 'none';
  for (const [key, value] of Object.entries(attributes)) element.setAttribute(key, String(value));
  if (content !== undefined) element.textContent = content;
  return element;
};

function showError(message) { $('error-message').textContent = message; $('error-message').hidden = false; }
function clearError() { $('error-message').hidden = true; $('error-message').textContent = ''; }
function toast(message) {
  $('toast').textContent = message; $('toast').hidden = false;
  clearTimeout(toast.timeout); toast.timeout = setTimeout(() => { $('toast').hidden = true; }, 3500);
}
function showDialog(id) { $(id).showModal(); }
function changeDetails(column, target) {
  const values = state.report.curves[column];
  const first = values[0], last = values.at(-1);
  target.className = '';
  if (first === null || last === null || first === 0) { target.textContent = 'Sista registrerade värdet'; return; }
  const change = (last - first) / Math.abs(first) * 100;
  target.textContent = `${change < 0 ? '↓' : '↑'} ${new Intl.NumberFormat('sv-SE', {maximumFractionDigits: 1}).format(Math.abs(change))} % från första epoken`;
  target.className = change <= 0 ? 'good' : 'warm';
}

function render() {
  const report=state.report;if(!report)return;
  $('analysis-results').hidden=false;$('empty-analysis').hidden=true;
  const assessment=assessTraining(report),result=categoryResult(report,state.example?.category),predicted=result?.source==='model_category';
  $('run-name').textContent=state.example?.name || state.name;
  $('run-badge').textContent=state.example ? (state.example.illustrative?'DEMO':'EXEMPELLOGG') : state.currentRun ? ({running:'PÅGÅR',completed:'SPARAD',failed:'AVBRUTEN'}[state.currentRun.status]) : 'DIN LOGG';
  $('run-details').textContent=report.num_epochs+' träningsrundor · '+(state.example?.illustrative?'Illustrerat scenario':state.example?'Publicerad träningslogg':state.currentRun?state.currentRun.project:'Dina uppmätta resultat');
  $('stat-epochs').textContent=report.num_epochs;$('stat-train').textContent=number(report.final.train_loss);$('stat-val').textContent=number(report.final.val_loss);
  $('stat-quality').textContent=Math.round((1-report.data_quality.missing_rate)*100)+' %';
  changeDetails('train_loss',$('stat-train-change'));changeDetails('val_loss',$('stat-val-change'));
  $('diagnosis-title').textContent=assessment.title;$('diagnosis-description').textContent=assessment.summary;$('assessment-next').textContent=assessment.next;
  $('diagnosis-badge').textContent={attention:'Granska närmare',positive:'Förbättring syns',neutral:'Undersök vidare'}[assessment.tone];
  $('diagnosis-badge').className='badge '+(assessment.tone==='attention'?'badge-amber':'badge-muted');
  document.querySelector('.assessment-panel').className='panel assessment-panel '+assessment.tone;
  $('diagnosis-note').textContent='Bedömning från loggens mönster. Ingen säker grundorsak eller verifierad lösning har fastställts.';
  $('technical-title').textContent=predicted?'Modellens tekniska förslag':'Modellstöd efter träning';
  $('technical-description').textContent=predicted?AREAS[result.area].meaning:'Kurvbedömningen och guiderna fungerar redan nu. Den lärda modellen behöver tränas och utvärderas innan dess kategoriförslag kan bedömas som tillförlitliga.';
  $('technical-source').textContent=predicted?'Modellens experimentella kategori används för att välja ytterligare kontrollsteg. Den är inte samma sak som kurvbedömningen.':report.model_skip_reason==='insufficient_training_signals'?'Loggen räcker för kurvor, men interna träningssignaler saknas för den lärda modellen.':'Ingen modellprediktion har gjorts för den här loggen.';
  $('category-eyebrow').textContent=predicted?'Rå modellkategori':state.example?.category?'Datasetets experimentetikett (inte en diagnos)':'';
  $('category-code').textContent=result?.area || '';
  $('demo-notice').hidden=!state.example;
  $('demo-notice').querySelector('span').textContent=state.example?.illustrative?'Illustrerad demo: värdena är skapade för att förklara mönstret. Bedömningen räknas från kurvorna; ingen AI-prediktion görs.':'Publicerad exempellogg. Bedömningen beskriver de uppmätta kurvorna, inte datasetets mutationsetikett.';
  $('demo-playback').hidden=!state.example?.illustrative;
  $('demo-progress').hidden=!state.example?.illustrative || exploration.reveal===null;
  if(exploration.reveal!==null)$('demo-progress').textContent='Visar träningen till runda '+report.epochs[exploration.reveal]+' av '+report.epochs.at(-1)+'. Bedömningen gäller hela scenariot.';
  for(const button of document.querySelectorAll('[data-demo]'))button.setAttribute('aria-pressed',String(button.dataset.demo===state.example?.id));
  $('save-run').disabled=Boolean(state.currentRun);
  $('save-status').textContent=state.currentRun?'Sparad i '+state.currentRun.project+' · revision '+state.currentRun.revision:'Spara körningen och jämför vad som händer efter din nästa ändring.';
  $('analyze-button').disabled=Boolean(state.currentRun&&state.currentRun.status!=='completed');
  $('feedback-actions').hidden=!(predicted&&state.currentRun&&state.api.local);
  $('data-detail-label').textContent=report.data_quality.missing_values+' signalvärden saknas. Det är väntat i en enkel logg och innebär inte i sig ett träningsfel.';
  $('report-message').lastChild.textContent=' Bedömning av '+report.num_epochs+' träningsrundor klar';
  renderInspection();renderGuidance();renderObservations();renderChart();
}

function renderObservations() {
  const report = state.report, observations = [];
  const startTrain = report.curves.train_loss[0], endTrain = report.final.train_loss;
  const startVal = report.curves.val_loss[0], endVal = report.final.val_loss;
  if ([startTrain, endTrain, startVal, endVal].every(value => value !== null)) {
    if (endTrain < startTrain && endVal > startVal) observations.push(['Kurvorna går åt olika håll', 'Träningsloss sjunker samtidigt som valideringsloss ökar mellan första och sista epoken.', true]);
    else if (endTrain <= startTrain && endVal <= startVal) observations.push(['Loss sjunker i båda kurvorna', 'Både träningsloss och valideringsloss slutar lägre än eller lika med sitt första värde.', false]);
    else observations.push(['Jämför kurvornas utveckling', 'Första och sista värdet beskriver förändringen. Grafen visar även variationen mellan epokerna.', true]);
  } else observations.push(['Lossvärden saknas', 'Minst ett start- eller slutvärde saknas. Luckor i grafen visas utan att fyllas med noll.', true]);
  if (report.data_quality.missing_values) observations.push(['Vissa mätningar saknas', `${report.data_quality.missing_values} av ${report.data_quality.total_values} signalvärden saknas. Datatäckningen visar hur stor del som är uppmätt.`, true]);
  else observations.push(['Alla signalvärden är uppmätta', 'Ingen saknad eller icke ändlig mätning hittades i de använda kolumnerna.', false]);
  $('observations').replaceChildren();
  for (const [title, description, warm] of observations) {
    const item = document.createElement('div'); item.className = 'observation';
    const marker = document.createElement('span'); marker.className = `observation-marker${warm ? ' warm' : ''}`;
    const icon = svgElement('svg'); icon.append(svgElement('use', {href: warm ? '#i-info' : '#i-check'})); marker.append(icon);
    const text = document.createElement('div');
    const strong = document.createElement('strong'); strong.textContent = title;
    const paragraph = document.createElement('p'); paragraph.textContent = description;
    text.append(strong, paragraph); item.append(marker, text); $('observations').append(item);
  }
}

function renderChart() {
  if (!state.report) return;
  renderTrainingChart(state.report,{...exploration,chart:state.chart}, {
    number,svgElement,
    toggle(key,count) {
      if (exploration.hidden.has(key)) exploration.hidden.delete(key);
      else {
        const hiddenForChart=[...exploration.hidden].filter(item=>item.startsWith(state.chart+':')).length;
        if (hiddenForChart >= count-1) { toast('Behåll minst en kurva synlig.'); return; }
        exploration.hidden.add(key);
      }
      renderChart();
    },
    select(index) { stopPlayback(); exploration.reveal=null; exploration.index=index; renderChart(); },
  });
}

async function loadExample(example) {
  clearError();const version=++state.version;
  if(state.viewMode==='own'&&state.report)state.ownDraft=Object.fromEntries(['csv','name','example','report','currentRun','originalCsv','importInfo'].map(key=>[key,state[key]]));
  try {
    let csv=example.csv;
    if(!csv){const response=await fetch('examples/'+example.file);if(!response.ok)throw new Error('Exempelloggen kunde inte hämtas.');csv=await response.text();}
    if(version!==state.version)return;
    const report=analyzeCsv(csv,example.architecture);
    Object.assign(state,{csv,name:example.file,example,report,currentRun:null,originalCsv:csv,importInfo:null,viewMode:example.illustrative?'demo':'own'});
    resetExploration();$('architecture').value=example.architecture;$('file-input').value='';$('selected-file').hidden=true;
    showView('analysis');render();
  }catch(error){showError(error.message);}
}

function clearResult() {
  state.report=null;state.csv='';state.example=null;state.currentRun=null;state.originalCsv='';state.importInfo=null;
  resetExploration();$('analysis-results').hidden=true;$('empty-analysis').hidden=false;$('selected-file').hidden=true;
  $('guidance-checks').replaceChildren();$('inspection-findings').replaceChildren();$('training-chart').replaceChildren();$('epoch-values').replaceChildren();
  $('diagnosis-title').textContent='Väntar på en logg';$('category-code').textContent='';$('feedback-actions').hidden=true;
}
async function loadFile(file) {
  if(!file)return;clearError();const version=++state.version;
  state.ownDraft=null;
  clearResult();state.viewMode='own';showView('analysis');selectInputRoute('upload');
  try {
    if(!file.name.toLowerCase().endsWith('.csv'))throw new Error('Välj en CSV-fil. Hämta exempelmallen om du behöver se formatet.');
    if(file.size>MAX_BYTES)throw new Error('Filen får vara högst 5 MB.');
    const originalCsv=await file.text();if(version!==state.version)return;
    const imported=normalizeUploadCsv(originalCsv),report=analyzeCsv(imported.csv,$('architecture').value);
    Object.assign(state,{csv:imported.csv,name:file.name,report,originalCsv,importInfo:{provided_columns:imported.providedColumns,missing_columns:imported.missingColumns}});
    $('selected-file-name').textContent=file.name+' · klar att granska';$('selected-file').hidden=false;
    render();$('analysis-results').scrollIntoView({behavior:'smooth',block:'start'});toast('Loggen är klar. Här är kurvorna och bedömningen.');
  }catch(error){showError(error.message);}
}

async function runAnalysis() {
  clearError();
  if (!state.csv || !state.report) { showError('Välj en giltig CSV-fil eller en exempellogg först.'); return; }
  if (state.busy) return;
  if (state.currentRun && state.currentRun.status !== 'completed') { showError('Modellanalys kräver en avslutad körning.'); return; }
  state.busy = true;
  $('analyze-button').disabled = true; $('analyze-button').querySelector('span').textContent = 'Analyserar…';
  const version = state.version;
  try {
    let report;
    if (state.api.local && !state.example?.illustrative && state.currentRun?.source !== 'illustrative-demo') {
      const result = await apiRequest(state.currentRun ? `/api/runs/${state.currentRun.run_id}/analyze` : '/api/analyze', {method:'POST',body:JSON.stringify({api_version:1,csv:state.csv,architecture:$('architecture').value})});
      report = state.currentRun ? result.report : result;
      if (version === state.version && state.currentRun) state.currentRun = result;
    } else report = analyzeCsv(state.csv, $('architecture').value);
    if (version === state.version) { state.report = report; render(); toast(report.mode === 'llama' ? 'Modellanalysen är klar.' : 'Kurvor och data är kontrollerade.'); }
  } catch (error) { showError(error.message); }
  finally { state.busy = false; $('analyze-button').disabled = Boolean(state.currentRun && state.currentRun.status !== 'completed'); $('analyze-button').querySelector('span').textContent = 'Uppdatera analysen'; }
}

function download(content, filename, type) {
  const url = URL.createObjectURL(new Blob([content], {type}));
  const anchor = document.createElement('a'); anchor.href = url; anchor.download = filename; anchor.click();
  setTimeout(() => URL.revokeObjectURL(url), 500);
}

$('file-input').addEventListener('change', event => loadFile(event.target.files[0]));
$('dropzone').addEventListener('dragover', event => { event.preventDefault(); $('dropzone').classList.add('dragover'); });
$('dropzone').addEventListener('dragleave', () => $('dropzone').classList.remove('dragover'));
$('dropzone').addEventListener('drop', event => { event.preventDefault(); $('dropzone').classList.remove('dragover'); loadFile(event.dataTransfer.files[0]); });
$('analyze-button').addEventListener('click', runAnalysis);
$('chart-window').addEventListener('change',()=>{exploration.window=$('chart-window').value; if(state.report)renderChart();});
$('epoch-scrubber').addEventListener('input',()=>{stopPlayback();exploration.reveal=null;exploration.index=Number($('epoch-scrubber').value);if(state.report)renderChart();});
$('focus-best').addEventListener('click',()=>{
  const best=state.report?.inspection?.best_validation; if(!best)return;
  stopPlayback();exploration.reveal=null;exploration.window='all';$('chart-window').value='all';exploration.index=state.report.epochs.indexOf(best.epoch);exploration.hidden.delete('loss:Validering');
  document.querySelector('[data-chart="loss"]').click();
  $('chart-container').scrollIntoView({behavior:'smooth',block:'center'});
});
for(const id of ['task-context','review-area'])$(id).addEventListener('change',()=>{exploration.reviewed.clear();exploration.expanded.clear();if(state.report)renderGuidance();});
$('architecture').addEventListener('change', () => { if (state.csv && state.report) { state.version++; state.report = analyzeCsv(state.csv, $('architecture').value); state.example = null; state.currentRun = null; exploration.reviewed.clear(); exploration.expanded.clear(); render(); } });
$('nav-examples').addEventListener('click',()=>{$('model-details-dialog').close();showDialog('examples-dialog');});
for (const id of ['nav-guide', 'format-button']) $(id).addEventListener('click', () => showDialog('guide-dialog'));
for (const dialog of document.querySelectorAll('dialog')) {
  dialog.querySelector('.dialog-close').addEventListener('click', () => dialog.close());
  dialog.addEventListener('click', event => { if (event.target === dialog && (event.clientX < dialog.getBoundingClientRect().left || event.clientX > dialog.getBoundingClientRect().right || event.clientY < dialog.getBoundingClientRect().top || event.clientY > dialog.getBoundingClientRect().bottom)) dialog.close(); });
}
$('nav-analysis').addEventListener('click',openOwn);
$('nav-home').addEventListener('click',()=>showView('home'));
for(const id of ['nav-demo','start-demo','empty-demo','mode-demo'])$(id).addEventListener('click',()=>loadExample(makeDemo('diverging')));
for(const id of ['start-own','home-upload','mode-own'])$(id).addEventListener('click',openOwn);
for(const id of ['home-connect','nav-connect'])$(id).addEventListener('click',()=>showDialog('connect-dialog'));
$('model-details-button').addEventListener('click',()=>showDialog('model-details-dialog'));
$('input-upload').addEventListener('click',()=>selectInputRoute('upload'));
$('input-python').addEventListener('click',()=>selectInputRoute('python'));
$('demo-play').addEventListener('click',playDemo);
$('demo-reset').addEventListener('click',()=>{stopPlayback();exploration.reveal=null;exploration.index=null;render();});
for(const demo of DEMOS){
 const teaser=document.createElement('button');teaser.className='teaser-option';teaser.dataset.startDemo=demo.id;
 const text=document.createElement('span'),strong=document.createElement('strong'),small=document.createElement('small');strong.textContent=demo.short;small.textContent=demo.description;text.append(strong,small);
 const arrow=svgElement('svg');arrow.append(svgElement('use',{href:'#i-arrow'}));teaser.append(text,arrow);teaser.addEventListener('click',()=>loadExample(makeDemo(demo.id)));$('home-demo-options').append(teaser);
 const option=document.createElement('button');option.className='demo-option';option.dataset.demo=demo.id;option.setAttribute('aria-pressed','false');
 const icon=document.createElement('span');icon.className='round-icon';const chart=svgElement('svg');chart.append(svgElement('use',{href:'#i-chart'}));icon.append(chart);
 const copy=document.createElement('span'),title=document.createElement('strong'),description=document.createElement('small');title.textContent=demo.short;description.textContent=demo.description;copy.append(title,description);option.append(icon,copy);option.addEventListener('click',()=>loadExample(makeDemo(demo.id)));$('demo-selector').append(option);
}
$('download-csv').addEventListener('click', () => { if (state.csv) download(state.originalCsv || state.csv, state.name, 'text/csv;charset=utf-8'); else showError('Ingen giltig CSV att ladda ner.'); });
$('export-report').addEventListener('click', () => {
  if (!state.report) { showError('Analysera en giltig fil först.'); return; }
  const report = {...state.report, filename: state.name, demo: Boolean(state.example) || ['demo-example','illustrative-demo'].includes(state.currentRun?.source),
    exported_at:new Date().toISOString(), product_version:'0.4.0', assessment:assessTraining(state.report), import_normalization:state.importInfo, investigation: {...currentPlan(),reviewed_checks:[...exploration.reviewed],review_status_scope:'Current browser session; reviewed does not mean repaired.'}, run_metadata:state.currentRun ? Object.fromEntries(['run_id','revision','project','config','dataset','metric_name','source','input_sha256','created_at'].map(key=>[key,state.currentRun[key]])) : null,
    published_example_target: state.example?.category || null, source:state.example?.illustrative?{type:'illustrative_demo',demo_id:state.example.id,scope:'UI illustration only; not training data or a published answer key.'}:state.example?{dataset:'Deep4ge',example_id:state.example.source_example_id,source_sha256:state.example.source_sha256}:null};
  download(JSON.stringify(report, null, 2), 'modeldoctor-report.json', 'application/json'); toast('Rapporten har exporterats.');
});
for (const tab of document.querySelectorAll('[data-chart]')) {
  tab.addEventListener('click', () => {
    state.chart = tab.dataset.chart;
    for (const button of document.querySelectorAll('[data-chart]')) button.setAttribute('aria-selected', String(button === tab));
    if (state.report) renderChart();
  });
  tab.addEventListener('keydown', event => {
    if (!['ArrowLeft', 'ArrowRight'].includes(event.key)) return;
    event.preventDefault(); const tabs = [...document.querySelectorAll('[data-chart]')].filter(button=>button.getClientRects().length);
    const next = tabs[(tabs.indexOf(tab) + (event.key === 'ArrowRight' ? 1 : -1) + tabs.length) % tabs.length]; next.click(); next.focus();
  });
}
$('required-signals').textContent = SIGNALS.join(', ');
let lastChartWidth = 0;
new ResizeObserver(entries => {
  const width = entries[0].contentRect.width;
  if (Math.abs(width - lastChartWidth) > 1) { lastChartWidth = width; if (state.report) renderChart(); }
}).observe($('chart-container'));

async function initialize() {
  const [exampleResult, statusResult] = await Promise.allSettled([
    fetch('examples/index.json').then(response => { if (!response.ok) throw new Error('Exempelloggarna kunde inte hämtas.'); return response.json(); }),
    window.MODELDOCTOR_API_ENABLED ? fetch(`${API_BASE}/api/status`).then(response => { if (!response.ok) return null; return response.json(); }).catch(() => null) : Promise.resolve(null),
  ]);
  if (statusResult.status === 'fulfilled' && statusResult.value?.api_version === 1) {
    state.api = {...statusResult.value, local: true};
    $('privacy-text').textContent = 'Analys skickas till din lokala server. Du väljer vad som sparas i historiken.';
    $('history-mode').textContent = 'Sparade körningar lagras i din lokala databas. Python-klientens uppdateringar visas här automatiskt.';
    $('save-location').textContent = 'Loggen och angivna inställningar sparas i den lokala serverns databas.';
    $('auth-section').hidden = !state.api.auth_required;
    if (state.api.available) {
      $('model-status').lastChild.textContent = ' Modellstöd tillgängligt';
      $('sidebar-model').firstChild.textContent = 'Modellstöd tillgängligt';
      $('sidebar-model').querySelector('small').textContent = 'Llama-adaptern är tillgänglig';
    }
  }
  if (exampleResult.status === 'fulfilled') {
    state.examples = exampleResult.value;
    for (const example of state.examples) {
      const button = document.createElement('button'); button.className = 'sample-option';
      const icon = svgElement('svg'); icon.append(svgElement('use', {href: '#i-chart'}));
      const text = document.createElement('span'), title = document.createElement('strong'), subtitle = document.createElement('small');
      title.textContent = example.name; subtitle.textContent = `${example.epochs} epoker · ${example.description}`; text.append(title, subtitle);
      const arrow = svgElement('svg'); arrow.append(svgElement('use', {href: '#i-chevron'})); button.append(icon, text, arrow);
      button.addEventListener('click', () => { $('examples-dialog').close(); loadExample(example); }); $('example-list').append(button);
    }
    // Visitors start with an explanation. Demos are opened explicitly.
  } else { $('example-list').textContent='De publicerade loggarna kunde inte hämtas. De illustrerade demoscenarierna fungerar fortfarande.'; }
}
function showView(view) {
  if(view!=='analysis'){stopPlayback();exploration.reveal=null;}
  $('home-view').hidden=view!=='home';$('workspace-view').hidden=view==='home';
  $('analysis-view').hidden=view!=='analysis';$('runs-view').hidden=view!=='runs';
  for(const [id,active] of [['nav-home',view==='home'],['nav-demo',view==='analysis'&&state.viewMode==='demo'],['nav-analysis',view==='analysis'&&state.viewMode==='own'],['nav-runs',view==='runs']]){
    $(id).classList.toggle('active',active);if(active)$(id).setAttribute('aria-current','page');else $(id).removeAttribute('aria-current');
  }
  const demo=state.viewMode==='demo';
  $('demo-selector').hidden=!demo;$('own-setup').hidden=demo;
  $('mode-demo').setAttribute('aria-pressed',String(demo));$('mode-own').setAttribute('aria-pressed',String(!demo));
  document.querySelector('.mode-switch').hidden=view==='runs';
  $('workspace-eyebrow').textContent=view==='runs'?'LÄR AV DINA KÖRNINGAR':demo?'PROVA UTAN EGEN DATA':'DIN EGEN TRÄNING';
  $('workspace-title').textContent=view==='runs'?'Från en körning till nästa.':demo?'Se vad kurvorna berättar.':'Lägg till din träningslogg.';
  $('workspace-description').textContent=view==='runs'?'Jämför resultaten och inställningarna efter en ändring.':demo?'Välj ett scenario. Spela upp träningen och utforska bedömningen.':'Börja med en fil eller välj automatisk insamling från Python.';
  $('empty-analysis').hidden=Boolean(state.report);$('analysis-results').hidden=!state.report;
  if(view==='analysis'&&state.report)renderChart();
  if(view==='home')window.scrollTo({top:0,behavior:'smooth'});
}
function openOwn() {
  state.version++;
  if(state.viewMode==='demo'){
    const draft=state.ownDraft;clearResult();if(draft){Object.assign(state,draft);$('architecture').value=state.report.architecture;}
  }
  state.viewMode='own';showView('analysis');selectInputRoute('upload');if(state.report)render();window.scrollTo({top:0,behavior:'smooth'});
}
function selectInputRoute(route) {
  $('upload-route').hidden=route!=='upload';$('automatic-route').hidden=route!=='python';
  $('input-upload').setAttribute('aria-pressed',String(route==='upload'));$('input-python').setAttribute('aria-pressed',String(route==='python'));
}
function stopPlayback() {
  clearInterval(exploration.timer);exploration.timer=null;exploration.playing=false;
  if($('demo-play'))$('demo-play').querySelector('span').textContent='Spela upp';
}
function playDemo() {
  if(!state.example?.illustrative||!state.report)return;
  if(exploration.playing){stopPlayback();return;}
  state.chart='loss';document.querySelector('[data-chart="loss"]').click();
  exploration.window='all';$('chart-window').value='all';exploration.hidden.clear();
  if(exploration.reveal===null||exploration.reveal>=state.report.num_epochs-1)exploration.reveal=0;
  exploration.index=exploration.reveal;exploration.playing=true;$('demo-play').querySelector('span').textContent='Pausa';render();
  exploration.timer=setInterval(()=>{
    if(!state.example?.illustrative||!state.report){stopPlayback();return;}
    exploration.reveal=Math.min(state.report.num_epochs-1,exploration.reveal+1);exploration.index=exploration.reveal;
    $('demo-progress').hidden=false;$('demo-progress').textContent='Träningsrunda '+state.report.epochs[exploration.reveal]+' av '+state.report.epochs.at(-1)+'. Bedömningen gäller hela scenariot.';
    renderChart();if(exploration.reveal===state.report.num_epochs-1)stopPlayback();
  },220);
}

function currentPlan() {
  return reviewPlan(state.report,{exampleCategory:state.example?.category,area:$('review-area').value,task:$('task-context').value});
}
function updateProgress(checks) {
  const count=checks.filter(check=>exploration.reviewed.has(check.id)).length;
  $('checklist-progress').textContent=`${count} av ${checks.length} kontroller granskade`;
  $('checklist-bar').style.width=`${checks.length ? count/checks.length*100 : 0}%`;
}
function renderGuidance() {
  const plan=currentPlan();
  const best=state.report.inspection?.best_validation;
  $('focus-best').disabled=!best;
  $('focus-best').firstChild.textContent=best?'Visa bästa rundan · '+best.epoch:'Valideringsmätningar saknas';
  const sourceCopy={observed:'Kontrollerna börjar med mätningar i loggen.',model_category:'Modellens tekniska förslag väljer ytterligare kontrollsteg. Kontrollera förslaget innan du ändrar något.',published_example:'De extra stegen hör till originalexperimentets ändring. Ingen modellprediktion har gjorts.',user_selected:'Du valde vad du vill granska. Det ändrar bara kontrollstegen.',general:'Välj ett område för fler kontroller. Uppgiften anpassar bara guiderna.'};
  $('guidance-source').textContent=sourceCopy[plan.source];
  $('guidance-checks').replaceChildren();
  const active=new Set(plan.checks.map(check=>check.id));
  for(const id of exploration.reviewed)if(!active.has(id))exploration.reviewed.delete(id);
  for(const check of plan.checks) {
    const item=document.createElement('article');item.className='guide-check';item.classList.toggle('reviewed',exploration.reviewed.has(check.id));
    const heading=document.createElement('div');heading.className='check-heading';
    const input=document.createElement('input'),label=document.createElement('label');input.type='checkbox';input.id=`check-${check.id}`;input.checked=exploration.reviewed.has(check.id);
    input.setAttribute('aria-label',`Granskad: ${check.title}`);label.htmlFor=input.id;label.textContent=check.title;heading.append(input,label);
    input.addEventListener('change',()=>{if(input.checked)exploration.reviewed.add(check.id);else exploration.reviewed.delete(check.id);item.classList.toggle('reviewed',input.checked);updateProgress(plan.checks);});
    const origin=document.createElement('span');origin.className='check-origin';
    origin.textContent={observed:'Från en observation i loggen',model_category:'Guide för modellens föreslagna område',published_example:'Guide för exemplets facit',user_selected:'Guide för ditt valda område',general:'Allmän kontroll'}[check.source];
    const details=document.createElement('details'),summary=document.createElement('summary');summary.textContent='Så gör du kontrollen';details.open=exploration.expanded.has(check.id);
    details.addEventListener('toggle',()=>{if(!item.isConnected)return;if(details.open)exploration.expanded.add(check.id);else exploration.expanded.delete(check.id);});details.append(summary);
    for (const [title,text] of [['Varför börja här?',check.why],['Kontrollera',check.how],['Följ upp',check.verify]]) {
      const section=document.createElement('div'),strong=document.createElement('strong'),paragraph=document.createElement('p');section.className='check-section';strong.textContent=title;paragraph.textContent=text;section.append(strong,paragraph);details.append(section);
    }
    if(check.links.length) {const links=document.createElement('div');links.className='check-links';for(const source of check.links){const a=document.createElement('a');a.href=source.url;a.target='_blank';a.rel='noopener noreferrer';a.textContent=source.title+' ↗';links.append(a);}details.append(links);}
    item.append(heading,origin,details);$('guidance-checks').append(item);
  }
  updateProgress(plan.checks);
}
function renderInspection() {
  $('inspection-findings').replaceChildren();
  const inspection=state.report.inspection;
  if(inspection?.best_validation) {
    const note=document.createElement('div');note.className='best-note';
    const text=document.createElement('span');text.textContent='Lägsta registrerade valideringsloss';
    const value=document.createElement('strong');value.textContent=`${number(inspection.best_validation.value)} · epok ${inspection.best_validation.epoch}`;note.append(text,value);$('inspection-findings').append(note);
  }
  const percent=value=>`${new Intl.NumberFormat('sv-SE',{maximumFractionDigits:1}).format(value*100)} %`;
  for(const finding of inspection?.findings || []) {
    const item=document.createElement('article');item.className=`finding ${finding.level}`;
    const title=document.createElement('h3');title.className='finding-title';
    const dot=document.createElement('span');dot.className='status-dot';title.append(dot,document.createTextNode(finding.title));
    const e=finding.evidence, metrics={
      validation_regression:[['Bästa loss',number(e.best_loss)],['Sista loss',number(e.final_loss)],['Ökning från bästa',percent(e.relative_increase)]],
      flat_training_loss:[['Epoker',(e.epochs || []).join(', ')],['Relativ variation',percent(e.relative_range)]],
      zero_logged_gradient:[['Epoker',(e.epochs || []).join(', ')],['Maxgradient',(e.values || []).map(number).join(' · ')]],
      missing_validation:[['Valideringsmätningar',`${e.observed_epochs} av ${e.total_epochs} epoker`]],
      missing_measurements:[['Saknade värden',`${e.missing_values} av ${e.total_values}`],['Saknad andel',percent(e.missing_rate)]],
    }[finding.id] || [];
    const measured=document.createElement('div');measured.className='finding-metrics';
    for(const [label,value] of metrics){const chip=document.createElement('span'),strong=document.createElement('strong');strong.textContent=value;chip.append(document.createTextNode(label),strong);measured.append(chip);}
    const text=document.createElement('p');text.textContent=finding.next_step;
    const details=document.createElement('details'),summary=document.createElement('summary'),evidence=document.createElement('pre');summary.textContent='Så mäts kontrollen';evidence.textContent=JSON.stringify(finding.evidence,null,2);details.append(summary,evidence);
    item.append(title,measured,text,details);$('inspection-findings').append(item);
  }
  if(!inspection?.findings?.length){const note=document.createElement('p');note.className='section-intro';note.textContent='Inga av de tillgängliga kontrollerna gav utslag. Granska även resultat och kod; det fastställer inte att träningen är felfri.';$('inspection-findings').append(note);}
}

async function apiRequest(path, options = {}) {
  const response = await fetch(`${API_BASE}${path}`, {...options,headers:{'Content-Type':'application/json',...(state.token ? {Authorization:`Bearer ${state.token}`} : {}),...options.headers}});
  const result = await response.json();
  if (!response.ok) throw new Error(result.error || 'Servern kunde inte behandla anropet.');
  return result;
}

async function getRun(id) { return state.api.local ? apiRequest(`/api/runs/${id}`) : browserRuns.get(id); }

let refreshing = false;
async function refreshRuns() {
  if (refreshing) return;
  refreshing = true;
  try {
    state.runs = state.api.local ? (await apiRequest('/api/runs')).runs : await browserRuns.list();
    $('history-error').hidden = true;
    const project = $('project-filter').value, projects = [...new Set(state.runs.map(run=>run.project))].sort();
    $('project-filter').replaceChildren(new Option('Alla projekt',''), ...projects.map(name=>new Option(name,name)));
    if (projects.includes(project)) $('project-filter').value = project;
    for (const id of state.selected) if (!state.runs.some(run=>run.run_id === id)) state.selected.delete(id);
    renderRuns();
    const current = state.currentRun;
    if (current?.status === 'running') {
      const changed = state.runs.find(run=>run.run_id === current.run_id);
      if (changed && changed.revision !== current.revision) {
        const run = await getRun(current.run_id);
        if (state.currentRun?.run_id === current.run_id) { exploration.reviewed.clear(); exploration.expanded.clear(); Object.assign(state,{currentRun:run,csv:run.csv,report:run.report,name:run.name}); render(); }
      }
    }
  } catch (error) { $('history-error').textContent = error.message; $('history-error').hidden = false; }
  finally { refreshing = false; }
}

function renderRuns() {
  const filtered = state.runs.filter(run=>!$('project-filter').value || run.project === $('project-filter').value);
  $('nav-run-count').textContent = state.runs.length;
  $('history-empty').hidden = Boolean(filtered.length); $('runs-table').hidden = !filtered.length;
  $('compare-runs').disabled = state.selected.size !== 2; $('runs-body').replaceChildren();
  for (const run of filtered) {
    const row = document.createElement('tr');
    const selectCell = document.createElement('td'), checkbox = document.createElement('input');
    checkbox.type = 'checkbox'; checkbox.checked = state.selected.has(run.run_id);
    checkbox.disabled = !checkbox.checked && state.selected.size >= 2;
    checkbox.setAttribute('aria-label',`Välj ${run.name} för jämförelse`);
    checkbox.addEventListener('change',()=>{ if (checkbox.checked) state.selected.add(run.run_id); else state.selected.delete(run.run_id); renderRuns(); });
    selectCell.append(checkbox); row.append(selectCell);
    const nameCell = document.createElement('td'), name = document.createElement('strong'), meta = document.createElement('small');
    name.textContent = run.name; meta.textContent = `${run.project} · ${run.architecture} · ${new Date(run.updated_at).toLocaleString('sv-SE')}`;
    nameCell.append(name,meta); row.append(nameCell);
    const statusCell = document.createElement('td'), badge = document.createElement('span'); badge.className = `status-tag ${run.status}`;
    badge.textContent = {running:'PÅGÅR',completed:'KLAR',failed:'AVBRUTEN'}[run.status]; statusCell.append(badge); row.append(statusCell);
    for (const value of [run.num_epochs ?? run.report?.num_epochs,number(run.final?.val_loss ?? run.report?.final.val_loss),run.findings ?? run.report?.inspection.findings.length]) {
      const cell = document.createElement('td'); cell.textContent = String(value); row.append(cell);
    }
    const openCell = document.createElement('td'), open = document.createElement('button'); open.className = 'button button-quiet'; open.textContent = 'Öppna';
    open.setAttribute('aria-label',`Öppna ${run.name}`); open.addEventListener('click',()=>openRun(run.run_id)); openCell.append(open); row.append(openCell);
    const deleteCell = document.createElement('td'), remove = document.createElement('button'); remove.className = 'icon-button'; remove.textContent = '×'; remove.setAttribute('aria-label',`Ta bort ${run.name}`);
    remove.addEventListener('click',()=>{ pendingDelete = run; $('delete-description').textContent = `”${run.name}” tas bort från ${state.api.local ? 'den lokala databasen' : 'den här webbläsarens historik'}. Klientens eventuella lokala loggkopior påverkas inte.`; showDialog('delete-dialog'); }); deleteCell.append(remove); row.append(deleteCell);
    $('runs-body').append(row);
  }
}

async function openRun(id) {
  const version = ++state.version;
  try {
    const run = await getRun(id);
    if (state.version !== version) return;
    Object.assign(state,{currentRun:run,csv:run.csv,report:run.report,name:run.name,example:null});
    state.viewMode='own';state.originalCsv='';state.importInfo=null;resetExploration();
    $('architecture').value = run.architecture; clearError(); showView('analysis'); render();
  } catch (error) { toast(error.message); }
}

function drawComparison(a, b) {
  const chart = $('comparison-chart'), width = Math.max(300,Math.min(780,$('comparison-panel').clientWidth - 36)), height = width < 500 ? 240 : 285;
  chart.setAttribute('viewBox',`0 0 ${width} ${height}`); chart.replaceChildren(); $('comparison-legend').replaceChildren();
  const series = [[a,'#087f70','A'],[b,'#b66c25','B']], values = series.flatMap(([run])=>run.report.curves.val_loss).filter(value=>value !== null);
  for (const [run,color,label] of series) { const item = document.createElement('span'), dot = document.createElement('i'); dot.className = 'legend-dot'; dot.style.background = color; item.append(dot,document.createTextNode(`${label}: ${run.name}`)); $('comparison-legend').append(item); }
  if (!values.length) { chart.append(svgElement('text',{x:width/2,y:height/2,'text-anchor':'middle',fill:'#52666d','font-size':12},'Valideringsloss saknas.')); return; }
  const left = 65, right = 20, top = 20, bottom = 40;
  let low = Math.min(...values), high = Math.max(...values); const padding = (high - low || Math.abs(high) || 1) * .1; low -= padding; high += padding;
  const start = Math.min(a.report.epochs[0],b.report.epochs[0]), end = Math.max(a.report.epochs.at(-1),b.report.epochs.at(-1));
  const x = epoch=>left+(epoch-start)/Math.max(1,end-start)*(width-left-right), y = value=>top+(high-value)/(high-low)*(height-top-bottom);
  chart.append(svgElement('title',{},`Valideringsloss: A ${a.name}, B ${b.name}. Epok ${start} till ${end}.`));
  for (let i=0;i<=4;i++) { const value = low+(high-low)*i/4; chart.append(svgElement('line',{x1:left,x2:width-right,y1:y(value),y2:y(value),stroke:'#dfe8e1'})); chart.append(svgElement('text',{x:left-10,y:y(value)+4,fill:'#52666d','font-size':10,'text-anchor':'end'},number(value))); }
  for (let i=0;i<=4;i++) { const epoch = Math.round(start+(end-start)*i/4); chart.append(svgElement('text',{x:x(epoch),y:height-14,fill:'#52666d','font-size':10,'text-anchor':'middle'},epoch)); }
  for (const [run,color,label] of series) {
    let path='',gap=true;
    run.report.curves.val_loss.forEach((value,i)=>{ if (value === null) gap=true; else { path+=`${gap?'M':'L'}${x(run.report.epochs[i])},${y(value)} `; gap=false; } });
    chart.append(svgElement('path',{d:path,stroke:color,fill:'none','stroke-width':2.3,'stroke-dasharray':label === 'B' ? '5 4' : 'none'}));
  }
}

let compared = null, pendingDelete = null;
$('nav-runs').addEventListener('click',()=>{ showView('runs'); refreshRuns(); });
for (const id of ['connect-button','empty-connect']) $(id).addEventListener('click',()=>showDialog('connect-dialog'));
$('refresh-runs').addEventListener('click',refreshRuns);
$('project-filter').addEventListener('change',()=>{ state.selected.clear(); renderRuns(); });
$('copy-integration').addEventListener('click',async()=>{ try { await navigator.clipboard.writeText($('integration-code').textContent); toast('Exemplet är kopierat.'); } catch { toast('Markera och kopiera kodexemplet.'); } });
$('set-api-token').addEventListener('click',()=>{ state.token = $('api-token').value; $('api-token').value=''; refreshRuns(); $('connect-dialog').close(); });
$('save-run').addEventListener('click',()=>{ if (!state.report) { showError('Välj en giltig logg först.'); return; } $('save-name').value=state.name; $('save-error').hidden=true; showDialog('save-dialog'); });
$('save-form').addEventListener('submit',async event=>{
  event.preventDefault(); const version = state.version;
  try {
    const config = JSON.parse($('save-config').value);
    if (!config || Array.isArray(config) || typeof config !== 'object' || JSON.stringify(config).length > 16384) throw new Error('Konfigurationen måste vara ett JSON-objekt på högst 16 KB.');
    const run = {api_version:1,run_id:crypto.randomUUID(),revision:1,name:$('save-name').value,project:$('save-project').value,
      dataset:$('save-dataset').value,metric_name:$('save-metric').value,config,status:'completed',architecture:state.report.architecture,
      source:state.example?.illustrative?'illustrative-demo':state.example?'demo-example':'upload',signal_schema:'unverified',csv:state.csv};
    let saved;
    if (state.api.local) saved = (await apiRequest('/api/runs',{method:'POST',body:JSON.stringify(run)})).run;
    else { const hash = await crypto.subtle.digest('SHA-256',new TextEncoder().encode(run.csv)); saved = await browserRuns.save({...run,report:state.report,input_sha256:[...new Uint8Array(hash)].map(value=>value.toString(16).padStart(2,'0')).join(''),created_at:new Date().toISOString(),updated_at:new Date().toISOString()}); }
    if (version === state.version) { state.currentRun=saved; state.report=saved.report; render(); }
    $('save-dialog').close(); refreshRuns(); toast('Körningen är sparad.');
  } catch (error) { $('save-error').textContent=error.message; $('save-error').hidden=false; }
});
$('compare-runs').addEventListener('click',async()=>{
  try {
    const selected = [...state.selected]; if (selected.length !== 2) return;
    const [a,b] = await Promise.all(selected.map(getRun)), result = comparison(a,b); compared=[a,b];
    $('comparison-panel').hidden=false; $('comparison-description').textContent='Valideringsloss över epoker, tillsammans med ändrade träningsinställningar.';
    $('comparison-warning').hidden=!result.warnings.length; $('comparison-warning').textContent=result.warnings.join('\n');
    $('comparison-summary').textContent=result.best_delta === null ? 'Båda körningarna behöver valideringsloss för att visa skillnaden.' : `B:s lägsta valideringsloss minus A:s: ${number(result.best_delta)}. Negativt betyder lägre loss för B, förutsatt jämförbara lossdefinitioner och valideringsdata.`;
    $('config-differences').replaceChildren();
    for (const setting of result.settings.length ? result.settings : [{key:'Inga registrerade skillnader',a:null,b:null}]) {
      const row=document.createElement('tr'); for (const value of [setting.key,setting.a,setting.b]) { const cell=document.createElement('td'); cell.textContent=value === null ? '—' : typeof value === 'object' ? JSON.stringify(value) : String(value); row.append(cell); } $('config-differences').append(row);
    }
    drawComparison(a,b); $('comparison-panel').scrollIntoView({behavior:'smooth',block:'start'});
  } catch (error) { $('history-error').textContent=error.message; $('history-error').hidden=false; }
});
$('confirm-delete').addEventListener('click',async()=>{
  if (!pendingDelete) return;
  try { const id=pendingDelete.run_id; if (state.api.local) await apiRequest(`/api/runs/${id}`,{method:'DELETE'}); else await browserRuns.delete(id);
    if (state.currentRun?.run_id === id) { state.currentRun=null; render(); }
    state.selected.delete(id); $('comparison-panel').hidden=true; compared=null; $('delete-dialog').close(); await refreshRuns(); toast('Körningen har tagits bort.');
  } catch (error) { toast(error.message); }
});
for (const button of document.querySelectorAll('[data-feedback]')) button.addEventListener('click',async()=>{
  if (!state.currentRun) return;
  try { await apiRequest(`/api/runs/${state.currentRun.run_id}/feedback`,{method:'POST',body:JSON.stringify({verdict:button.dataset.feedback})}); toast('Återkopplingen är sparad för manuell granskning.'); }
  catch (error) { toast(error.message); }
});
window.addEventListener('resize',()=>{ if (compared && !$('comparison-panel').hidden) drawComparison(...compared); });
initialize().then(refreshRuns);
setInterval(()=>{ if (!$('runs-view').hidden || state.currentRun?.status === 'running') refreshRuns(); },5000);
