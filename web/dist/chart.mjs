const $=id=>document.getElementById(id);
export function renderTrainingChart(report,view,{number,svgElement,toggle,select}) {
  const all = report.epochs.map((_,i)=>i), indices=view.window === 'last10' ? all.slice(-10) : all;
  const start=indices[0], end=indices.at(-1);
  view.index=Math.max(start,Math.min(end,view.index ?? end));
  const definitions={
    loss:[['Träning',report.curves.train_loss,'#087f70'],['Validering',report.curves.val_loss,'#b66c25']],
    metric:[['Träning',report.curves.train_acc,'#087f70'],['Validering',report.curves.val_acc,'#b66c25']],
    gradient:[['Medelgradient',report.signals.mean_gradient,'#087f70'],['Maxgradient',report.signals.gradient_max,'#b66c25']],
    lr:[['Learning rate',report.signals.adjusted_lr,'#087f70']],
  };
  const allSeries=definitions[view.chart], series=allSeries.filter(([name])=>!view.hidden.has(`${view.chart}:${name}`));
  const titles={loss:'Loss',metric:'Registrerat mått',gradient:'Gradienter',lr:'Learning rate'};
  const subtitles={loss:'Klicka i grafen eller välj en träningsrunda.',metric:'Det registrerade måttet visas som loggat.',gradient:'Uppmätta gradienter. Mättidpunkten påverkar tolkningen.',lr:'Uppmätta träningssteg. Tomma fält förblir saknade värden.'};
  $('chart-help').textContent={loss:'Loss är ett mått på modellens fel. Lägre är bättre inom samma jämförbara mätning. Jämför träning och validering.',metric:'Det här är måttet din logg använder, till exempel accuracy eller ett regressionsmått. Vi visar originalvärdena utan att anta att högre alltid är bättre.',gradient:'Gradienter styr hur vikterna ändras. Noll eller stora värden kan behöva granskas, men mätmetoden och tidpunkten spelar roll.',lr:'Learning rate bestämmer hur stora träningsstegen är. Loggen visar de värden som registrerats; den avgör inte ensam vilket värde som är rätt.'}[view.chart];
  $('chart-subtitle').textContent=subtitles[view.chart];
  $('chart-legend').replaceChildren();
  for (const [name,,color] of allSeries) {
    const button=document.createElement('button'),dot=document.createElement('i');
    button.type='button';button.className='series-toggle';button.setAttribute('aria-pressed',String(!view.hidden.has(`${view.chart}:${name}`)));
    button.setAttribute('aria-label',`Visa kurvan ${name}`);dot.className='legend-dot';dot.style.background=color;
    button.append(dot,document.createTextNode(name));button.addEventListener('click',()=>toggle(`${view.chart}:${name}`,allSeries.length));$('chart-legend').append(button);
  }
  const chart=$('training-chart'),width=Math.max(300,Math.min(1000,$('chart-container').clientWidth-20));
  const height=width<500?275:340,left=60,right=22,top=25,bottom=45;
  const plotWidth=width-left-right,plotHeight=height-top-bottom;
  chart.replaceChildren();chart.setAttribute('viewBox',`0 0 ${width} ${height}`);
  chart.setAttribute('aria-label',`${titles[view.chart]} över epok ${report.epochs[start]} till ${report.epochs[end]}. Använd epokreglaget nedanför för exakta värden.`);
  $('chart-range').textContent=`Runda ${report.epochs[start]}–${report.epochs[end]}`;
  const scrubber=$('epoch-scrubber');scrubber.min=start;scrubber.max=end;scrubber.value=view.index;
  scrubber.setAttribute('aria-valuetext',`Epok ${report.epochs[view.index]}`);scrubber.disabled=start===end;
  $('selected-epoch').textContent=report.epochs[view.index];$('epoch-values').replaceChildren();
  for (const [name,values] of series) {
    const item=document.createElement('span'),value=document.createElement('strong');value.textContent=number(values[view.index]);
    item.append(document.createTextNode(name),value);$('epoch-values').append(item);
  }
  chart.onpointermove=null;chart.onpointerleave=null;chart.onclick=null;$('chart-tooltip').hidden=true;
  const finite=series.flatMap(([,values])=>indices.map(i=>values[i])).filter(Number.isFinite);
  if (!finite.length) {
    chart.append(svgElement('text',{x:width/2,y:height/2,'text-anchor':'middle',fill:'#52666d','font-size':13},'Inga uppmätta värden i det här intervallet.'));
    return;
  }
  let min=Math.min(...finite),max=Math.max(...finite);const padding=(max-min||Math.abs(max)||1)*.14;
  min-=padding;max+=padding;if(finite.every(value=>value>=0)&&min<0)min=0;
  const firstEpoch=report.epochs[start],lastEpoch=report.epochs[end];
  const x=epoch=>left+(firstEpoch===lastEpoch ? .5 : (epoch-firstEpoch)/(lastEpoch-firstEpoch))*plotWidth;
  const y=value=>top+(max-value)/(max-min)*plotHeight;
  chart.append(svgElement('title',{},chart.getAttribute('aria-label')));
  for(let i=0;i<=4;i++) {
    const value=min+(max-min)*i/4,position=y(value);
    chart.append(svgElement('line',{x1:left,x2:width-right,y1:position,y2:position,stroke:'#dfe8e1','stroke-width':1,'stroke-dasharray':'3 5'}));
    const tick=Math.abs(value)>=1e5||(Math.abs(value)<.001&&value!==0)?value.toExponential(1):new Intl.NumberFormat('sv-SE',{maximumFractionDigits:max-min<.1?4:2}).format(value);
    chart.append(svgElement('text',{x:left-11,y:position+4,fill:'#52666d','font-size':10,'font-family':'Consolas, monospace','text-anchor':'end'},tick));
  }
  const ticks=[...new Set(Array.from({length:6},(_,i)=>Math.round(firstEpoch+(lastEpoch-firstEpoch)*i/5)))];
  for (const epoch of ticks) chart.append(svgElement('text',{x:x(epoch),y:height-23,fill:'#52666d','font-size':10,'font-family':'Consolas, monospace','text-anchor':'middle'},epoch));
  chart.append(svgElement('text',{x:width/2,y:height-3,fill:'#60757d','font-size':9,'text-anchor':'middle'},'TRÄNINGSRUNDA'));
  for(const [name,values,color] of series) {
    let path='',gap=true;
    for(const i of indices){if(view.reveal!==null&&view.reveal!==undefined&&i>view.reveal)break;const value=values[i];if(!Number.isFinite(value))gap=true;else {path+=`${gap?'M':'L'}${x(report.epochs[i])},${y(value)} `;gap=false;}}
    chart.append(svgElement('path',{d:path,stroke:color,fill:'none','stroke-width':2.5,'stroke-linecap':'round','stroke-linejoin':'round','stroke-dasharray':name==='Validering'||name==='Maxgradient'?'6 3':'none'}));
  }
  const best=report.inspection?.best_validation;
  if(view.chart==='loss'&&best&&best.epoch>=firstEpoch&&best.epoch<=lastEpoch&&(view.reveal===null||view.reveal===undefined||report.epochs[view.reveal]>=best.epoch)&&!view.hidden.has('loss:Validering')) {
    chart.append(svgElement('circle',{cx:x(best.epoch),cy:y(best.value),r:5,fill:'#fff',stroke:'#b66c25','stroke-width':2}));
    chart.append(svgElement('text',{x:Math.min(width-right-75,Math.max(left+10,x(best.epoch)+10)),y:Math.max(top+10,y(best.value)-12),fill:'#98601f','font-size':10},`Bäst · runda ${best.epoch}`));
  }
  const selectedX=x(report.epochs[view.index]);
  chart.append(svgElement('line',{x1:selectedX,x2:selectedX,y1:top,y2:height-bottom,stroke:'#79998a','stroke-width':1,'stroke-dasharray':'4 4'}));
  for(const [,values,color] of series)if(Number.isFinite(values[view.index]))chart.append(svgElement('circle',{cx:selectedX,cy:y(values[view.index]),r:4,fill:color,stroke:'#fff','stroke-width':2}));
  const nearest=event=> {
    const rect=chart.getBoundingClientRect(),position=(event.clientX-rect.left)/rect.width*width;
    const epoch=firstEpoch+Math.max(0,Math.min(1,(position-left)/plotWidth))*(lastEpoch-firstEpoch);
    return indices.reduce((best,i)=>Math.abs(report.epochs[i]-epoch)<Math.abs(report.epochs[best]-epoch)?i:best,start);
  };
  chart.onclick=event=>select(nearest(event));
  chart.onpointermove=event=>{
    const index=nearest(event),rect=chart.getBoundingClientRect(),tooltip=$('chart-tooltip');
    tooltip.textContent=`Träningsrunda ${report.epochs[index]}\n${series.map(([name,values])=>`${name}: ${number(values[index])}`).join('\n')}\nKlicka för att välja`;
    tooltip.hidden=false;tooltip.style.left=`${Math.max(0,Math.min(rect.width-180,event.clientX-rect.left+14))}px`;tooltip.style.top='24px';
  };
  chart.onpointerleave=()=>{$('chart-tooltip').hidden=true;};
}
