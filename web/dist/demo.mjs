import {REQUIRED_COLUMNS} from './csv.mjs';
// Illustrative UI scenarios, kept entirely outside training/evaluation data and answer keys.
export const DEMOS = [
  {id:'diverging',name:'Bättre på träning. Sämre på ny data.',short:'Kurvorna går åt olika håll',description:'Se varför mer träning inte alltid ger bättre resultat.',tone:'attention'},
  {id:'improving',name:'Båda resultaten förbättras.',short:'Båda kurvorna sjunker',description:'Se hur förbättring på träning och separat data kan se ut.',tone:'positive'},
  {id:'plateau',name:'Träningen har stannat av.',short:'Utvecklingen planar ut',description:'Se vad du kan kontrollera när resultaten slutar förändras.',tone:'neutral'},
];
export function makeDemo(id) {
  const demo=DEMOS.find(item=>item.id===id);if(!demo)throw new Error('Okänt demoexempel.');
  const rows=Array.from({length:24},(_,i)=>{
    const train=id==='plateau' ? i<8 ? 1.15-i*.06 : .67 : 1.2*Math.exp(-i/9)+.13;
    const validation=id==='diverging' ? i<=8 ? 1.35-i*.105 : .51+(i-8)*.043 : id==='plateau' ? i<8 ? 1.26-i*.05 : .86 : 1.25*Math.exp(-i/10)+.22;
    const values={epoch:String(i),train_loss:train.toFixed(4),val_loss:validation.toFixed(4)};
    return REQUIRED_COLUMNS.map(name=>values[name] ?? '').join(',');
  });
  return {...demo,illustrative:true,architecture:'unknown',file:`modeldoctor-demo-${id}.csv`,epochs:24,csv:`${REQUIRED_COLUMNS.join(',')}\n${rows.join('\n')}\n`};
}
