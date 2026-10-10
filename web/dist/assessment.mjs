// A readable assessment of logged patterns. Independent of the experimental category classifier.
export function assessTraining(report) {
  const findings=report.inspection?.findings || [],has=id=>findings.some(f=>f.id===id);
  let id='inconclusive',title='Underlaget ger ingen tydlig bedömning',summary='Granska hela förloppet och resultatet på separat data innan du ändrar något.',next='Spara en referenskörning och jämför nästa ändring på samma valideringsdata.',tone='neutral';
  if(has('missing_validation')) {id='missing_validation';title='Resultatet på ny data saknas';summary='Du ser träningens utveckling, men loggen saknar valideringsloss. Därför går det inte att bedöma hur modellen fungerar på separat data.';next='Logga valideringsloss efter varje träningsrunda.';tone='attention';}
  else if(has('zero_logged_gradient')) {id='zero_gradient';title='Gradientmätningen behöver kontrolleras';summary='De senaste tre registrerade maxgradienterna är noll. Kontrollera först när de mäts och om modellens vikter uppdateras.';next='Granska en liten batch före och efter ett optimizer-steg.';tone='attention';}
  else if(has('validation_regression')) {
    const train=report.curves.train_loss,valid=train.filter(Number.isFinite),first=train[0],last=train.at(-1);
    if(Number.isFinite(first)&&Number.isFinite(last)&&last<first&&valid.length>=5) {id='diverging';title='Träningen förbättras, men valideringen försämras';summary='Modellen får lägre fel på träningsdatan medan resultatet på separat data har försämrats från sin bästa nivå. Mönstret kan tyda på överanpassning; loggen bekräftar ingen grundorsak.';}
    else {id='validation_regression';title='Valideringsresultatet har försämrats';summary='Felet på separat data är högre i sista rundan än i den bästa registrerade rundan. Kontrollera om samma mönster återkommer.';}
    next='Jämför den bästa sparade versionen med den sista på samma valideringsdata. Kontrollera också att mätningen är jämförbar.';tone='attention';
  } else if(has('flat_training_loss')) {id='plateau';title='Träningens utveckling har planat ut';summary='Träningsloss ändras mycket lite under de senaste fem rundorna. Det kan vara konvergens, en mätfråga eller ett problem i träningsloopen.';next='Kontrollera att nya värden loggas och att träningsbara vikter faktiskt uppdateras.';}
  else {
    const train=report.curves.train_loss,val=report.curves.val_loss;
    const complete=report.num_epochs>=5 && [...train,...val].every(Number.isFinite);
    if(complete&&train.at(-1)<train[0]&&val.at(-1)<val[0]) {id='improving';title='Både träning och validering förbättras';summary='Båda kurvorna slutar lägre än de börjar. Det visar en förbättring i loggen, men avgör inte om resultatet är tillräckligt bra för din uppgift.';next='Jämför med en referenskörning och dina kvalitetsmål på samma valideringsdata.';tone='positive';}
  }
  return {version:'assessment-1',id,title,summary,next,tone,source:'logged_patterns',scope:'Pattern assessment, not a learned diagnosis or a verified cause.'};
}
