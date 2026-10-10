// Explicit, versioned observations. These never masquerade as model predictions.
export function inspectReport(report) {
  const findings = [], train = report.curves.train_loss, validation = report.curves.val_loss;
  const valid = validation.map((value, i) => [i, value]).filter(([, value]) => value !== null);
  let best = null;
  if (valid.length) {
    const [index, value] = valid.reduce((current, item) => item[1] < current[1] ? item : current);
    best = {epoch: report.epochs[index], value};
    const last = validation.at(-1), increase = last === null ? 0 : (last - value) / Math.max(Math.abs(value), 1e-12);
    if (valid.length >= 5 && last !== null && index <= validation.length - 3 && increase >= .15)
      findings.push({id:'validation_regression', level:'attention', title:'Valideringsloss har ökat från sin bästa nivå',
        evidence:{best_epoch:best.epoch,best_loss:value,final_loss:last,relative_increase:increase,threshold:.15},
        next_step:'Jämför checkpointen från bästa epoken med den sista på samma valideringsdata. Kontrollera om mönstret återkommer i en ny körning.'});
  } else findings.push({id:'missing_validation',level:'attention',title:'Valideringsloss saknas',
    evidence:{observed_epochs:0,total_epochs:report.num_epochs},
    next_step:'Logga loss på separat valideringsdata för att kunna följa generalisering och jämföra checkpoints.'});
  const window = train.slice(-5);
  if (train.length >= 8 && window.every(value => value !== null)) {
    const relativeRange = (Math.max(...window) - Math.min(...window)) / Math.max(Math.abs(window[0]), 1e-12);
    if (relativeRange < .01) findings.push({id:'flat_training_loss',level:'info',title:'Träningsloss förändras lite under de senaste fem epokerna',
      evidence:{epochs:report.epochs.slice(-5),relative_range:relativeRange,threshold:.01},
      next_step:'Kontrollera om valideringen också har stabiliserats. Granska learning rate och att parametrarna uppdateras innan du ändrar träningen.'});
  }
  const gradients = report.signals.gradient_max.slice(-3);
  if (gradients.length === 3 && gradients.every(value => value === 0)) findings.push({id:'zero_logged_gradient',level:'attention',title:'Registrerad maxgradient är noll i tre epoker',
    evidence:{epochs:report.epochs.slice(-3),values:gradients},
    next_step:'Kontrollera när gradienterna mäts, att backward körs och vilka parametrar som har requires_grad. Loggen fastställer ingen orsak.'});
  if (report.data_quality.missing_values) findings.push({id:'missing_measurements',level:'info',title:'En del mätningar saknas',evidence:{...report.data_quality},
    next_step:'Komplettera instrumenteringen där det behövs. Tomma mätningar behandlas som saknade värden och ersätts inte med noll.'});
  return {version:'inspection-1',best_validation:best,findings,scope:'Observed patterns and checks to investigate; no causal diagnosis or validated repair.'};
}
