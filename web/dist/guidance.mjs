// Human-written investigation guides. Never training targets, model explanations or verified repairs.
export const AREAS = {
  no_injected_fault: {title:'Ingen injicerad ändring', meaning:'Datasetets baseline: experimentet kördes utan en injicerad ändring. Det garanterar inte bra resultat.'},
  hyperparameter_fault: {title:'Hyperparametrar', meaning:'Learning rate, batchstorlek, antal epoker eller hur batchar byggs. Kategorin anger inte vilken inställning som är fel eller åt vilket håll den bör ändras.'},
  layer_fault: {title:'Lager och arkitektur', meaning:'Nätverkets lager eller deras konfiguration. Kategorin pekar inte ut ett specifikt lager.'},
  loss_fault: {title:'Lossfunktion', meaning:'Valet av lossfunktion. Kategorin säger inte vilken loss du använder, varför den är fel eller vilken du bör byta till.'},
  activation_fault: {title:'Aktiveringsfunktion', meaning:'Aktiveringsfunktioner i nätverket. Kategorin pekar inte ut en viss funktion eller ett lager.'},
  optimization_fault: {title:'Optimizer och gradienter', meaning:'Val av optimizer eller gradient clipping. Kategorin fastställer inte en specifik inställning.'},
  weight_fault: {title:'Viktinitialisering', meaning:'Hur vikter eller bias initialiseras. Kategorin anger ingen uppmätt grundorsak.'},
  regularization_fault: {title:'Regularisering', meaning:'Dropout eller annan regularisering. Kategorin säger inte om styrkan bör öka eller minska.'},
};
const step = (id,title,why,how,verify,source='general',links=[]) => ({id,title,why,how,verify,source,links});
const controlled = 'Behåll samma data, split och övriga inställningar. Jämför valideringsresultat och upprepa med flera seeds innan du drar en slutsats.';
const pytorch = name => ({title:`PyTorch: ${name}`,url:`https://docs.pytorch.org/docs/stable/generated/torch.nn.${name}.html`});

function lossSteps(task,source) {
  let how='Skriv ned uppgiften, lossfunktionen, modellens sista aktivering och target-formatet. Kontrollera att de hör ihop enligt ramverkets dokumentation. Välj uppgift ovan för en mer specifik kontroll.';
  let links=[];
  if (task === 'multiclass') {
    how='Om du använder PyTorch CrossEntropyLoss: mata in logits utan en extra softmax. Kontrollera klassdimensionen. Targets kan vara klassindex eller klassfördelningar; kontrollera rätt shape, dtype och giltiga värden för det format du valt.';
    links=[pytorch('CrossEntropyLoss')];
  } else if (task === 'binary') {
    how='Om du använder PyTorch BCEWithLogitsLoss: mata in logits utan en extra sigmoid. Targets ska ha samma shape som input och värden mellan 0 och 1. Kontrollera även att target-kodningen stämmer för varje output.';
    links=[pytorch('BCEWithLogitsLoss')];
  } else if (task === 'regression') {
    how='Om du använder PyTorch MSELoss: jämför output- och target-shape; de ska matcha. Kontrollera också target-enheter och skalning så att ett annat target-format inte smyger in i träning eller validering.';
    links=[pytorch('MSELoss')];
  }
  return [step('loss_contract','Stämmer loss med uppgiften?', 'En loss-kategori räcker inte för att välja en ersättare. Börja med kontraktet mellan output och targets.',how,'Kör en liten batch, granska output/targets och beräkna loss. Testa en ändring först när kontrollen visar en konkret avvikelse.',source,links),
    step('loss_consistency','Mäts samma loss i träning och validering?','Kurvor går bara att jämföra om beräkningen betyder samma sak.','Kontrollera lossfunktion, reduction, maskning, klassvikter och normalisering i båda stegen. Dokumentera avsiktliga skillnader.','Beräkna båda loss-varianterna på samma lilla batch. Förklara eventuella skillnader innan du jämför två körningar.',source)];
}

function areaSteps(area,task,source) {
  if (area === 'loss_fault') return lossSteps(task,source);
  const guides = {
    hyperparameter_fault:[
      step('effective_settings','Kontrollera de inställningar som faktiskt används','Sparad konfiguration kan skilja sig från körningen efter scheduler, återstart eller ändrade standardvärden.','Läs learning rate från optimizerns parametergrupper under körningen. Jämför med schedulern, den effektiva batchstorleken och antal optimizer-steg. Enbart en kategori visar inte om learning rate är för hög eller låg.','Spara inställningarna och den uppmätta learning rate-kurvan med nästa logg.',source),
      step('one_change','Prova en inställning i taget','En jämförelse hjälper dig avgöra om en ändring påverkar resultatet.','Utgå från en dokumenterad referenskörning. Ändra en motiverad inställning och spara båda körningarnas konfigurationer.',controlled,source)],
    layer_fault:[step('layer_shapes','Granska dimensioner genom nätverket','Arkitekturkategorin pekar inte ut vilket lager du ska ändra.','Kör en liten batch genom modellen. Följ dimensionerna lager för lager, kontrollera output mot targets och att alla avsedda lager deltar i forward-pass.','Verifiera ett forward- och backward-pass innan en längre jämförelsekörning.',source)],
    activation_fault:[step('activation_flow','Följ aktiveringar och gradienter per lager','En aggregerad gradient kan dölja vad som händer i ett enskilt lager.','Mät aktiveringarnas spridning och gradienter för varje lager på en liten batch. Granska även sista aktiveringen tillsammans med lossfunktionen.','Om ett lager avviker: undersök dess input, aktivering och implementation. '+controlled,source)],
    optimization_fault:[step('optimizer_step','Verifiera att optimizer uppdaterar rätt parametrar','En loggkategori visar inte om felet gäller optimizer, clipping eller träningsloopen.','Kontrollera att parametrarna är med i optimizer, att backward körs, att step sker vid rätt tillfälle och att clipping mäts före och efter. Jämför en avsiktligt träningsbar vikt före och efter ett steg.','Bekräfta en viktuppdatering på en liten batch. '+controlled,source)],
    weight_fault:[step('initial_weights','Granska initialisering och återladdning','Kategorin anger ett område att undersöka, inte vilka vikter som är fel.','Kontrollera seed, initialisering per lager, bias och vilka vikter som laddas från checkpoint. Kontrollera att återladdade vikter inte initialiseras om.','Dokumentera viktstatistik före första steget. '+controlled,source)],
    regularization_fault:[step('train_eval_mode','Kontrollera regularisering och eval-läge','Dropout och annan regularisering behöver granskas i den faktiska träningskoden.','Kontrollera dropout-värden, weight decay och övriga regulariseringstermer. Bekräfta att valideringen använder eval-läge och att regularisering inte räknas dubbelt.','Dokumentera avsiktliga skillnader mellan träning och validering. '+controlled,source)],
  };
  return guides[area] || [];
}

export function categoryResult(report,exampleCategory=null) {
  if (report.mode === 'llama' && Object.hasOwn(AREAS,report.diagnosis)) return {area:report.diagnosis,source:'model_category',label:'Modellförslag'};
  if (Object.hasOwn(AREAS,exampleCategory)) return {area:exampleCategory,source:'published_example',label:'Exemplets facit'};
  return null;
}

export function reviewPlan(report,{exampleCategory=null,area='auto',task='unknown'}={}) {
  const category=categoryResult(report,exampleCategory), findings=report.inspection?.findings || [], checks=[];
  const has=id=>findings.some(item=>item.id===id);
  if (has('missing_validation')) checks.push(step('add_validation','Lägg till valideringsmätningar','Loggen saknar valideringsloss, så utvecklingen på separat data kan inte bedömas.','Använd en separat och dokumenterad valideringssplit. Logga dess loss per epok, i eval-läge och med samma definition vid varje mätning.','Kontrollera att nästa logg innehåller valideringsvärden. Jämför utvecklingen över epoker.','observed'));
  if (has('zero_logged_gradient')) checks.push(step('verify_gradients','Verifiera gradientmätningen','De senaste registrerade maxgradienterna är noll. Det kan bero på mätningen eller på träningsloopen.','Kontrollera när gradienterna mäts i förhållande till backward och zero_grad. Granska sedan requires_grad och en avsiktligt träningsbar parameter före och efter optimizer.step.','Logga gradienter vid en dokumenterad tidpunkt och kontrollera att en förväntad vikt uppdateras.','observed'));
  if (has('validation_regression')) checks.push(step('review_best_epoch','Undersök perioden efter bästa epoken','Valideringsloss slutar tydligt högre än sitt lägsta registrerade värde.','Markera bästa epoken i grafen. Kontrollera samma split, lossdefinition och eval-läge genom körningen. Om du sparat checkpoints: jämför checkpointen där med den sista.','Om försämringen återkommer i jämförbara körningar kan du testa checkpointval eller early stopping. Följ upp på validering; använd inte sluttestet för att välja inställning.','observed'));
  if (has('flat_training_loss')) checks.push(step('verify_plateau','Kontrollera varför loss står still','Loss varierar mycket lite i de senaste fem registrerade epokerna.','Kontrollera att loggen mäter ny loss varje epok och att träningsbara parametrar uppdateras. Granska därefter learning rate och målens skalning.','Bekräfta mätning och viktuppdatering innan du ändrar en träningsinställning.','observed'));
  const chosen=area === 'auto' ? category?.area || null : Object.hasOwn(AREAS,area) ? area : null;
  const source=area !== 'auto' ? 'user_selected' : category?.source || 'general';
  checks.push(...areaSteps(chosen,task,source));
  if (!checks.length) checks.push(step('set_reference','Skapa en jämförbar referenskörning','De tillgängliga kontrollerna fastställer inget specifikt fel.','Dokumentera dataset, split, seed, lossdefinition och konfiguration. Välj ett område ovan om du vill granska exempelvis loss eller optimizer.','Spara referensen och nästa körning. '+controlled));
  let headline='Granska utvecklingen innan du ändrar något', description='Loggen ger ingen bekräftad grundorsak. Börja med en jämförbar referens och välj ett område att kontrollera.';
  if (has('missing_validation')) {headline='Valideringsdata saknas i loggen';description='Lägg till valideringsloss för att kunna följa resultatet på separat data.';}
  else if (has('zero_logged_gradient')) {headline='Nollgradienter behöver kontrolleras';description='Börja med att verifiera hur och när gradienterna registreras.';}
  else if (has('validation_regression')) {headline='Valideringsloss försämras efter bästa epoken';description='Jämför bästa och sista epoken. Kontrollera mätningen innan du drar en slutsats om orsaken.';}
  else if (has('flat_training_loss')) {headline='Träningsloss har planat ut';description='Kontrollera mätningen och parameteruppdateringarna innan du ändrar inställningarna.';}
  return {version:'guidance-1',headline,description,category,area:chosen,task,source,checks,
    scope:'Human-written investigation steps. Checked means reviewed, not repaired. No causal diagnosis or validated repair.'};
}
