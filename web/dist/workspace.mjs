export class BrowserRuns {
  async database() {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open('modeldoctor-pilot-v1', 1);
      request.onupgradeneeded = () => request.result.createObjectStore('runs', {keyPath:'run_id'});
      request.onsuccess = () => resolve(request.result);
      request.onerror = () => reject(new Error('Webbläsaren kunde inte öppna historiken.'));
    });
  }
  async transaction(mode, operation) {
    const database = await this.database();
    return new Promise((resolve, reject) => {
      const transaction = database.transaction('runs', mode);
      const request = operation(transaction.objectStore('runs'));
      transaction.oncomplete = () => { database.close(); resolve(request.result); };
      transaction.onerror = () => { database.close(); reject(new Error('Historiken kunde inte sparas. Kontrollera webbläsarens lagringsutrymme.')); };
      transaction.onabort = transaction.onerror;
    });
  }
  async list() { return (await this.transaction('readonly', store => store.getAll())).sort((a,b) => b.updated_at.localeCompare(a.updated_at)); }
  async get(id) { const run = await this.transaction('readonly', store => store.get(id)); if (!run) throw new Error('Körningen finns inte.'); return run; }
  async save(run) { await this.transaction('readwrite', store => store.put(run)); return run; }
  async delete(id) { await this.transaction('readwrite', store => store.delete(id)); }
}

export function comparison(a, b) {
  const warnings = [];
  for (const [key, title] of [['dataset','dataset'],['architecture','arkitektur'],['metric_name','mått']]) {
    if (a[key] !== b[key]) warnings.push(`Körningarna har olika ${title}.`);
  }
  if ([a.dataset,b.dataset].some(value => !value || value === 'unspecified')) warnings.push('Dataset saknas i metadata. Kontrollera att samma data och valideringssplit används.');
  if (a.project !== b.project) warnings.push('Körningarna kommer från olika projekt.');
  const keys = [...new Set([...Object.keys(a.config || {}), ...Object.keys(b.config || {})])].sort();
  const settings = keys.filter(key => JSON.stringify(a.config?.[key]) !== JSON.stringify(b.config?.[key]))
    .map(key => ({key,a:a.config?.[key] ?? null,b:b.config?.[key] ?? null}));
  const av = a.report.inspection.best_validation, bv = b.report.inspection.best_validation;
  return {warnings, settings, best_delta:av && bv ? bv.value - av.value : null,
    interpretation:'Skillnader beskriver körningarna. De bevisar inte att en ändrad inställning orsakade resultatet.'};
}
