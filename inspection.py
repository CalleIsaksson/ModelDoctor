"""Evidence-based inspection, separate from the learned fault classifier."""

VERSION = "inspection-1"


def inspect_report(report):
    findings = []
    train, validation = report["curves"]["train_loss"], report["curves"]["val_loss"]
    valid = [(i, value) for i, value in enumerate(validation) if value is not None]
    best = None
    if valid:
        index, value = min(valid, key=lambda item: item[1])
        best = {"epoch": report["epochs"][index], "value": value}
        last = validation[-1]
        if len(valid) >= 5 and last is not None and index <= len(validation) - 3:
            increase = (last - value) / max(abs(value), 1e-12)
            if increase >= .15:
                findings.append({"id": "validation_regression", "level": "attention",
                    "title": "Valideringsloss har ökat från sin bästa nivå",
                    "evidence": {"best_epoch": best["epoch"], "best_loss": value,
                                 "final_loss": last, "relative_increase": increase, "threshold": .15},
                    "next_step": "Jämför checkpointen från bästa epoken med den sista på samma valideringsdata. Kontrollera om mönstret återkommer i en ny körning."})
    else:
        findings.append({"id": "missing_validation", "level": "attention",
            "title": "Valideringsloss saknas",
            "evidence": {"observed_epochs": 0, "total_epochs": report["num_epochs"]},
            "next_step": "Logga loss på separat valideringsdata för att kunna följa generalisering och jämföra checkpoints."})
    window = train[-5:]
    if len(train) >= 8 and all(value is not None for value in window):
        relative_range = (max(window) - min(window)) / max(abs(window[0]), 1e-12)
        if relative_range < .01:
            findings.append({"id": "flat_training_loss", "level": "info",
                "title": "Träningsloss förändras lite under de senaste fem epokerna",
                "evidence": {"epochs": report["epochs"][-5:], "relative_range": relative_range, "threshold": .01},
                "next_step": "Kontrollera om valideringen också har stabiliserats. Granska learning rate och att parametrarna uppdateras innan du ändrar träningen."})
    gradients = report["signals"]["gradient_max"][-3:]
    if len(gradients) == 3 and all(value == 0 for value in gradients):
        findings.append({"id": "zero_logged_gradient", "level": "attention",
            "title": "Registrerad maxgradient är noll i tre epoker",
            "evidence": {"epochs": report["epochs"][-3:], "values": gradients},
            "next_step": "Kontrollera när gradienterna mäts, att backward körs och vilka parametrar som har requires_grad. Loggen fastställer ingen orsak."})
    if report["data_quality"]["missing_values"]:
        findings.append({"id": "missing_measurements", "level": "info",
            "title": "En del mätningar saknas",
            "evidence": report["data_quality"].copy(),
            "next_step": "Komplettera instrumenteringen där det behövs. Tomma mätningar behandlas som saknade värden och ersätts inte med noll."})
    return {"version": VERSION, "best_validation": best, "findings": findings,
            "scope": "Observed patterns and checks to investigate; no causal diagnosis or validated repair."}
