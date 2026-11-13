(() => {
  const form = document.getElementById('prediction-form');
  if (!form) return;
  const catalogElement = document.getElementById('assessment-catalog');
  const catalog = JSON.parse(catalogElement.textContent);
  const releaseLag = Number(catalogElement.dataset.releaseLag);
  const module = document.getElementById('code_module');
  const presentation = document.getElementById('code_presentation');
  const target = document.getElementById('target_assessment_id');
  const rows = document.getElementById('history-rows');
  const add = document.getElementById('add-history');
  const message = document.getElementById('history-message');
  let nextHistoryRow = 0;
  const label = a => `${a.assessment_type} · day ${a.deadline} · ${a.weight}% weight · #${a.id_assessment}`;
  const offering = () => catalog.filter(a => a.code_module === module.value && a.code_presentation === presentation.value);
  function populate(select, items, chosen = '') {
    select.replaceChildren(new Option('Select an assessment', ''));
    items.forEach(a => select.add(new Option(label(a), String(a.id_assessment))));
    select.value = items.some(a => String(a.id_assessment) === chosen) ? chosen : '';
  }
  function updateHistory() {
    const selectedTarget = offering().find(a => String(a.id_assessment) === target.value);
    // Even an on-time submission cannot be usable until deadline + release lag.
    // Actual submission days are checked again by the server when predicting.
    const eligible = selectedTarget ? offering().filter(a =>
      String(a.id_assessment) !== target.value &&
      a.deadline + releaseLag < selectedTarget.deadline - 1
    ).sort((a, b) => a.deadline - b.deadline || a.id_assessment - b.id_assessment) : [];
    let removed = 0;
    rows.querySelectorAll('.prior-assessment').forEach(select => {
      const chosen = select.dataset.selected || select.value;
      delete select.dataset.selected;
      if (!eligible.length || (chosen && !eligible.some(a => String(a.id_assessment) === chosen))) {
        select.closest('.history-row').remove();
        removed++;
      } else {
        populate(select, eligible, chosen);
      }
    });
    add.disabled = !eligible.length || rows.children.length >= 50;
    if (!selectedTarget) {
      message.textContent = 'Select the assessment to predict before adding previous results.';
    } else if (!eligible.length) {
      message.textContent = `No earlier assessments can meet the ${releaseLag}-day score-availability rule for this target. You can estimate without previous scores.`;
    } else if (removed) {
      message.textContent = 'Results that cannot precede this target were removed. Only eligible earlier assessments are listed.';
    } else {
      message.textContent = 'Only earlier assessments that can meet the score-availability rule are listed. Late submissions may still be excluded.';
    }
  }
  function summary() {
    const a = catalog.find(a => String(a.id_assessment) === target.value);
    document.getElementById('assessment-summary').textContent = a
      ? `Type: ${a.assessment_type} · Weight: ${a.weight}% · Deadline: day ${a.deadline} · Prediction: day ${a.deadline - 1}`
      : 'Choose a module, presentation, and assessment from the OULAD course schedule.';
    updateHistory();
  }
  function updateOffering(clearHistory = false) {
    const options = [...new Set(catalog.filter(a => a.code_module === module.value).map(a => a.code_presentation))].sort();
    const previous = presentation.value;
    [...presentation.options].forEach(option => {
      option.disabled = !!option.value && !options.includes(option.value);
    });
    if (previous && !options.includes(previous)) presentation.value = '';
    populate(target, offering(), target.dataset.selected || target.value);
    delete target.dataset.selected;
    if (clearHistory && rows.children.length) {
      rows.replaceChildren();
      message.textContent = 'Previous results were cleared because the course offering changed.';
    }
    summary();
  }
  module.addEventListener('change', () => updateOffering(true));
  presentation.addEventListener('change', () => updateOffering(true));
  target.addEventListener('change', summary);
  add.addEventListener('click', () => {
    if (add.disabled || rows.children.length >= 50) return;
    const fragment = document.getElementById('history-template').content.cloneNode(true);
    const rowKey = `new-${nextHistoryRow++}`;
    fragment.querySelectorAll('[id], [aria-describedby]').forEach(element => {
      ['id', 'aria-describedby'].forEach(attribute => {
        if (element.hasAttribute(attribute)) {
          element.setAttribute(attribute, element.getAttribute(attribute).replaceAll('__ROW__', rowKey));
        }
      });
    });
    rows.append(fragment);
    updateHistory();
    rows.lastElementChild.querySelector('select').focus();
  });
  rows.addEventListener('click', event => {
    if (event.target.classList.contains('remove-row')) {
      event.target.closest('.history-row').remove();
      updateHistory();
    }
  });
  updateOffering();
})();
