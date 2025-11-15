(function(){
  const SERVER_URL = "https://aging-analyzer.onrender.com/analyze";

  const fileInput = document.getElementById('aa-file');
  const preview = document.getElementById('aa-preview');
  const sendBtn = document.getElementById('aa-send');
  const status = document.getElementById('aa-status');
  const resultsDiv = document.getElementById('aa-results');

  function renderResults(json) {
    if (!json) return '<p>No hay resultados</p>';
    if (json.error) return '<p style="color:#b00">Error: ' + json.error + '</p>';
    const map = [
      ['elasticity','Elasticidad (0-100, 100 mejor)'],
      ['wrinkles_general','Arrugas generales (0-100)'],
      ['wrinkles_deep','Arrugas profundas (0-100)'],
      ['expression_lines','Líneas de expresión (0-100)'],
      ['pigmentation','Pigmentación / manchas (0-100)'],
      ['age_biological','Edad biológica estimada']
    ];
    let html = '<ul style="padding-left:18px;margin:0;">';
    map.forEach(([k,label]) => {
      if (k in json && json[k] !== null && json[k] !== undefined) {
        html += `<li><strong>${label}:</strong> ${json[k]}</li>`;
      }
    });
    html += '</ul>';
    if (json.debug) {
      html += '<details style="margin-top:8px;font-size:0.9em;"><summary>Información técnica (debug)</summary><pre style="white-space:pre-wrap">'+ JSON.stringify(json.debug, null, 2) +'</pre></details>';
    }
    return html;
  }

  fileInput.addEventListener('change', () => {
    const f = fileInput.files[0];
    if (!f) { preview.src = ''; sendBtn.disabled = true; return; }
    sendBtn.disabled = false;
    preview.src = URL.createObjectURL(f);
    resultsDiv.style.display = 'none';
    status.textContent = '';
  });

  sendBtn.addEventListener('click', async () => {
    const f = fileInput.files[0];
    if (!f) return;
    if (!SERVER_URL || SERVER_URL.includes('TU-SERVIDOR')) {
      alert('Configura SERVER_URL en el código con la URL de tu servidor antes de usar.');
      return;
    }

    sendBtn.disabled = true;
    status.style.color = '#333';
    status.textContent = 'Subiendo imagen y procesando...';
    resultsDiv.style.display = 'none';
    try {
      const form = new FormData();
      form.append('file', f);
      const resp = await fetch(SERVER_URL, {
        method: 'POST',
        body: form
      });
      if (!resp.ok) {
        const text = await resp.text();
        status.style.color = '#b00';
        status.textContent = 'Error del servidor: ' + resp.status + ' — ' + text;
        sendBtn.disabled = false;
        return;
      }
      const json = await resp.json();
      resultsDiv.innerHTML = renderResults(json);
      resultsDiv.style.display = 'block';
      status.style.color = '#080';
      status.textContent = 'Análisis completo';
    } catch (err) {
      console.error(err);
      status.style.color = '#b00';
      status.textContent = 'Error de red: ' + err.toString();
    } finally {
      sendBtn.disabled = false;
    }
  });
})();
