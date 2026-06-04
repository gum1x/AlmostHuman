// ---- State ----
let currentChat = null;
let chats = {};
let selectedMessageId = null;
let tunablesMeta = null;           // from /api/tunables
let currentOverrides = {};         // live edits for next re-run
let lastPipelineResult = null;     // for re-render after edits
let lastTargetId = null;
let currentRunSteps = [];            // accumulate during WS streaming so populateTunables gets real step data for values
let pendingReplyTo = null;           // set when right-clicking a bot msg to reply to it specifically as user

// Bulk simulation state (for "drag 500 msgs → live replay from 0 with model replies appearing one-by-one")
let originalScripts = {};  // name -> deep copy of the msgs as-imported (used for Reset + to derive the user-turn sequence)
let simState = null;       // null | {active:boolean, paused:boolean, index:number, turns:array, delay:number, botCount:number, total:number, idMap:{[origId]:liveId} }


// ---- DOM refs ----
const sidebar = document.getElementById('chat-list');
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');
const emptyState = document.getElementById('empty-state');
const chatView = document.getElementById('chat-view');
const chatTitle = document.getElementById('chat-title');
const messagesDiv = document.getElementById('messages');
const pipelineSteps = document.getElementById('pipeline-steps');
const msgInput = document.getElementById('msg-input');
const btnSend = document.getElementById('btn-send');
const btnRun = document.getElementById('btn-run');
const btnDelete = document.getElementById('btn-delete');
const tunablesPanel = document.getElementById('tunables-panel');
const tunablesSliders = document.getElementById('tunables-sliders');
const btnReRun = document.getElementById('btn-rerun-overrides');
const btnReset = document.getElementById('btn-reset-overrides');
const btnClear = document.getElementById('btn-clear-overrides');

// Bulk sim controls (populated after DOM ready; some may be null if HTML not updated)
const btnSimulate = document.getElementById('btn-simulate');
const btnSimPause = document.getElementById('btn-sim-pause');
const btnSimStop = document.getElementById('btn-sim-stop');
const btnResetOrig = document.getElementById('btn-reset-orig');
const simStatusEl = document.getElementById('sim-status');
const simDelaySel = document.getElementById('sim-delay');

// ---- File upload / drag-drop ----
dropZone.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.classList.add('dragover'); });
dropZone.addEventListener('dragleave', () => dropZone.classList.remove('dragover'));
dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('dragover');
  handleFiles(e.dataTransfer.files);
});
fileInput.addEventListener('change', () => { handleFiles(fileInput.files); fileInput.value = ''; });

async function handleFiles(files) {
  for (const file of files) {
    if (!file.name.endsWith('.json')) continue;
    try {
      const text = await file.text();
      let data = JSON.parse(text);
      let messages = Array.isArray(data) ? data : (data.messages || []);
      if (!messages.length) continue;

      const name = file.name.replace('.json', '');
      const res = await fetch('/api/upload', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ name, messages }),
      });
      const result = await res.json();
      if (result.status === 'ok') {
        chats[name] = messages;
        originalScripts[name] = JSON.parse(JSON.stringify(messages));
        renderSidebar();
      }
    } catch (err) {
      console.error('Failed to parse', file.name, err);
    }
  }
}

// ---- Sidebar ----
function renderSidebar() {
  sidebar.innerHTML = '';
  for (const [name, msgs] of Object.entries(chats)) {
    const li = document.createElement('li');
    li.className = name === currentChat ? 'active' : '';
    const lastMsg = msgs[msgs.length - 1];
    const preview = lastMsg ? (lastMsg.text || lastMsg.text_raw || '').slice(0, 60) : '';
    li.innerHTML = `
      <span class="chat-name">${escHtml(name)}</span>
      <span class="chat-meta">${msgs.length} messages</span>
      <span class="chat-preview">${escHtml(preview)}</span>
    `;
    li.addEventListener('click', () => openChat(name));
    sidebar.appendChild(li);
  }
}

// ---- Open chat ----
function openChat(name) {
  currentChat = name;
  selectedMessageId = null;
  pendingReplyTo = null;
  if (simState && simState.active) {
    simState.active = false;
    simState.paused = false;
  }
  setSimulatingUI(false);
  updateSimUI();
  if (msgInput) msgInput.placeholder = "Type a message as the test user...";
  emptyState.classList.add('hidden');
  chatView.classList.remove('hidden');
  chatTitle.textContent = name;
  pipelineSteps.innerHTML = '';
  renderSidebar();
  renderMessages();
  updateRunButton();

  // Make the tunable editor visible immediately when a chat is opened.
  // This is the main place users will look to "edit the numbers".
  // Sliders start at defaults (or last known); user can drag before first Run too.
  showTunablesEditor();
}

// ---- Render messages ----
function renderMessages() {
  const msgs = chats[currentChat] || [];
  messagesDiv.innerHTML = '';
  for (const msg of msgs) {
    const div = document.createElement('div');
    const isBot = msg.is_bot || msg.sender_id === 0;
    const isSelected = selectedMessageId === msg.message_id;
    div.className = `msg ${isBot ? 'bot-msg' : 'user-msg'}${isSelected ? ' selected' : ''}`;
    div.dataset.messageId = msg.message_id;

    let html = '';
    html += `<div class="msg-meta">${isBot ? 'BOT' : 'user_' + (msg.sender_id || '?')} &middot; #${msg.message_id}</div>`;
    if (msg.reply_to_message_id) {
      html += `<div class="reply-indicator">replying to #${msg.reply_to_message_id}</div>`;
    }
    html += `<div>${escHtml(msg.text || msg.text_raw || msg.text_cleaned || '')}</div>`;
    div.innerHTML = html;

    if (isBot) {
      div.title = "Right-click to reply as another user";
      div.addEventListener('contextmenu', (e) => {
        e.preventDefault();
        showContextMenu(e, msg);
      });
    }

    // Click to select as target
    div.addEventListener('click', () => {
      if (selectedMessageId === msg.message_id) {
        // Deselect
        selectedMessageId = null;
      } else {
        selectedMessageId = msg.message_id;
      }
      renderMessages();
      updateRunButton();
    });

    // Double-click to immediately run pipeline for this message
    div.addEventListener('dblclick', (e) => {
      e.preventDefault();
      selectedMessageId = msg.message_id;
      renderMessages();
      updateRunButton();
      runPipeline();
    });

    messagesDiv.appendChild(div);
  }
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

function updateRunButton() {
  if (selectedMessageId) {
    btnRun.textContent = `Run #${selectedMessageId}`;
    btnRun.title = `Run pipeline targeting message #${selectedMessageId}`;
  } else {
    btnRun.textContent = 'Run Pipeline';
    btnRun.title = 'Run through full pipeline (targets latest message)';
  }
}

// ---- Send message ----
async function sendMessage() {
  if (!currentChat) return;
  if (simState && simState.active) return;
  const text = msgInput.value.trim();
  if (!text) return;
  msgInput.value = '';
  btnSend.disabled = true;
  btnRun.disabled = true;

  try {
    const body = { text, sender_id: 999999 };
    if (pendingReplyTo) {
      body.reply_to_message_id = pendingReplyTo;
      pendingReplyTo = null;
      if (msgInput) msgInput.placeholder = "Type a message as the test user...";
    }
    const res = await fetch(`/api/send/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await res.json();

    if (result.user_message) chats[currentChat].push(result.user_message);
    if (result.bot_message) chats[currentChat].push(result.bot_message);

    selectedMessageId = null;
    renderMessages();
    renderSidebar();
    updateRunButton();

    if (result.pipeline) renderPipeline(result.pipeline);
  } catch (err) {
    addSystemMsg('Error: ' + err.message);
    // restore pending if send failed?
    if (body.reply_to_message_id) pendingReplyTo = body.reply_to_message_id;
  } finally {
    btnSend.disabled = false;
    btnRun.disabled = false;
  }
}

btnSend.addEventListener('click', sendMessage);
msgInput.addEventListener('keydown', (e) => {
  if (e.key === 'Enter') {
    sendMessage();
  } else if (e.key === 'Escape' && pendingReplyTo) {
    e.preventDefault();
    clearPendingReply();
  }
});

// ---- Run pipeline ----
async function runPipeline() {
  if (!currentChat) return;
  if (simState && simState.active) return;
  const targetId = selectedMessageId;
  lastTargetId = targetId;
  btnRun.disabled = true;
  const label = targetId ? `Running #${targetId}...` : 'Running...';
  btnRun.textContent = label;
  pipelineSteps.innerHTML = `<div class="step-header"><span class="spinner"></span> ${escHtml(label)}</div>`;
  currentRunSteps = [];
  // do not clear currentOverrides here — user may have set dials before hitting Run

  try {
    // Try WebSocket for real-time streaming (note: overrides not carried on ws path; falls back for edits)
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${location.host}/ws/run/${encodeURIComponent(currentChat)}`);
    pipelineSteps.innerHTML = '';

    ws.onopen = () => {
      const initPayload = {};
      if (targetId) initPayload.target_message_id = targetId;
      if (currentOverrides && Object.keys(currentOverrides).length > 0) {
        initPayload.overrides = currentOverrides;
      }
      ws.send(JSON.stringify(initPayload));
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'step') {
        currentRunSteps.push(msg.step);
        appendStep(msg.step);
      } else if (msg.type === 'complete') {
        renderPipelineResult(msg);
        lastPipelineResult = {
          steps: currentRunSteps,
          response_text: msg.response_text,
          decision: msg.decision,
          total_duration_ms: msg.total_duration_ms
        };
        if (msg.response_text) {
          const msgs = chats[currentChat];
          const lastId = Math.max(...msgs.map(m => m.message_id || 0), 0);
          msgs.push({
            message_id: lastId + 1,
            chat_id: msgs[0]?.chat_id || 0,
            sender_id: 0,
            text: msg.response_text,
            is_bot: true,
            reply_to_message_id: msg.decision?.reply_to_message_id || targetId || null,
          });
          renderMessages();
        }
        populateTunables(lastPipelineResult);
        // Scroll the right panel so the tunable editor is visible after seeing the output
        scrollToTunables();
      } else if (msg.type === 'error') {
        pipelineSteps.innerHTML += `<div class="step"><div class="step-header"><span class="step-status error"></span><span class="step-name">Error</span></div><div class="step-body open"><span class="error-text">${escHtml(msg.message)}</span></div></div>`;
      }
    };

    ws.onerror = async () => {
      ws.close();
      const body = targetId ? { target_message_id: targetId, overrides: currentOverrides } : { overrides: currentOverrides };
      const res = await fetch(`/api/run/${encodeURIComponent(currentChat)}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
      });
      const result = await res.json();
      renderPipeline(result);
    };

    ws.onclose = () => {
      btnRun.disabled = false;
      updateRunButton();
    };
  } catch (err) {
    const body = targetId ? { target_message_id: targetId, overrides: currentOverrides } : { overrides: currentOverrides };
    const res = await fetch(`/api/run/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await res.json();
    renderPipeline(result);
    btnRun.disabled = false;
    updateRunButton();
  }
}

btnRun.addEventListener('click', runPipeline);

// ---- Tunables (editable dials) wiring ----

async function loadTunablesMeta() {
  try {
    const res = await fetch('/api/tunables');
    tunablesMeta = await res.json();
  } catch (e) {
    console.warn('Could not load tunables meta, using fallback', e);
    tunablesMeta = { readonly: ['is_reply_to_bot'], tunables: {} };  // fallback; real data comes from /api/tunables (now only threshold + willingness)
  }
}

function getCurrentSignalValuesFromResult(result) {
  // Try to pull the "as run" values so sliders start at the numbers the last run actually used.
  const vals = {};
  for (const step of (result && result.steps) || []) {
    if (step.name === 'precomputed_signals' && step.data && step.data.block) {
      // parse "foo=1.2 | bar=3 | ..."
      const parts = String(step.data.block).split('|').map(s => s.trim());
      for (const p of parts) {
        const eq = p.indexOf('=');
        if (eq > 0) {
          const k = p.slice(0, eq).trim();
          const v = p.slice(eq + 1).trim();
          const num = parseFloat(v);
          vals[k] = isNaN(num) ? v : num;
        }
      }
    }
    if (step.name === 'brief' && step.data && typeof step.data.tension_level === 'number') {
      vals['tension'] = step.data.tension_level;
    }
    if (step.name === 'engagement_gate' && step.data && step.data.factors) {
      // some factors may be editable indirectly via weights
    }
  }
  // also carry any overrides that were applied last time
  if (result && result.steps) {
    for (const step of result.steps) {
      if (step.data && step.data.overrides) {
        Object.assign(vals, step.data.overrides);
      }
      if (step.data && step.data.overrides_applied) {
        Object.assign(vals, step.data.overrides_applied);
      }
    }
  }
  return vals;
}

function populateTunables(result) {
  if (!tunablesPanel || !tunablesSliders || !tunablesMeta) return;
  tunablesSliders.innerHTML = '';
  const currentVals = getCurrentSignalValuesFromResult(result || lastPipelineResult);
  const tunables = (tunablesMeta && tunablesMeta.tunables) || {};
  const readonly = new Set((tunablesMeta && tunablesMeta.readonly) || []);

  let any = false;
  for (const [key, meta] of Object.entries(tunables)) {
    if (readonly.has(key)) continue;
    any = true;
    const row = document.createElement('div');
    row.className = 'tunable-row';

    const valNow = (currentVals[key] !== undefined) ? currentVals[key] : (meta.default !== undefined ? meta.default : 0);
    // keep last edited if user is mid-edit
    const liveVal = (currentOverrides[key] !== undefined) ? currentOverrides[key] : valNow;

    const label = document.createElement('div');
    label.className = 'tunable-label';
    label.innerHTML = `<span class="tname">${escHtml(meta.label || key)}</span> <span class="tval" id="val-${key}">${Number(liveVal).toFixed( (meta.step||0.01)<0.1 ? 2 : 0 )}</span>`;

    const slider = document.createElement('input');
    slider.type = 'range';
    slider.min = meta.min;
    slider.max = meta.max;
    slider.step = meta.step || 0.01;
    slider.value = liveVal;
    slider.dataset.key = key;
    slider.className = 'tunable-slider';

    const desc = document.createElement('div');
    desc.className = 'tunable-desc';
    desc.textContent = meta.desc || '';

    slider.addEventListener('input', () => {
      const v = parseFloat(slider.value);
      currentOverrides[key] = v;
      const vEl = document.getElementById('val-' + key);
      if (vEl) vEl.textContent = v.toFixed( (meta.step||0.01)<0.1 ? 2 : 0 );
      // mark edited
      row.classList.add('edited');
    });

    row.appendChild(label);
    row.appendChild(slider);
    row.appendChild(desc);
    tunablesSliders.appendChild(row);
  }

  if (any) {
    tunablesPanel.classList.remove('hidden');
  } else {
    tunablesPanel.classList.add('hidden');
  }
}

function showTunablesEditor() {
  // Robust way to ensure the editor is visible as soon as a chat is selected.
  // Users will see the drag-to-edit controls right away (with friendly names),
  // can adjust before hitting Run, and re-run will use currentOverrides.
  if (!tunablesPanel || !tunablesSliders) return;

  if (tunablesMeta && tunablesMeta.tunables && Object.keys(tunablesMeta.tunables).length > 0) {
    populateTunables({ steps: currentRunSteps.length ? currentRunSteps : [] });
  } else {
    // Meta not ready yet (race on fresh load) – load then show defaults
    loadTunablesMeta().then(() => {
      populateTunables({ steps: [] });
    }).catch(() => {
      // fallback will have been set inside loadTunablesMeta
      populateTunables({ steps: [] });
    });
  }
}

function scrollToTunables() {
  // After a run completes (seeing the decision/output), bring the editor into view
  // so it's obvious where to tweak the numbers and hit "Re-run with edits".
  setTimeout(() => {
    if (tunablesPanel && !tunablesPanel.classList.contains('hidden')) {
      tunablesPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    } else if (tunablesPanel) {
      tunablesPanel.classList.remove('hidden');
      tunablesPanel.scrollIntoView({ behavior: 'smooth', block: 'nearest' });
    }
  }, 80);
}

function resetOverrides() {
  currentOverrides = {};
  populateTunables(lastPipelineResult || { steps: [] });
}

function clearOverrides() {
  currentOverrides = {};
  // Keep the editor visible; just reset all dials to their meta defaults
  populateTunables(lastPipelineResult || { steps: [] });
}

async function reRunWithOverrides() {
  if (!currentChat) return;
  const targetId = lastTargetId || selectedMessageId;
  btnReRun.disabled = true;
  const orig = btnReRun.textContent;
  btnReRun.textContent = 'Re-running...';

  try {
    const body = { target_message_id: targetId || undefined, overrides: currentOverrides };
    const res = await fetch(`/api/run/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await res.json();
    if (result && result.steps) {
      lastPipelineResult = result;
      lastTargetId = targetId;
      // re-render steps + result
      pipelineSteps.innerHTML = '';
      for (const step of result.steps || []) appendStep(step);
      renderPipelineResult(result);
      // refresh sliders with new actual values + our edits still active
      populateTunables(result);
      // tiny badge on the decision/gate steps if overrides were used
      highlightEditedSteps(result);
      scrollToTunables();
    }
  } catch (err) {
    addSystemMsg('Re-run error: ' + err.message);
  } finally {
    btnReRun.disabled = false;
    btnReRun.textContent = orig;
  }
}

function highlightEditedSteps(result) {
  // Add "(edited)" badges to steps that show overrides_applied
  const steps = pipelineSteps.querySelectorAll('.step');
  steps.forEach((div, i) => {
    const step = (result.steps || [])[i];
    if (step && step.data && (step.data.overrides_applied || step.data.overrides)) {
      const nameEl = div.querySelector('.step-name');
      if (nameEl && !nameEl.textContent.includes('(edited)')) {
        const b = document.createElement('span');
        b.className = 'edited-badge';
        b.textContent = ' (edited)';
        nameEl.appendChild(b);
      }
    }
  });
}

if (btnReRun) btnReRun.addEventListener('click', reRunWithOverrides);
if (btnReset) btnReset.addEventListener('click', () => { resetOverrides(); });
if (btnClear) btnClear.addEventListener('click', () => { clearOverrides(); });

// ---- Delete chat ----
btnDelete.addEventListener('click', async () => {
  if (!currentChat) return;
  await fetch(`/api/chat/${encodeURIComponent(currentChat)}`, { method: 'DELETE' });
  delete chats[currentChat];
  delete originalScripts[currentChat];
  if (simState && currentChat) {
    // simState is shared; just stop it
    simState.active = false;
    simState.paused = false;
  }
  currentChat = null;
  pendingReplyTo = null;
  if (msgInput) msgInput.placeholder = "Type a message as the test user...";
  selectedMessageId = null;
  setSimulatingUI(false);
  updateSimUI();
  chatView.classList.add('hidden');
  emptyState.classList.remove('hidden');
  renderSidebar();
});

// ---- Pipeline rendering ----
function renderPipeline(result) {
  lastPipelineResult = result;
  lastTargetId = selectedMessageId;
  currentRunSteps = result.steps || [];
  pipelineSteps.innerHTML = '';
  for (const step of result.steps || []) {
    appendStep(step);
  }
  renderPipelineResult(result);
  populateTunables(result);
  highlightEditedSteps(result);
  scrollToTunables();
}

function appendStep(step) {
  const div = document.createElement('div');
  div.className = 'step';
  const hasError = !!step.error;
  const statusClass = hasError ? 'error' : 'ok';
  const dataStr = step.data ? JSON.stringify(step.data, null, 2) : '';

  const isRewriter = step.name === 'style_rewriter';
  let nameLabel = isRewriter ? 'style_rewriter (local model)' : step.name;
  const hasEdits = !!(step.data && (step.data.overrides_applied || step.data.overrides));
  if (hasEdits) nameLabel += ' <span class="edited-badge">(edited)</span>';

  div.innerHTML = `
    <div class="step-header">
      <div style="display:flex;align-items:center;">
        <span class="step-status ${statusClass}"></span>
        <span class="step-name">${nameLabel}</span>
      </div>
      <span class="step-time">${step.duration_ms}ms</span>
    </div>
    <div class="step-body">
      ${hasError ? `<span class="error-text">${escHtml(step.error)}</span>` : ''}
      ${dataStr ? `<pre>${escHtml(dataStr)}</pre>` : ''}
    </div>
  `;

  div.querySelector('.step-header').addEventListener('click', () => {
    div.querySelector('.step-body').classList.toggle('open');
  });

  pipelineSteps.appendChild(div);
}

function renderPipelineResult(result) {
  const hasResponse = !!result.response_text;
  const styleApplied = result.decision?.style_rewriter_applied;
  const div = document.createElement('div');
  div.className = `pipeline-result ${hasResponse ? '' : 'no-response'}`;
  div.innerHTML = `
    <h4>${hasResponse ? 'Bot Response' : 'No Response (silent)'}${styleApplied ? ' <span style="color:var(--purple);font-size:11px;">[local model phrased]</span>' : ''}</h4>
    ${hasResponse ? `<div class="response-text">${escHtml(result.response_text)}</div>` : ''}
    ${result.decision ? `<pre style="font-size:11px;color:var(--text-muted);margin-top:8px;">${escHtml(JSON.stringify({
      should_respond: result.decision.should_respond,
      confidence: result.decision.confidence,
      reasoning: result.decision.reasoning,
      plan: result.decision.plan,
      tone: result.decision.tone_calibration,
    }, null, 2))}</pre>` : ''}
    <div class="total-time">Total: ${result.total_duration_ms || 0}ms</div>
  `;
  pipelineSteps.appendChild(div);
}

function addSystemMsg(text) {
  const div = document.createElement('div');
  div.className = 'msg system-msg';
  div.textContent = text;
  messagesDiv.appendChild(div);
  messagesDiv.scrollTop = messagesDiv.scrollHeight;
}

// ---- Util ----
function escHtml(str) {
  if (!str) return '';
  return String(str).replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ---- Right-click context menu for bot messages ----
let contextMenuEl = null;

function initContextMenu() {
  contextMenuEl = document.getElementById('context-menu');
  if (!contextMenuEl) {
    // Fallback create if not in HTML
    contextMenuEl = document.createElement('div');
    contextMenuEl.id = 'context-menu';
    contextMenuEl.className = 'context-menu';
    contextMenuEl.innerHTML = '<div class="cm-item" data-action="reply-user">Reply as another user to this message</div>';
    document.body.appendChild(contextMenuEl);
  }
  contextMenuEl.addEventListener('click', (e) => {
    const action = e.target.dataset.action;
    const msgId = contextMenuEl.dataset.msgId;
    contextMenuEl.style.display = 'none';
    if (action === 'reply-user' && msgId && currentChat) {
      startReplyToBotMessage(parseInt(msgId, 10));
    }
  });
  // Hide when clicking anywhere else
  document.addEventListener('click', (e) => {
    if (contextMenuEl && !contextMenuEl.contains(e.target)) {
      contextMenuEl.style.display = 'none';
    }
  });
  // Hide on escape
  document.addEventListener('keydown', (e) => {
    if (e.key === 'Escape' && contextMenuEl) contextMenuEl.style.display = 'none';
  });
}

function showContextMenu(e, msg) {
  if (!contextMenuEl) initContextMenu();
  contextMenuEl.style.display = 'block';
  contextMenuEl.style.left = e.pageX + 'px';
  contextMenuEl.style.top = e.pageY + 'px';
  contextMenuEl.dataset.msgId = msg.message_id;
  // hide if menu goes off screen? simple clamp
  setTimeout(() => {
    const rect = contextMenuEl.getBoundingClientRect();
    if (rect.right > window.innerWidth) {
      contextMenuEl.style.left = (e.pageX - rect.width) + 'px';
    }
    if (rect.bottom > window.innerHeight) {
      contextMenuEl.style.top = (e.pageY - rect.height) + 'px';
    }
  }, 0);
}

function startReplyToBotMessage(messageId) {
  pendingReplyTo = messageId;
  if (msgInput) {
    msgInput.placeholder = `Replying to #${messageId} as another user... (type & send; Esc to cancel)`;
    msgInput.title = `Replying specifically to message #${messageId}`;
    msgInput.focus();
  }
  // Optional: also select it as target for visibility
  selectedMessageId = messageId;
  renderMessages();
  updateRunButton();
}

// Clear pending reply when appropriate
function clearPendingReply() {
  pendingReplyTo = null;
  if (msgInput) {
    msgInput.placeholder = "Type a message as the test user...";
    msgInput.title = '';
  }
}

// ---- Bulk simulation helpers (live replay of imported transcripts with model replies appearing live) ----

function isBotMsg(m) {
  return !!(m && (m.is_bot || m.sender_id === 0));
}

function sleep(ms) {
  return new Promise(resolve => setTimeout(resolve, ms));
}

function updateSimUI() {
  if (!simStatusEl) return;
  const sim = simState;
  if (!sim || !sim.active) {
    simStatusEl.classList.add('hidden');
    if (btnSimPause) { btnSimPause.classList.add('hidden'); btnSimPause.textContent = '⏸ Pause'; }
    if (btnSimStop) btnSimStop.classList.add('hidden');
    if (btnSimulate) btnSimulate.classList.remove('hidden');
    if (btnResetOrig) btnResetOrig.classList.remove('hidden');
    return;
  }
  simStatusEl.classList.remove('hidden');
  simStatusEl.textContent = `Simulating ${sim.index}/${sim.total} • bots: ${sim.botCount || 0}`;
  if (btnSimulate) btnSimulate.classList.add('hidden');
  if (btnSimStop) btnSimStop.classList.remove('hidden');
  if (btnSimPause) {
    btnSimPause.classList.remove('hidden');
    btnSimPause.textContent = sim.paused ? '▶ Resume' : '⏸ Pause';
  }
  if (btnResetOrig) btnResetOrig.classList.add('hidden');
}

function setSimulatingUI(on) {
  if (btnRun) btnRun.disabled = on;
  if (btnSend) btnSend.disabled = on;
  if (msgInput) msgInput.disabled = on;
  if (btnDelete) btnDelete.disabled = on;
  if (btnReRun) btnReRun.disabled = on;
  if (btnReset) btnReset.disabled = on;
  if (btnClear) btnClear.disabled = on;
  if (on && btnSimulate) btnSimulate.classList.add('btn-simulating');
  else if (btnSimulate) btnSimulate.classList.remove('btn-simulating');
}

async function animateLastBotMessage(fullText) {
  if (!fullText || !messagesDiv) return;
  const last = messagesDiv.lastElementChild;
  if (!last || !last.classList.contains('bot-msg')) return;
  const content = last.querySelector(':scope > div:last-child');
  if (!content) return;
  const text = String(fullText);
  content.textContent = '';
  const step = 18; // ms/char for streaming feel
  for (let i = 0; i < text.length; i++) {
    content.textContent += text[i];
    messagesDiv.scrollTop = messagesDiv.scrollHeight;
    await sleep(step);
  }
}

function getSimDelay() {
  if (simDelaySel) {
    const v = parseInt(simDelaySel.value, 10);
    if (!isNaN(v)) return v;
  }
  return 250;
}

async function playOneTurn(turn) {
  if (!currentChat || !simState) return;
  const origId = turn && turn.message_id;
  const text = (turn && (turn.text || turn.text_raw || turn.text_cleaned)) || '';
  if (!text) return;
  const origReply = turn ? turn.reply_to_message_id : null;
  const liveReply = (origReply != null && simState.idMap && simState.idMap[origReply] != null)
    ? simState.idMap[origReply]
    : null;
  const sender = turn ? (turn.sender_id || 999999) : 999999;
  try {
    const body = { text: text, sender_id: sender };
    if (liveReply != null) body.reply_to_message_id = liveReply;
    const res = await fetch(`/api/send/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(body),
    });
    const result = await res.json();
    let botAdded = false;
    if (result.user_message) {
      chats[currentChat].push(result.user_message);
      if (origId != null && result.user_message.message_id != null) {
        simState.idMap = simState.idMap || {};
        simState.idMap[origId] = result.user_message.message_id;
      }
    }
    if (result.bot_message) {
      chats[currentChat].push(result.bot_message);
      botAdded = true;
      simState.botCount = (simState.botCount || 0) + 1;
    }
    renderMessages();
    renderSidebar();
    // "live" model response via typing animation (one-by-one streaming appearance)
    if (botAdded && result.bot_message && simState.delay > 0) {
      await animateLastBotMessage(result.bot_message.text || result.bot_message.text_raw || '');
    }
  } catch (err) {
    addSystemMsg('Sim turn error: ' + (err.message || err));
  }
}

async function playSimulationLoop() {
  if (!simState) return;
  while (simState.active && !simState.paused && simState.index < simState.total) {
    const turn = simState.turns[simState.index];
    simState.index = simState.index + 1;
    await playOneTurn(turn);
    updateSimUI();
    if (simState.delay > 0) {
      await sleep(simState.delay);
    }
  }
  if (simState.index >= simState.total) {
    simState.active = false;
    simState.paused = false;
    addSystemMsg('Bulk simulation complete.');
  }
  setSimulatingUI(false);
  updateSimUI();
  if (btnRun) btnRun.disabled = false;
  if (btnSend) btnSend.disabled = false;
  if (msgInput) msgInput.disabled = false;
  if (btnDelete) btnDelete.disabled = false;
}

async function startBulkSimulation() {
  if (!currentChat) return;
  const base = originalScripts[currentChat] || chats[currentChat] || [];
  const turns = base.filter(m => !isBotMsg(m)).map(m => JSON.parse(JSON.stringify(m)));
  if (!turns.length) {
    addSystemMsg('No user turns found to simulate.');
    return;
  }
  if (turns.length > 50) {
    const ok = confirm(`Bulk simulate ${turns.length} user messages? This will run the pipeline ${turns.length} times (may be slow + use API quota if real key is set).`);
    if (!ok) return;
  }
  // Start from 0 on both client and server (also clears bot_mems for clean carry)
  chats[currentChat] = [];
  try {
    await fetch(`/api/reset/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: [] }),
    });
  } catch (e) { /* non fatal */ }
  renderMessages();
  renderSidebar();
  if (pipelineSteps) {
    pipelineSteps.innerHTML = '<div class="step-header">Bulk sim active — per-turn pipeline steps hidden for performance/clarity on long transcripts (use manual Send/Run after for details on a specific turn).</div>';
  }
  simState = {
    active: true,
    paused: false,
    index: 0,
    turns: turns,
    delay: getSimDelay(),
    botCount: 0,
    total: turns.length,
    idMap: {},
  };
  setSimulatingUI(true);
  updateSimUI();
  try {
    await playSimulationLoop();
  } catch (e) {
    addSystemMsg('Sim loop error: ' + (e.message || e));
    if (simState) simState.active = false;
    setSimulatingUI(false);
    updateSimUI();
  }
}

async function resetToOriginal() {
  if (!currentChat) return;
  const orig = originalScripts[currentChat];
  if (!orig) {
    addSystemMsg('No original snapshot to reset to for this chat.');
    return;
  }
  const copy = JSON.parse(JSON.stringify(orig));
  chats[currentChat] = copy;
  try {
    await fetch(`/api/reset/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ messages: copy }),
    });
  } catch (e) { /* local still updated */ }
  if (simState) {
    simState.active = false;
    simState.paused = false;
  }
  selectedMessageId = null;
  pendingReplyTo = null;
  if (msgInput) {
    msgInput.placeholder = "Type a message as the test user...";
    msgInput.disabled = false;
  }
  setSimulatingUI(false);
  renderMessages();
  renderSidebar();
  updateRunButton();
  updateSimUI();
  addSystemMsg('Reset to original import.');
}

function stopSimulation() {
  if (!simState) return;
  simState.active = false;
  simState.paused = false;
  setSimulatingUI(false);
  updateSimUI();
  if (btnRun) btnRun.disabled = false;
  if (btnSend) btnSend.disabled = false;
  if (msgInput) msgInput.disabled = false;
  addSystemMsg('Simulation stopped.');
}

function initSimControls() {
  if (btnSimulate) {
    btnSimulate.addEventListener('click', () => {
      if (simState && simState.active) return;
      startBulkSimulation();
    });
  }
  if (btnSimPause) {
    btnSimPause.addEventListener('click', () => {
      if (!simState || !simState.active) return;
      simState.paused = !simState.paused;
      if (!simState.paused) {
        playSimulationLoop();
      }
      updateSimUI();
    });
  }
  if (btnSimStop) {
    btnSimStop.addEventListener('click', () => stopSimulation());
  }
  if (btnResetOrig) {
    btnResetOrig.addEventListener('click', resetToOriginal);
  }
  if (simDelaySel) {
    simDelaySel.addEventListener('change', () => {
      if (simState && simState.active) {
        simState.delay = getSimDelay();
      }
    });
  }
}

// ---- Init: load existing chats + tunables meta ----
(async () => {
  try {
    initContextMenu();
    initSimControls();
    await loadTunablesMeta();
    const res = await fetch('/api/chats');
    const data = await res.json();
    for (const [name, info] of Object.entries(data)) {
      const chatRes = await fetch(`/api/chat/${encodeURIComponent(name)}`);
      const chatData = await chatRes.json();
      if (chatData.messages) {
        chats[name] = chatData.messages;
        originalScripts[name] = JSON.parse(JSON.stringify(chatData.messages));
      }
    }
    renderSidebar();
  } catch (err) {
    console.error('Failed to load chats', err);
  }
})();
