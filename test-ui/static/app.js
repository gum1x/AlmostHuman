// ---- State ----
let currentChat = null;
let chats = {};
let selectedMessageId = null;

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
  emptyState.classList.add('hidden');
  chatView.classList.remove('hidden');
  chatTitle.textContent = name;
  pipelineSteps.innerHTML = '';
  renderSidebar();
  renderMessages();
  updateRunButton();
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
  const text = msgInput.value.trim();
  if (!text) return;
  msgInput.value = '';
  btnSend.disabled = true;
  btnRun.disabled = true;

  try {
    const res = await fetch(`/api/send/${encodeURIComponent(currentChat)}`, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text, sender_id: 999999 }),
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
  } finally {
    btnSend.disabled = false;
    btnRun.disabled = false;
  }
}

btnSend.addEventListener('click', sendMessage);
msgInput.addEventListener('keydown', (e) => { if (e.key === 'Enter') sendMessage(); });

// ---- Run pipeline ----
async function runPipeline() {
  if (!currentChat) return;
  const targetId = selectedMessageId;
  btnRun.disabled = true;
  const label = targetId ? `Running #${targetId}...` : 'Running...';
  btnRun.textContent = label;
  pipelineSteps.innerHTML = `<div class="step-header"><span class="spinner"></span> ${escHtml(label)}</div>`;

  try {
    // Try WebSocket for real-time streaming
    const wsProto = location.protocol === 'https:' ? 'wss:' : 'ws:';
    const ws = new WebSocket(`${wsProto}//${location.host}/ws/run/${encodeURIComponent(currentChat)}`);
    pipelineSteps.innerHTML = '';

    ws.onopen = () => {
      // Send target_message_id if selected
      if (targetId) {
        ws.send(JSON.stringify({ target_message_id: targetId }));
      }
    };

    ws.onmessage = (event) => {
      const msg = JSON.parse(event.data);
      if (msg.type === 'step') {
        appendStep(msg.step);
      } else if (msg.type === 'complete') {
        renderPipelineResult(msg);
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
      } else if (msg.type === 'error') {
        pipelineSteps.innerHTML += `<div class="step"><div class="step-header"><span class="step-status error"></span><span class="step-name">Error</span></div><div class="step-body open"><span class="error-text">${escHtml(msg.message)}</span></div></div>`;
      }
    };

    ws.onerror = async () => {
      // Fallback to HTTP
      ws.close();
      const body = targetId ? { target_message_id: targetId } : {};
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
    const body = targetId ? { target_message_id: targetId } : {};
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

// ---- Delete chat ----
btnDelete.addEventListener('click', async () => {
  if (!currentChat) return;
  await fetch(`/api/chat/${encodeURIComponent(currentChat)}`, { method: 'DELETE' });
  delete chats[currentChat];
  currentChat = null;
  selectedMessageId = null;
  chatView.classList.add('hidden');
  emptyState.classList.remove('hidden');
  renderSidebar();
});

// ---- Pipeline rendering ----
function renderPipeline(result) {
  pipelineSteps.innerHTML = '';
  for (const step of result.steps || []) {
    appendStep(step);
  }
  renderPipelineResult(result);
}

function appendStep(step) {
  const div = document.createElement('div');
  div.className = 'step';
  const hasError = !!step.error;
  const statusClass = hasError ? 'error' : 'ok';
  const dataStr = step.data ? JSON.stringify(step.data, null, 2) : '';

  // Highlight style_rewriter step specially
  const isRewriter = step.name === 'style_rewriter';
  const nameLabel = isRewriter ? 'style_rewriter (local model)' : step.name;

  div.innerHTML = `
    <div class="step-header">
      <div style="display:flex;align-items:center;">
        <span class="step-status ${statusClass}"></span>
        <span class="step-name">${escHtml(nameLabel)}</span>
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

// ---- Init: load existing chats ----
(async () => {
  try {
    const res = await fetch('/api/chats');
    const data = await res.json();
    for (const [name, info] of Object.entries(data)) {
      const chatRes = await fetch(`/api/chat/${encodeURIComponent(name)}`);
      const chatData = await chatRes.json();
      if (chatData.messages) chats[name] = chatData.messages;
    }
    renderSidebar();
  } catch (err) {
    console.error('Failed to load chats', err);
  }
})();
