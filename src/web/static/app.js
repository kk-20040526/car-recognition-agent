const state = { sessionId: null, image: null, busy: false, lastCarId: null, favorites: [], selected: new Set() };

const el = (id) => document.getElementById(id);

function addBubble(role, text, options = {}) {
  const wrapper = document.createElement("div");
  wrapper.className = `msg ${role}`;
  const bubble = document.createElement("div");
  bubble.className = "bubble" + (options.warn ? " warn" : "") + (options.error ? " error" : "");
  bubble.textContent = text;
  if (options.imageUrl) {
    const img = document.createElement("img");
    img.src = options.imageUrl;
    bubble.appendChild(img);
  }
  if (options.meta) {
    const meta = document.createElement("div");
    meta.className = "meta";
    meta.textContent = options.meta;
    bubble.appendChild(meta);
  }
  wrapper.appendChild(bubble);
  el("chat").appendChild(wrapper);
  el("chat").scrollTop = el("chat").scrollHeight;
  return bubble;
}

function renderToolCalls(calls) {
  const box = el("toolLog");
  if (!calls.length) return;
  box.querySelectorAll(".empty").forEach((node) => node.remove());
  calls.forEach((call, index) => {
    const div = document.createElement("div");
    div.className = "tool";
    const ok = call.result && call.result.ok !== false;
    div.innerHTML = `<strong>${index + 1}. ${call.name}</strong>
      <span class="badge ${ok ? "" : "warn"}">${ok ? "成功" : "失败"}</span>
      <span class="badge">${call.elapsed_ms} ms</span>
      <pre>参数: ${escapeHtml(call.arguments)}\n结果: ${escapeHtml(JSON.stringify(call.result, null, 2))}</pre>`;
    box.prepend(div);
  });
}

function escapeHtml(text) {
  return String(text).replace(/[&<>"']/g, (c) =>
    ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}

function displayName(name) {
  return String(name || "").replace(/_/g, " ");
}

async function refreshFavorites() {
  const data = await fetch("/api/favorites").then((r) => r.json());
  state.favorites = data.favorites || [];
  el("favCount").textContent = data.count;
  const ids = new Set(state.favorites.map((item) => item.car_id));
  state.selected = new Set([...state.selected].filter((id) => ids.has(id)));
  rebuildFavoriteFilters();
  renderFavorites();
}

function rebuildFavoriteFilters() {
  const brands = [...new Set(state.favorites.map((item) => item.brand).filter(Boolean))].sort();
  const levels = [...new Set(state.favorites.map((item) => item.level).filter(Boolean))].sort();
  const brandSel = el("favBrand");
  const levelSel = el("favLevel");
  const keepBrand = brandSel.value;
  const keepLevel = levelSel.value;
  brandSel.innerHTML = '<option value="">全部品牌</option>' +
    brands.map((b) => `<option value="${escapeHtml(b)}">${escapeHtml(b)}</option>`).join("");
  levelSel.innerHTML = '<option value="">全部级别</option>' +
    levels.map((l) => `<option value="${escapeHtml(l)}">${escapeHtml(l)}</option>`).join("");
  brandSel.value = brands.includes(keepBrand) ? keepBrand : "";
  levelSel.value = levels.includes(keepLevel) ? keepLevel : "";
}

function filteredFavorites() {
  const kw = el("favFilter").value.trim().toLowerCase();
  const brand = el("favBrand").value;
  const level = el("favLevel").value;
  return state.favorites.filter((item) => {
    if (brand && item.brand !== brand) return false;
    if (level && item.level !== level) return false;
    if (kw) {
      const hay = [item.car_name, item.car_id, item.note, item.brand, item.level, item.body_type]
        .join(" ").toLowerCase();
      if (!hay.includes(kw)) return false;
    }
    return true;
  });
}

function renderFavorites() {
  const box = el("favorites");
  box.innerHTML = "";
  const items = filteredFavorites();
  if (!items.length) {
    box.innerHTML = `<p class="empty">${state.favorites.length ? "没有符合条件的车" : "收藏夹是空的"}</p>`;
    updateSelectionBar();
    return;
  }
  items.forEach((item) => {
    const div = document.createElement("div");
    div.className = "item fav-item";
    const checked = state.selected.has(item.car_id);
    const thumb = item.image_url
      ? `<img class="fav-thumb" src="${escapeHtml(item.image_url)}" alt=""
           onerror="this.style.display='none'">`
      : '<div class="fav-thumb placeholder"></div>';
    const level = item.confidence_level === "low"
      ? '<span class="badge warn">低置信度</span>'
      : (item.confidence_level === "medium" ? '<span class="badge warn">待确认</span>' : "");
    const conf = (typeof item.confidence === "number" && item.confidence > 0)
      ? ` · 置信度 ${(item.confidence * 100).toFixed(1)}%` : "";
    const meta = [item.brand, item.level, item.body_type].filter(Boolean).join(" · ");
    div.innerHTML = `<label class="fav-check" title="选中后可对比">
        <input type="checkbox" ${checked ? "checked" : ""} data-id="${escapeHtml(item.car_id)}" />
      </label>${thumb}<div class="fav-body">
      <div class="title">${escapeHtml(displayName(item.car_name))}<span class="badge">${escapeHtml(item.car_id)}</span>${level}</div>
      <div class="desc">${escapeHtml(meta || item.note || "（无备注）")}${conf}</div>
      <div class="fav-actions">
        <button class="mini" data-act="detail">资料</button>
        <button class="mini" data-act="compare">对比</button>
        <button class="mini" data-act="remove">删除</button>
      </div></div>`;
    div.querySelector('input[type="checkbox"]').addEventListener("change", (event) => {
      event.stopPropagation();
      toggleSelection(item.car_id, event.target.checked);
    });
    div.addEventListener("click", (event) => {
      if (event.target.closest(".fav-check") || event.target.closest(".fav-actions")) return;
      openCarDetail(item.car_id);
    });
    div.querySelector('[data-act="detail"]').addEventListener("click", (event) => {
      event.stopPropagation();
      openCarDetail(item.car_id);
    });
    div.querySelector('[data-act="compare"]').addEventListener("click", (event) => {
      event.stopPropagation();
      compareFavorite(item.car_id);
    });
    div.querySelector('[data-act="remove"]').addEventListener("click", (event) => {
      event.stopPropagation();
      removeFavorite(item.car_id);
    });
    box.appendChild(div);
  });
  updateSelectionBar();
}

function toggleSelection(carId, checked) {
  if (checked) {
    if (state.selected.size >= 2) {
      openModal("最多选两辆", '<p class="modal-error">一次最多对比两辆车，先取消一辆再选。</p>');
      renderFavorites();
      return;
    }
    state.selected.add(carId);
  } else {
    state.selected.delete(carId);
  }
  updateSelectionBar();
}

function updateSelectionBar() {
  const count = state.selected.size;
  el("selectionCount").textContent = `已选 ${count} 辆`;
  el("compareSelected").disabled = count !== 2;
}

function clearSelection() {
  state.selected.clear();
  renderFavorites();
}

// ---------------------------------------------------------------- 资料卡 / 对比
function ensureModal() {
  let mask = document.getElementById("modalMask");
  if (mask) return mask;
  mask = document.createElement("div");
  mask.id = "modalMask";
  mask.className = "modal-mask";
  mask.innerHTML = `<div class="modal" role="dialog" aria-modal="true">
    <div class="modal-head"><span id="modalTitle"></span>
      <button id="modalClose" class="mini" type="button">关闭</button></div>
    <div id="modalBody" class="modal-body"></div></div>`;
  document.body.appendChild(mask);
  mask.addEventListener("click", (event) => { if (event.target === mask) closeModal(); });
  mask.querySelector("#modalClose").addEventListener("click", closeModal);
  return mask;
}

function openModal(title, bodyHtml) {
  const mask = ensureModal();
  mask.querySelector("#modalTitle").textContent = title;
  mask.querySelector("#modalBody").innerHTML = bodyHtml;
  mask.classList.add("show");
}

function closeModal() {
  const mask = document.getElementById("modalMask");
  if (mask) mask.classList.remove("show");
}

async function openCarDetail(carId) {
  const fav = state.favorites.find((item) => item.car_id === carId);
  try {
    const response = await fetch(`/api/cars/${encodeURIComponent(carId)}`);
    const data = await response.json();
    if (!response.ok) { openModal("车型资料", `<p class="modal-error">${escapeHtml(data.detail || "车型不存在")}</p>`); return; }
    const similar = (data.similar_cars || []).map((c) => `${c.car_id} ${c.car_name}`).join("、") || "无";
    const image = fav && fav.image_url
      ? `<img class="detail-image" src="${escapeHtml(fav.image_url)}" alt="${escapeHtml(displayName(data.car_name))}">` : "";
    const conf = fav && typeof fav.confidence === "number" && fav.confidence > 0
      ? `<dt>置信度</dt><dd>${(fav.confidence * 100).toFixed(1)}%${fav.confidence_level === "low" ? ' <span class="badge warn">低置信度</span>' : (fav.confidence_level === "medium" ? ' <span class="badge warn">待确认</span>' : "")}</dd>` : "";
    const note = fav && fav.note ? `<dt>备注</dt><dd>${escapeHtml(fav.note)}</dd>` : "";
    openModal(`${displayName(data.car_name)} 资料卡`, `
      ${image}
      <dl class="detail-list">
        <dt>车型</dt><dd>${escapeHtml(displayName(data.car_name))} <span class="badge">${escapeHtml(data.car_id)}</span></dd>
        <dt>品牌</dt><dd>${escapeHtml(data.brand || "-")}</dd>
        <dt>车系</dt><dd>${escapeHtml(data.model || "-")}</dd>
        <dt>级别</dt><dd>${escapeHtml(data.level || "-")}</dd>
        <dt>车身形式</dt><dd>${escapeHtml(data.body_type || "-")}</dd>
        <dt>识别支持</dt><dd>${data.supported ? "支持图片识别" : "仅资料查询"}</dd>
        ${conf}${note}
        <dt>简介</dt><dd>${escapeHtml(data.description || "-")}</dd>
        <dt>相似车型</dt><dd>${escapeHtml(similar)}</dd>
      </dl>`);
  } catch (error) {
    openModal("车型资料", `<p class="modal-error">加载失败：${escapeHtml(String(error))}</p>`);
  }
}

function openPicker(title, items, onPick) {
  const html = `<div class="pick-list">${items.map((item) =>
    `<button type="button" class="pick-item" data-id="${escapeHtml(item.car_id)}">${escapeHtml(displayName(item.car_name))}
       <span class="badge">${escapeHtml(item.car_id)}</span></button>`).join("")}</div>`;
  openModal(title, html);
  document.querySelectorAll("#modalBody .pick-item").forEach((button) => {
    button.addEventListener("click", () => {
      const picked = items.find((item) => item.car_id === button.dataset.id);
      closeModal();
      if (picked) onPick(picked);
    });
  });
}

async function removeFavorite(carId) {
  const item = state.favorites.find((fav) => fav.car_id === carId);
  const name = (item && item.car_name) || carId;
  if (!window.confirm(`确定从收藏夹删除「${name}」吗？`)) return;
  try {
    const response = await fetch(`/api/favorites/${encodeURIComponent(carId)}`, { method: "DELETE" });
    const data = await response.json();
    if (!response.ok) {
      openModal("删除失败", `<p class="modal-error">${escapeHtml(data.detail || "删除失败")}</p>`);
      return;
    }
    await refreshFavorites();
  } catch (error) {
    openModal("删除失败", `<p class="modal-error">请求失败：${escapeHtml(String(error))}</p>`);
  }
}

async function compareFavorite(carId) {
  const name = displayName((state.favorites.find((item) => item.car_id === carId) || {}).car_name || carId);
  if (state.lastCarId && state.lastCarId !== carId) {
    await runCompare(carId, state.lastCarId);
    return;
  }
  const others = state.favorites.filter((item) => item.car_id !== carId);
  if (!others.length) {
    openModal("没法对比", '<p class="modal-error">收藏里只有这一辆车。先识别另一辆车，或再收藏一辆再来对比。</p>');
    return;
  }
  openPicker(`选择与「${name}」对比的车`, others, (other) => runCompare(carId, other.car_id));
}

async function runCompare(carIdA, carIdB) {
  try {
    const response = await fetch("/api/compare", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ car_id_a: carIdA, car_id_b: carIdB }),
    });
    const data = await response.json();
    if (!response.ok) {
      openModal("对比失败", `<p class="modal-error">${escapeHtml(data.detail || "对比失败")}</p>`);
      return;
    }
    const row = (label, key) => `<tr><th>${label}</th><td>${escapeHtml(data.car_a[key] || "-")}</td>
      <td>${escapeHtml(data.car_b[key] || "-")}</td></tr>`;
    openModal("车型对比", `<table class="cmp-table">
      <thead><tr><th></th><th>${escapeHtml(displayName(data.car_a.car_name))}</th><th>${escapeHtml(displayName(data.car_b.car_name))}</th></tr></thead>
      <tbody>${row("品牌", "brand")}${row("级别", "level")}${row("车身形式", "body_type")}${row("描述", "description")}
      <tr><th>结论</th><td colspan="2">${escapeHtml(data.summary)}</td></tr>
      <tr><th>建议</th><td colspan="2">${escapeHtml(data.advice || "-")}</td></tr></tbody></table>`);
  } catch (error) {
    openModal("对比失败", `<p class="modal-error">请求失败：${escapeHtml(String(error))}</p>`);
  }
}

async function refreshHistory() {
  const data = await fetch("/api/recognitions?limit=20").then((r) => r.json());
  const box = el("history");
  box.innerHTML = "";
  if (!data.count) {
    box.innerHTML = '<p class="empty">还没有识别记录</p>';
    return;
  }
  data.records.forEach((item) => {
    const div = document.createElement("div");
    div.className = "item";
    const unknown = item.predicted_label === "unknown";
    div.innerHTML = `<div class="title">${item.car_name || "unknown"}
        <span class="badge ${unknown ? "warn" : ""}">${item.predicted_label}</span></div>
      <div class="desc">${item.image_id} · 置信度 ${(item.confidence * 100).toFixed(1)}%</div>`;
    box.appendChild(div);
  });
}

let classCache = [];
async function loadClasses() {
  const data = await fetch("/api/classes").then((r) => r.json());
  classCache = data.classes;
  el("classCount").textContent = data.count;
  renderClasses("");
}

function renderClasses(filter) {
  const box = el("classes");
  const keyword = filter.trim().toLowerCase();
  const items = classCache.filter((c) =>
    !keyword || c.car_name.toLowerCase().includes(keyword) || c.brand.toLowerCase().includes(keyword));
  box.innerHTML = "";
  if (!items.length) {
    box.innerHTML = '<p class="empty">没有匹配的车型</p>';
    return;
  }
  items.forEach((c) => {
    const div = document.createElement("div");
    div.className = "item";
    div.innerHTML = `<div class="title">${c.car_name}<span class="badge">${c.car_id}</span></div>
      <div class="desc">${c.brand} · ${c.body_type} · ${c.level}</div>`;
    box.appendChild(div);
  });
}

async function checkStatus() {
  const data = await fetch("/api/status").then((r) => r.json());
  const parts = [];
  parts.push(data.model_loaded ? `视觉模型已加载（${data.model_arch || "-"}）` : "视觉模型未加载");
  parts.push(data.agent_ready ? `智能体就绪：${data.deepseek_model}` : "智能体未就绪");
  if (data.model_val_macro_f1) {
    const f1 = Number(data.model_val_macro_f1);
    parts.push(`验证集 Macro-F1 ${Number.isFinite(f1) ? f1.toFixed(4) : data.model_val_macro_f1}`);
  }
  el("status").innerHTML = parts.join(" · ") +
    (data.message ? `<br><span style="color:#c0392b">${data.message}</span>` : "");
}

async function startSession() {
  const data = await fetch("/api/sessions", { method: "POST" }).then((r) => r.json());
  state.sessionId = data.session_id;
}

el("image").addEventListener("change", (event) => {
  const file = event.target.files[0];
  state.image = file || null;
  el("uploadLabel").textContent = file ? file.name : "选择图片";
  el("uploadLabel").parentElement.classList.toggle("has-file", Boolean(file));
});

el("composer").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy) return;
  const text = el("message").value.trim();
  if (!text && !state.image) return;
  state.busy = true;
  el("send").disabled = true;

  let previewUrl = "";
  if (state.image) {
    previewUrl = URL.createObjectURL(state.image);
  }
  addBubble("user", text || "请看看这张图片", { imageUrl: previewUrl });
  el("message").value = "";

  const form = new FormData();
  form.append("session_id", state.sessionId);
  form.append("message", text);
  if (state.image) form.append("image", state.image);

  const thinking = addBubble("assistant", "正在识别与处理…");
  try {
    const response = await fetch("/api/chat", { method: "POST", body: form });
    const data = await response.json();
    thinking.textContent = data.reply || "（没有返回内容）";
    thinking.className = "bubble" + (data.error ? " error" : (data.low_confidence ? " warn" : ""));
    renderToolCalls(data.tool_calls || []);
    const classified = (data.tool_calls || []).find(
      (call) => call.name === "classify_car" && call.result && call.result.ok && !call.result.is_unknown);
    if (classified) state.lastCarId = classified.result.car_id;
    await Promise.all([refreshFavorites(), refreshHistory()]);
  } catch (error) {
    thinking.textContent = "请求失败：" + error;
    thinking.className = "bubble error";
  } finally {
    state.image = null;
    el("image").value = "";
    el("uploadLabel").textContent = "选择图片";
    el("uploadLabel").parentElement.classList.remove("has-file");
    state.busy = false;
    el("send").disabled = false;
    if (previewUrl) URL.revokeObjectURL(previewUrl);
  }
});

el("classFilter").addEventListener("input", (event) => renderClasses(event.target.value));

// ---------------------------------------------------------------- 测试集批量测试
function setBatchStatus(text, kind = "") {
  const box = el("batchStatus");
  box.textContent = text;
  box.className = "hint" + (kind === "error" ? " error-text" : "");
}

function renderBatchResult(data) {
  const summary = data.summary;
  const box = el("batchSummary");
  const metrics = [];
  metrics.push(`<span class="metric">图片 ${summary.total} 张</span>`);
  if (summary.has_ground_truth) {
    metrics.push(`<span class="metric">Macro-F1 ${summary.macro_f1}（51 类口径）</span>`);
    if (summary.present_classes && summary.present_classes < 51) {
      metrics.push(`<span class="metric">Macro-F1 ${summary.macro_f1_present}`
        + `（只算实际出现的 ${summary.present_classes} 个类别）</span>`);
    }
    metrics.push(`<span class="metric">Accuracy ${summary.accuracy}</span>`);
    metrics.push(`<span class="metric">unknown 检出率 ${summary.unknown_detection_rate}</span>`);
    metrics.push(`<span class="metric">已知类误拒率 ${summary.known_false_reject_rate}</span>`);
    metrics.push(`<span class="metric warn">错分 ${summary.misclassified} 张</span>`);
  } else {
    metrics.push(`<span class="metric">unknown ${summary.unknown_count} 张</span>`);
    metrics.push(`<span class="metric">平均置信度 ${summary.avg_confidence}</span>`);
    metrics.push(`<span class="metric warn">低置信度 ${summary.low_confidence_count} 张</span>`);
  }
  metrics.push(`<span class="metric">耗时 ${summary.elapsed_seconds}s</span>`);
  metrics.push(`<span class="metric">拒识阈值 ${summary.threshold}</span>`);
  if (summary.rejected_by_threshold) {
    metrics.push(`<span class="metric warn">按阈值改判 ${summary.rejected_by_threshold} 张</span>`);
  }
  const check = summary.check || {};
  const checkLine = check.ok
    ? `<span class="check-ok">✓ 三件套自检通过</span>（${check.rows} 行记录 / ${check.unknown} 张 unknown / npz ${check.npz_mib} MiB）`
    : `<span class="check-bad">✗ 自检发现问题：${(check.issues || []).join("；")}</span>`;
  box.innerHTML = `<div class="metrics">${metrics.join("")}</div>
    <div class="tiny">EduBench 提交文件：${checkLine}</div>
    <div class="link-row">
      <a href="${summary.predictions_url}" download>下载 predictions.csv</a>
      <a href="${summary.result_csv_url}" download>下载 result.csv</a>
      <a href="${summary.result_npz_url}" download>下载 result.npz</a>
      <a href="${summary.results_url}" download>下载明细表</a>
    </div>
    <div class="path-note">
      已写入提交目录：${escapeHtml(summary.submission_dir)}<br>
      已重新打包：${escapeHtml(summary.submission_zip)}
    </div>`;

  const list = el("batchResults");
  if (!data.preview || !data.preview.length) {
    list.innerHTML = "";
    return;
  }
  const rows = data.preview.map((item) => {
    const mark = item.correct === null || item.correct === undefined
      ? ""
      : (item.correct ? '<span class="ok">✓</span>' : '<span class="bad">✗</span>');
    const truth = item.true_label ? ` / 真实 ${escapeHtml(item.true_label)}` : "";
    return `<div class="result-row">
      <span class="name">${escapeHtml(item.image_id)}</span>
      <span>${escapeHtml(item.car_name || item.predicted_label)} · ${(item.confidence * 100).toFixed(1)}%${truth} ${mark}</span>
    </div>`;
  }).join("");
  list.innerHTML = `<div class="item"><div class="title">明细(前 ${data.preview.length} 条)</div>${rows}</div>`;
}

async function runBatchTest(formData) {
  el("runBatch").disabled = true;
  el("batchSummary").innerHTML = "";
  el("batchResults").innerHTML = "";
  el("batchProgress").hidden = false;
  el("batchBar").style.width = "0%";
  setBatchStatus("正在准备…");
  try {
    const started = await fetch("/api/batch/run", { method: "POST", body: formData })
      .then((r) => r.json());
    if (!started.ok) {
      setBatchStatus("测试失败: " + started.error, "error");
      el("batchProgress").hidden = true;
      return;
    }
    const total = started.total || 0;
    setBatchStatus(`共 ${total} 张图片, 已进入队列…`);
    while (true) {
      await new Promise((resolve) => setTimeout(resolve, 1000));
      const job = await fetch(`/api/batch/job/${started.job_id}`).then((r) => r.json());
      const processed = job.processed || 0;
      const known = job.total || total;
      const percent = known ? Math.round((processed / known) * 100) : 0;
      el("batchBar").style.width = percent + "%";
      if (job.state === "queued") {
        setBatchStatus("排队中(另一个批量测试正在运行)…");
        continue;
      }
      if (job.state === "running") {
        setBatchStatus(`正在测试… 已处理 ${processed} / ${known} 张（${percent}%），已用 ${job.elapsed}s`);
        continue;
      }
      if (job.state === "error") {
        setBatchStatus("测试失败: " + job.error, "error");
        el("batchProgress").hidden = true;
        return;
      }
      el("batchBar").style.width = "100%";
      setBatchStatus(`测试完成: ${job.summary.folder}（${job.summary.total} 张，用时 ${job.elapsed}s）`);
      renderBatchResult({ summary: job.summary, preview: job.preview });
      setTimeout(() => { el("batchProgress").hidden = true; }, 1500);
      return;
    }
  } catch (error) {
    setBatchStatus("请求失败: " + error, "error");
    el("batchProgress").hidden = true;
  } finally {
    el("runBatch").disabled = false;
  }
}

el("pickFolder").addEventListener("click", async () => {
  setBatchStatus("请在弹出的对话框里选择文件夹…");
  try {
    const data = await fetch("/api/batch/pick_folder", { method: "POST" }).then((r) => r.json());
    if (data.ok) {
      el("folderPath").value = data.path;
      setBatchStatus("已选择: " + data.path);
    } else {
      setBatchStatus(data.error, "error");
    }
  } catch (error) {
    setBatchStatus("无法打开文件夹对话框: " + error, "error");
  }
});

el("pickBrowserFolder").addEventListener("click", () => el("dirInput").click());

el("dirInput").addEventListener("change", (event) => {
  const files = Array.from(event.target.files || []);
  if (!files.length) return;
  const form = new FormData();
  form.append("mode", "upload");
  form.append("folder_path", "");
  form.append("threshold", el("threshold").value || "0");
  files.forEach((file) => form.append("files", file, file.webkitRelativePath || file.name));
  runBatchTest(form);
});

el("runBatch").addEventListener("click", () => {
  const path = el("folderPath").value.trim();
  if (!path) {
    setBatchStatus("请先选择文件夹, 或填写文件夹路径", "error");
    return;
  }
  const form = new FormData();
  form.append("mode", "path");
  form.append("folder_path", path);
  form.append("threshold", el("threshold").value || "0");
  runBatchTest(form);
});

// ---------------------------------------------------------------- 车源库筛选 / 多选
el("favFilter").addEventListener("input", renderFavorites);
el("favBrand").addEventListener("change", renderFavorites);
el("favLevel").addEventListener("change", renderFavorites);
el("clearSelection").addEventListener("click", clearSelection);
el("compareSelected").addEventListener("click", () => {
  const ids = [...state.selected];
  if (ids.length === 2) runCompare(ids[0], ids[1]);
});
document.querySelectorAll("#quickAsks button").forEach((button) => {
  button.addEventListener("click", () => {
    if (state.busy) return;
    el("message").value = button.dataset.q || "";
    el("composer").requestSubmit();
  });
});

(async function init() {
  await startSession();
  await checkStatus();
  await Promise.all([refreshFavorites(), refreshHistory(), loadClasses()]);
  addBubble("assistant",
    "你好，我是识车助手，给二手车商用的识车工具。\n" +
    "可以直接问“轻卡有哪些”“介绍一下大众朗逸”，也可以发一张车辆照片；收藏里的车能点开、能对比。\n" +
    "试试点下面的快捷问题，或上传一张车图。");
})();
