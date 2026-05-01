let currentUser = null;
let currentUserType = null;
let activeFilters = [];
let fridge = [];
let lastQuery = '';
let currentRecipeId = null;
let currentWhy = null;
let currentRating = 0;
let currentIngredients = [];

function goHome() {
  document.getElementById('q').value = '';
  lastQuery = '';
  closeRecipe();
  document.getElementById('list-title').textContent = 'Discover Recipes';
  document.getElementById('list-subtitle').textContent = 'Based on your available ingredients';
  loadRecs();
}

// ── HTTP helpers ───────────────────────────────────────────────────────────

async function get(url) {
  const response = await fetch(url);
  if (!response.ok) throw new Error(response.status);
  return response.json();
}

async function post(url, body) {
  const response = await fetch(url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  return response.json();
}

// ── Status + user badge ────────────────────────────────────────────────────

function setStatus(ok) {
  document.getElementById('status-dot').className = 'status-dot' + (ok ? ' active' : '');
  document.getElementById('status-label').textContent = ok ? 'Connected' : 'Offline';
}

function setUserTypeBadge(type) {
  const el = document.getElementById('user-type-badge');
  if (type === 'cold') {
    el.innerHTML = '<span class="user-type-chip cold">New user · popularity fallback</span>';
  } else if (type === 'session') {
    el.innerHTML = '<span class="user-type-chip session">New user · content-based</span>';
  } else if (type === 'known') {
    el.innerHTML = '<span class="user-type-chip known">Known user · personalized</span>';
  } else {
    el.innerHTML = '';
  }
}

// ── Recipe cards ───────────────────────────────────────────────────────────

function showList(recipes, emptyMessage = 'No results.') {
  const list = document.getElementById('list');

  if (!recipes || recipes.length === 0) {
    list.innerHTML = `<div class="msg">${emptyMessage}</div>`;
    return;
  }

  list.innerHTML = '';

  for (const recipe of recipes) {
    const card = document.createElement('div');
    card.className = 'card';

    const mins   = recipe.minutes    ? `<span class="meta-chip">⏱ ${recipe.minutes} min</span>` : '';
    const rating = recipe.avg_rating ? `<span class="meta-chip">★ ${recipe.avg_rating}</span>` : '';
    const cals   = recipe.calories   ? `<span class="meta-chip">🔥 ${Math.round(recipe.calories)} kcal</span>` : '';

    const desc = recipe.description || '';
    const shortDesc = desc.length > 130 ? desc.slice(0, 130) + '…' : desc;

    card.innerHTML = `
      <div class="card-img-wrap">
        <div class="card-img-fallback">
          <svg width="28" height="28" viewBox="0 0 24 24" fill="none" stroke="#d1d5db" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
        </div>
        <img class="card-img" src="/images/${recipe.recipe_id}.jpg"
             onerror="this.style.display='none'"
             onload="this.style.opacity='1'"
             alt="">
      </div>
      <div class="card-body">
        <div class="card-title">${recipe.name}</div>
        <div class="card-desc">${shortDesc}</div>
        <div class="card-meta">${mins}${rating}${cals}</div>
      </div>
      <svg class="card-arrow" width="16" height="16" viewBox="0 0 24 24" fill="none" stroke="currentColor" stroke-width="2"><path d="M7 17L17 7M17 7H7M17 7v10"/></svg>`;

    card.onclick = () => openRecipe(recipe.recipe_id, recipe.why ?? null);
    list.appendChild(card);
  }
}

// ── Recipe modal ───────────────────────────────────────────────────────────

async function openRecipe(id, why = null) {
  currentRecipeId = id;
  currentWhy = why;
  currentRating = 0;
  currentIngredients = [];

  const overlay = document.getElementById('modal-overlay');
  overlay.classList.add('open');
  document.body.style.overflow = 'hidden';

  document.getElementById('m-name').textContent = 'Loading…';
  document.getElementById('m-tag').textContent = 'Loading…';
  document.getElementById('m-rating-row').innerHTML = '';
  document.getElementById('m-stats').innerHTML = '';
  document.getElementById('m-ings').innerHTML = '';
  document.getElementById('m-steps').innerHTML = '';
  document.getElementById('m-desc').textContent = '';
  document.getElementById('m-why-section').style.display = 'none';
  document.getElementById('m-add-fridge').style.display = 'none';
  document.getElementById('m-ing-count').textContent = '';
  resetImgHeader();
  renderModalStars(0);

  try {
    const recipe = await get(`/recipe/${id}`);
    currentIngredients = recipe.ingredients || [];

    const imgWrap = document.getElementById('m-img');
    const img = new Image();
    img.src = `/images/${recipe.recipe_id}.jpg`;
    img.onload = () => {
      imgWrap.innerHTML = `<img src="/images/${recipe.recipe_id}.jpg" class="modal-header-img-el" alt="${recipe.name}">`;
    };

    document.getElementById('m-tag').textContent = guessMealTag(recipe);
    document.getElementById('m-rating-row').innerHTML =
      `<span class="modal-star-icon">★</span>
       <span class="modal-rating-val">${recipe.avg_rating}</span>
       <span class="modal-rating-count">(${recipe.n_ratings} reviews)</span>`;

    document.getElementById('m-name').textContent = recipe.name;

    const diff = difficulty(recipe);
    document.getElementById('m-stats').innerHTML = `
      <div class="stat-chip"><span class="stat-icon">⏱</span><div><span class="stat-label">PREP TIME</span><span class="stat-val">${recipe.minutes} mins</span></div></div>
      <div class="stat-chip"><span class="stat-icon">🍽</span><div><span class="stat-label">STEPS</span><span class="stat-val">${recipe.n_steps}</span></div></div>
      <div class="stat-chip" title="Estimated from prep time and number of steps"><span class="stat-icon">👨‍🍳</span><div><span class="stat-label">DIFFICULTY ~</span><span class="stat-val">${diff}</span></div></div>`;

    document.getElementById('m-ing-count').textContent = `(${currentIngredients.length})`;
    const fridgeSet = new Set(fridge.map(s => s.toLowerCase()));
    let hasMissing = false;

    const ingHtml = currentIngredients.map((ing, i) => {
      const ingLower = ing.toLowerCase();
      const inFridge = fridgeSet.has(ingLower) || [...fridgeSet].some(f => ingLower.includes(f) || f.includes(ingLower));
      if (!inFridge) hasMissing = true;
      return `<label class="ing-row ${inFridge ? 'in-fridge' : ''}">
        <input type="checkbox" class="ing-check" data-idx="${i}" ${inFridge ? 'checked' : ''}>
        <span class="ing-name">${ing}</span>
        ${inFridge ? '<span class="ing-fridge-tag">✓ fridge</span>' : ''}
      </label>`;
    }).join('');

    document.getElementById('m-ings').innerHTML = ingHtml;
    if (hasMissing) document.getElementById('m-add-fridge').style.display = '';

    document.getElementById('m-desc').textContent = recipe.description || '—';

    const steps = parseList(recipe.steps);
    document.getElementById('m-steps').innerHTML = steps.map((step, i) =>
      `<li class="modal-step"><span class="step-num">${i + 1}</span><p>${step}</p></li>`
    ).join('');

    renderWhy(why);
    renderModalStars(0);
  } catch {
    document.getElementById('m-name').textContent = 'Could not load recipe.';
  }
}

function resetImgHeader() {
  document.getElementById('m-img').innerHTML = `
    <div class="modal-img-placeholder">
      <svg width="44" height="44" viewBox="0 0 24 24" fill="none" stroke="#d1d5db" stroke-width="1.5"><rect x="3" y="3" width="18" height="18" rx="2"/><circle cx="8.5" cy="8.5" r="1.5"/><path d="m21 15-5-5L5 21"/></svg>
    </div>`;
}

function closeRecipe() {
  document.getElementById('modal-overlay').classList.remove('open');
  document.body.style.overflow = '';
}

function overlayClick(e) {
  if (e.target === document.getElementById('modal-overlay')) closeRecipe();
}

document.addEventListener('keydown', e => {
  if (e.key === 'Escape') closeRecipe();
});

document.addEventListener('DOMContentLoaded', () => {
  document.getElementById('q').addEventListener('keydown', e => {
    if (e.key === 'Enter') search();
  });
  document.getElementById('fi').addEventListener('keydown', e => {
    if (e.key === 'Enter') addFridge();
  });
});

function difficulty(recipe) {
  if (recipe.minutes <= 15 && recipe.n_steps <= 5) return 'Beginner';
  if (recipe.minutes <= 45 && recipe.n_steps <= 10) return 'Easy';
  if (recipe.minutes <= 90 && recipe.n_steps <= 15) return 'Intermediate';
  return 'Advanced';
}

function guessMealTag(recipe) {
  const tags = new Set((recipe.tags || []).map(t => t.toLowerCase()));
  const name = (recipe.name || '').toLowerCase();
  if (tags.has('breakfast') || name.includes('breakfast') || name.includes('egg') || name.includes('pancake') || name.includes('waffle')) return 'Best for Breakfast';
  if (tags.has('dessert') || name.includes('cake') || name.includes('cookie') || name.includes('brownie') || name.includes('pie')) return 'Best for Dessert';
  if (tags.has('lunch') || name.includes('sandwich') || name.includes('salad') || name.includes('wrap')) return 'Best for Lunch';
  if (tags.has('dinner') || name.includes('steak') || name.includes('roast') || name.includes('pasta')) return 'Best for Dinner';
  if (tags.has('snack') || name.includes('dip') || name.includes('chip') || name.includes('snack')) return 'Great Snack';
  return 'Featured Recipe';
}

function addMissingToFridge() {
  const fridgeSet = new Set(fridge.map(s => s.toLowerCase()));
  const missing = currentIngredients.filter(ing => {
    const ingLower = ing.toLowerCase();
    return !fridgeSet.has(ingLower) && ![...fridgeSet].some(f => ingLower.includes(f) || f.includes(ingLower));
  });
  for (const ing of missing) {
    if (!fridge.includes(ing)) fridge.push(ing);
  }
  renderFridge();
  document.getElementById('m-add-fridge').textContent = `✓ Added ${missing.length} item${missing.length !== 1 ? 's' : ''}`;
  document.getElementById('m-add-fridge').disabled = true;
  if (!lastQuery) loadRecs();
}

// ── Why recommended ────────────────────────────────────────────────────────

const STRATEGY_LABEL = {
  personalized:    '⭐ Matched to your taste',
  session_content: '🎯 Based on your recent ratings',
  seeds:           '🔍 Similar to your search',
  search_reranked: '🔍 Search result, ranked for you',
  popular:         '🔥 Popular pick',
};

function renderWhy(why) {
  const section = document.getElementById('m-why-section');
  if (!why) {
    section.style.display = 'none';
    return;
  }
  const label = STRATEGY_LABEL[why.strategy] || why.strategy;
  let html = `<div class="why-strategy">${label}</div>
              <div class="why-rank">Retrieved #${why.retrieval_rank} of 100 candidates</div>`;
  if (why.fridge_pct > 0) {
    const ingredients = why.fridge_matched.map(i => `<span class="why-ing">${i}</span>`).join('');
    html += `<div class="why-fridge">
               <span class="why-fridge-label">${why.fridge_pct}% fridge match</span>
               ${ingredients}
             </div>`;
  }
  document.getElementById('m-why').innerHTML = html;
  section.style.display = '';
}

// ── Star rating ────────────────────────────────────────────────────────────

function renderModalStars(selected) {
  currentRating = selected;
  const container = document.getElementById('m-stars');
  container.innerHTML = '';

  for (let i = 1; i <= 5; i++) {
    const star = document.createElement('span');
    star.className = 'star' + (i <= selected ? ' on' : '');
    star.textContent = '★';
    star.onmouseenter = () => hoverStars(i);
    star.onmouseleave = () => renderModalStars(selected);
    star.onclick = () => submitRating(i);
    container.appendChild(star);
  }
}

function hoverStars(hoveredIndex) {
  const stars = document.querySelectorAll('#m-stars .star');
  stars.forEach((star, i) => {
    if (i < hoveredIndex) {
      star.classList.add('on');
    } else {
      star.classList.remove('on');
    }
  });
}

async function submitRating(value) {
  renderModalStars(value);
  try {
    await post(`/user/${currentUser}/rate`, { recipe_id: currentRecipeId, rating: value });
    // If this was a cold-start user, they now have session ratings
    if (currentUserType === 'cold') {
      currentUserType = 'session';
      setUserTypeBadge('session');
    }
    loadReviewed();
    loadRecs();
  } catch {}
}

function resetRating() {
  renderModalStars(0);
}

// ── Recommendations ────────────────────────────────────────────────────────

async function loadRecs() {
  document.getElementById('list').innerHTML = '<div class="msg">Loading…</div>';
  document.getElementById('list-title').textContent = 'Discover Recipes';
  document.getElementById('list-subtitle').textContent = 'Based on your available ingredients';

  let url = `/recommendations/${currentUser}?top_k=10`;
  if (activeFilters.length > 0) url += `&preferences=${encodeURIComponent(activeFilters.join(','))}`;
  for (const ing of fridge) url += `&fridge=${encodeURIComponent(ing)}`;

  try {
    const data = await get(url);
    showList(data.recommendations, 'No recommendations yet — rate some recipes!');
  } catch {
    document.getElementById('list').innerHTML = '<div class="msg">Could not load recommendations.</div>';
  }
}

async function loadReviewed() {
  const el = document.getElementById('reviews');
  try {
    const data = await get(`/user/${currentUser}`);
    el.innerHTML = data.ratings.slice(0, 6).map(r =>
      `<div class="review-row">
         <span class="review-name">${r.name.slice(0, 24)}</span>
         <span class="review-score">${r.rating}/5 <span class="review-star">★</span></span>
       </div>`
    ).join('') || '<span class="msg">None yet.</span>';
  } catch {
    el.innerHTML = '<span class="msg">None yet.</span>';
  }
}

async function loadUser(id, type = 'known') {
  currentUser = String(id);
  currentUserType = type;
  document.getElementById('uid').textContent = currentUser;
  fridge = [];
  activeFilters = [];
  document.getElementById('q').value = '';
  lastQuery = '';
  document.querySelectorAll('#diet-pills .pill').forEach(p => p.classList.remove('on'));

  if (type === 'known') {
    try {
      const data = await get(`/user/${currentUser}`);
      if (data.profile?.fridge?.length) fridge = data.profile.fridge;
      if (data.profile?.diet) activeFilters = [data.profile.diet];
    } catch {}
  }

  setUserTypeBadge(type);
  renderFridge();
  await Promise.all([loadRecs(), loadReviewed()]);
}

async function randomUser() {
  const btn = document.querySelector('.icon-btn[title="Random training user"]');
  const prevContent = btn?.innerHTML;
  if (btn) {
    btn.textContent = '⏳';
    btn.disabled = true;
  }
  closeRecipe();
  try {
    const data = await get('/random-user');
    await loadUser(data.user_id, 'known');
  } catch {
    document.getElementById('list').innerHTML = '<div class="msg">Could not reach the server — is the backend running?</div>';
    setStatus(false);
  } finally {
    if (btn) {
      btn.innerHTML = prevContent;
      btn.disabled = false;
    }
  }
}

async function coldStartUser() {
  closeRecipe();
  // Generate a random 8-character ID so cold-start users don't collide
  const randomId = Math.random().toString(36).slice(2, 10);
  await loadUser(`new-${randomId}`, 'cold');
}

// ── Search ─────────────────────────────────────────────────────────────────

async function search() {
  lastQuery = document.getElementById('q').value.trim();
  if (!lastQuery) {
    loadRecs();
    return;
  }

  document.getElementById('list').innerHTML = '<div class="msg">Searching…</div>';
  document.getElementById('list-title').textContent = `Results for "${lastQuery}"`;
  document.getElementById('list-subtitle').textContent = 'Searching…';

  const [textResult, semanticResult] = await Promise.allSettled([
    get(`/search?q=${encodeURIComponent(lastQuery)}&limit=20&mode=text`),
    get(`/search?q=${encodeURIComponent(lastQuery)}&limit=20&mode=semantic`),
  ]);

  const seen = new Set();
  const merged = [];
  const textRecipes = textResult.value?.results || [];
  const semanticRecipes = semanticResult.value?.results || [];

  for (const recipe of [...textRecipes, ...semanticRecipes]) {
    if (!seen.has(recipe.recipe_id)) {
      seen.add(recipe.recipe_id);
      merged.push(recipe);
    }
  }

  if (merged.length === 0) {
    showList([], 'No results found.');
    document.getElementById('list-subtitle').textContent = 'No matches';
    return;
  }

  showList(merged.slice(0, 20));
  document.getElementById('list-subtitle').textContent = 'Personalizing…';

  try {
    const currentQuery = lastQuery;
    const data = await post(`/rerank/${currentUser}`, {
      recipe_ids: merged.map(r => r.recipe_id),
      fridge,
      preferences: activeFilters.length > 0 ? activeFilters.join(',') : null,
    });
    // Only update the list if the user hasn't typed a new query since this one started
    if (lastQuery === currentQuery) {
      showList(data.results, 'No results found.');
      document.getElementById('list-subtitle').textContent = `${data.results.length} personalized results`;
    }
  } catch {
    document.getElementById('list-subtitle').textContent = 'Text + semantic search';
  }
}

// ── Dietary filters ────────────────────────────────────────────────────────

function toggleFilter(element, filter) {
  const idx = activeFilters.indexOf(filter);
  if (idx > -1) {
    activeFilters.splice(idx, 1);
    element.classList.remove('on');
  } else {
    activeFilters.push(filter);
    element.classList.add('on');
  }
  if (!lastQuery) loadRecs();
}

// ── Fridge ─────────────────────────────────────────────────────────────────

let fridgeExpanded = false;
const FRIDGE_COLLAPSE_THRESHOLD = 6;

function toggleFridge() {
  fridgeExpanded = !fridgeExpanded;
  renderFridge();
}

function renderFridge() {
  const container = document.getElementById('fridge');
  const toggle = document.getElementById('fridge-toggle');
  container.innerHTML = '';

  const visible = (fridgeExpanded || fridge.length <= FRIDGE_COLLAPSE_THRESHOLD)
    ? fridge
    : fridge.slice(0, FRIDGE_COLLAPSE_THRESHOLD);

  visible.forEach((ing) => {
    const pill = document.createElement('span');
    pill.className = 'pill on';
    pill.textContent = ing + ' ✕';
    pill.onclick = () => removeFridge(fridge.indexOf(ing));
    container.appendChild(pill);
  });

  if (fridge.length > FRIDGE_COLLAPSE_THRESHOLD) {
    toggle.style.display = '';
    toggle.textContent = fridgeExpanded
      ? 'Show less'
      : `Show all (${fridge.length - FRIDGE_COLLAPSE_THRESHOLD} more)`;
  } else {
    toggle.style.display = 'none';
  }
}

function removeFridge(i) {
  fridge.splice(i, 1);
  renderFridge();
  if (!lastQuery) loadRecs();
}

function addFridge() {
  const input = document.getElementById('fi');
  const value = input.value.trim().toLowerCase();
  if (!value || fridge.includes(value)) return;
  fridge.push(value);
  input.value = '';
  renderFridge();
  if (!lastQuery) loadRecs();
}

// ── Edit user ID ───────────────────────────────────────────────────────────

function editUser() {
  const label = document.getElementById('uid');
  const input = document.createElement('input');
  input.className = 'user-id-input';
  input.value = currentUser;
  label.replaceWith(input);
  input.focus();
  input.select();

  async function confirmEdit() {
    const newId = input.value.trim();
    const newLabel = document.createElement('b');
    newLabel.id = 'uid';
    newLabel.textContent = newId || currentUser;
    input.replaceWith(newLabel);
    if (newId && newId !== currentUser) await loadUser(newId, 'known');
  }

  input.onblur = confirmEdit;
  input.onkeydown = e => {
    if (e.key === 'Enter') input.blur();
    if (e.key === 'Escape') { input.value = currentUser; input.blur(); }
  };
}

// ── Utilities ──────────────────────────────────────────────────────────────

/*
 * parseList converts a recipe's steps or ingredients field into an array of strings.
 *
 * The data comes from Python and may arrive in different formats:
 *   - Already a JS array: ["step one", "step two"]
 *   - A JSON string: "[\"step one\", \"step two\"]"
 *   - A Python stringified list: "['step one', 'step two']"  ← single quotes, not valid JSON
 *
 * JSON.parse handles the second case. The manual parser below handles the third.
 */
function parseList(str) {
  if (!str) return [];
  if (Array.isArray(str)) return str;

  // Try standard JSON first
  try {
    const parsed = JSON.parse(str);
    if (Array.isArray(parsed)) return parsed;
  } catch {}

  // Fall back to manually parsing a Python-style list like ['a', 'b', 'c']
  const trimmed = str.trim();
  if (trimmed.startsWith('[') && trimmed.endsWith(']')) {
    const inner = trimmed.slice(1, -1);
    const items = [];
    let current = '';
    let inString = false;
    let quoteChar = '';

    for (let i = 0; i < inner.length; i++) {
      const ch = inner[i];

      if (!inString && (ch === "'" || ch === '"')) {
        inString = true;
        quoteChar = ch;
        continue;
      }
      if (inString && ch === quoteChar && inner[i - 1] !== '\\') {
        inString = false;
        continue;
      }
      if (!inString && ch === ',' && inner[i + 1] === ' ') {
        items.push(current.trim());
        current = '';
        i++; // skip the space after the comma
        continue;
      }
      current += ch;
    }

    if (current.trim()) items.push(current.trim());
    if (items.length > 0) return items;
  }

  return [str];
}

// ── Init ───────────────────────────────────────────────────────────────────

async function init() {
  try {
    const { diets } = await get('/diets');
    const container = document.getElementById('diet-pills');
    container.innerHTML = '';
    for (const diet of diets) {
      const pill = document.createElement('span');
      pill.className = 'pill';
      pill.textContent = diet.charAt(0).toUpperCase() + diet.slice(1).replace('-', ' ');
      pill.onclick = () => toggleFilter(pill, diet);
      container.appendChild(pill);
    }
  } catch {
    document.getElementById('diet-pills').innerHTML = '<span class="msg">—</span>';
  }

  try {
    const data = await get('/random-user');
    setStatus(true);
    await loadUser(data.user_id, 'known');
  } catch {
    setStatus(false);
    document.getElementById('uid').textContent = '—';
    document.getElementById('list').innerHTML = '<div class="msg">Backend not available — start the server and refresh.</div>';
    document.getElementById('list-title').textContent = 'Server Offline';
    document.getElementById('list-subtitle').textContent = 'uvicorn app.main:app --reload';
  }
}

init();
