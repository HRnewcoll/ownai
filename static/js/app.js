/* OwnAI – Frontend JavaScript */

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function applyTheme(theme) {
  document.documentElement.setAttribute('data-theme', theme);
  localStorage.setItem('ownai-theme', theme);
  const icon = document.getElementById('themeIcon');
  const label = document.getElementById('themeLabel');
  if (icon) icon.className = theme === 'dark' ? 'fas fa-moon' : 'fas fa-sun';
  if (label) label.textContent = theme === 'dark' ? 'Dark Mode' : 'Light Mode';
}

document.addEventListener('DOMContentLoaded', () => {
  // Restore theme
  const saved = localStorage.getItem('ownai-theme') || 'dark';
  applyTheme(saved);

  // Theme toggle
  const themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', () => {
      const current = document.documentElement.getAttribute('data-theme');
      applyTheme(current === 'dark' ? 'light' : 'dark');
    });
  }

  // Sidebar toggle (desktop)
  const sidebarToggle = document.getElementById('sidebarToggle');
  const sidebar = document.getElementById('sidebar');
  const main = document.getElementById('mainContent');

  if (sidebarToggle && sidebar) {
    sidebarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('collapsed');
      if (main) main.classList.toggle('expanded');
    });
  }

  // Mobile sidebar toggle
  const topbarToggle = document.getElementById('topbarToggle');
  if (topbarToggle && sidebar) {
    topbarToggle.addEventListener('click', () => {
      sidebar.classList.toggle('mobile-open');
    });
    // Close on outside click
    document.addEventListener('click', (e) => {
      if (window.innerWidth <= 768 &&
          !sidebar.contains(e.target) &&
          e.target !== topbarToggle) {
        sidebar.classList.remove('mobile-open');
      }
    });
  }

  // Auto-dismiss alerts after 5s
  document.querySelectorAll('.alert').forEach(el => {
    setTimeout(() => el.remove(), 5000);
  });
});

// ---------------------------------------------------------------------------
// Model management
// ---------------------------------------------------------------------------

function deleteModel(id, name) {
  if (!confirm(`Delete model "${name}"? This cannot be undone.`)) return;
  fetch(`/api/models/${id}`, { method: 'DELETE' })
    .then(r => r.json())
    .then(d => {
      if (d.success) {
        // Remove card from DOM
        const card = document.getElementById(`modelCard${id}`);
        if (card) {
          card.style.transition = 'opacity .3s';
          card.style.opacity = '0';
          setTimeout(() => {
            card.remove();
            // If no more models, reload to show empty state
            const list = document.querySelector('.model-list, .model-list-full');
            if (list && !list.children.length) location.reload();
          }, 300);
        } else {
          location.reload();
        }
      } else {
        alert('Could not delete model: ' + (d.error || 'unknown error'));
      }
    })
    .catch(() => alert('Network error. Please try again.'));
}

// ---------------------------------------------------------------------------
// Utility helpers
// ---------------------------------------------------------------------------

function closeModal(id) {
  const el = document.getElementById(id);
  if (el) el.style.display = 'none';
}

// Close modal on backdrop click (global handler for dynamically created modals)
document.addEventListener('click', e => {
  if (e.target && e.target.classList.contains('modal-backdrop')) {
    e.target.closest('.modal').style.display = 'none';
  }
});

// Close modals on Escape key
document.addEventListener('keydown', e => {
  if (e.key === 'Escape') {
    document.querySelectorAll('.modal').forEach(m => m.style.display = 'none');
  }
});

// ---------------------------------------------------------------------------
// Toast notifications
// ---------------------------------------------------------------------------

function showToast(message, type = 'success') {
  const container = document.querySelector('.flash-container') || createToastContainer();
  const div = document.createElement('div');
  div.className = `alert alert-${type}`;
  div.innerHTML = `${message} <button class="alert-close" onclick="this.parentElement.remove()">×</button>`;
  container.appendChild(div);
  setTimeout(() => div.remove(), 4000);
}

function createToastContainer() {
  const div = document.createElement('div');
  div.className = 'flash-container';
  div.style.cssText = 'position:fixed;top:72px;right:1.5rem;z-index:999;min-width:300px;max-width:420px;';
  document.body.appendChild(div);
  return div;
}
