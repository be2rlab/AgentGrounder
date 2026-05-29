/*!
  script.js
  ──────────────────────────────────────────────────────────────────────────────
  All interactivity for the AgentGrounder project page:
    - Sticky nav:   adds a shadow class on scroll, highlights active section
                    links via IntersectionObserver.
    - BibTeX copy:  copies the code block content to clipboard + shows a
                    brief "Copied!" confirmation.
    - Button hover  lift/scale handled in CSS, but we also handle the
   interaction:   copy-button state here.
   Dependency-free, respects prefers-reduced-motion.
  ──────────────────────────────────────────────────────────────────────────────
*/

(function () {
  "use strict";

  /* ═══════════════════════════════════════════════════════════════════════════
     Helpers
     ═════════════════════════════════════════════════════════════════════════── */

  /** True when the user prefers reduced motion. */
  const prefersReducedMotion = () =>
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;

  /** Debounce — returns a function that delays invoking fn. */
  function debounce(fn, ms) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), ms);
    };
  }


  /* ═══════════════════════════════════════════════════════════════════════════
     1.  Sticky nav — shadow on scroll
     ═══════════════════════════════════════════════════════════════════════════ */

  const nav = document.getElementById("navbar");
  if (nav) {
    const toggleShadow = () => {
      nav.classList.toggle("nav--scrolled", window.scrollY > 10);
    };
    window.addEventListener("scroll", toggleShadow, { passive: true });
    toggleShadow(); // set initial state
  }


  /* ═══════════════════════════════════════════════════════════════════════════
     2.  Active nav-link highlighting via IntersectionObserver
     ═══════════════════════════════════════════════════════════════════════════ */

  const navLinks = document.querySelectorAll(".nav__link");
  if (navLinks.length && "IntersectionObserver" in window) {
    const sections = [];
    navLinks.forEach((link) => {
      const id = link.getAttribute("href").replace("#", "");
      const el = document.getElementById(id);
      if (el) sections.push({ el, link });
    });

    const observer = new IntersectionObserver(
      (entries) => {
        entries.forEach((entry) => {
          if (entry.isIntersecting) {
            // remove active from all, then add to the intersecting one
            navLinks.forEach((l) => l.classList.remove("nav__link--active"));
            const match = sections.find((s) => s.el === entry.target);
            if (match) match.link.classList.add("nav__link--active");
          }
        });
      },
      {
        rootMargin: "-40% 0px -55% 0px", // trigger when section is near top
        threshold: 0,
      }
    );

    sections.forEach(({ el }) => observer.observe(el));
  }


  /* ═══════════════════════════════════════════════════════════════════════════
     3.  BibTeX copy-to-clipboard
     ═══════════════════════════════════════════════════════════════════════════ */

  const copyBtn = document.getElementById("bibtexCopyBtn");
  const codeEl = document.getElementById("bibtexCode");

  if (copyBtn && codeEl) {
    copyBtn.addEventListener("click", async () => {
      try {
        await navigator.clipboard.writeText(codeEl.textContent.trim());
      } catch {
        // Fallback for older browsers
        const ta = document.createElement("textarea");
        ta.value = codeEl.textContent.trim();
        ta.style.position = "fixed";
        ta.style.left = "-9999px";
        document.body.appendChild(ta);
        ta.select();
        document.execCommand("copy");
        document.body.removeChild(ta);
      }

      copyBtn.textContent = "Copied!";
      copyBtn.classList.add("bibtex__copy--copied");

      setTimeout(() => {
        copyBtn.textContent = "Copy";
        copyBtn.classList.remove("bibtex__copy--copied");
      }, 2000);
    });
  }


  /* ═══════════════════════════════════════════════════════════════════════════
     4.  Smooth-scroll for anchor links (progressive enhancement)
     ═══════════════════════════════════════════════════════════════════════════ */

  /* ═══════════════════════════════════════════════════════════════════════════
     5.  Dark mode toggle
     ═══════════════════════════════════════════════════════════════════════════ */

  const themeToggle = document.getElementById("themeToggle");
  if (themeToggle) {
    const stored = localStorage.getItem("theme");
    if (stored === "dark") {
      document.documentElement.setAttribute("data-theme", "dark");
    }

    themeToggle.addEventListener("click", () => {
      const html = document.documentElement;
      const isDark = html.getAttribute("data-theme") === "dark";
      if (isDark) {
        html.removeAttribute("data-theme");
        localStorage.setItem("theme", "light");
      } else {
        html.setAttribute("data-theme", "dark");
        localStorage.setItem("theme", "dark");
      }
    });
  }

  /* ═══════════════════════════════════════════════════════════════════════════
     6.  Smooth-scroll for anchor links (progressive enhancement)
     ═══════════════════════════════════════════════════════════════════════════ */

  document.querySelectorAll('a[href^="#"]').forEach((anchor) => {
    anchor.addEventListener("click", (e) => {
      const href = anchor.getAttribute("href");
      if (href === "#") return;
      const target = document.getElementById(href.slice(1));
      if (!target || prefersReducedMotion()) return;

      e.preventDefault();
      const top = target.getBoundingClientRect().top + window.scrollY - 72;
      window.scrollTo({ top, behavior: "smooth" });
    });
  });

})();
