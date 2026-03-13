/**
 * LEAA Archery Demo — UI Controller
 */

(function () {
    'use strict';

    let scene;
    let stats = { shots: 0, hits: 0 };
    let firing = false;

    // DOM refs
    const commandInput = document.getElementById('command-input');
    const fireBtn = document.getElementById('fire-btn');
    const randomizeBtn = document.getElementById('randomize-btn');
    const resultBanner = document.getElementById('result-banner');
    const statShots = document.getElementById('stat-shots');
    const statHits = document.getElementById('stat-hits');
    const statAccuracy = document.getElementById('stat-accuracy');

    // -----------------------------------------------------------------------
    // Init
    // -----------------------------------------------------------------------

    window.addEventListener('DOMContentLoaded', () => {
        const container = document.getElementById('canvas-container');
        scene = new ArcheryScene(container);

        fetch('/api/scene')
            .then(r => r.json())
            .then(data => {
                scene.buildTargets(data);
            });

        fireBtn.addEventListener('click', doFire);
        randomizeBtn.addEventListener('click', doRandomize);
        commandInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doFire();
        });
    });

    // -----------------------------------------------------------------------
    // Fire
    // -----------------------------------------------------------------------

    function doFire() {
        if (firing) return;
        const command = commandInput.value.trim();
        if (!command) {
            commandInput.placeholder = 'Enter a command first!';
            return;
        }

        firing = true;
        fireBtn.disabled = true;
        hideBanner();

        fetch('/api/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command }),
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showBanner(data.error, false);
                    firing = false;
                    fireBtn.disabled = false;
                    return;
                }

                // Highlight target
                scene.highlightTarget(data.target_id);

                // Animate arrow
                scene.fireArrow(data.trajectory_points, () => {
                    // Update stats
                    stats.shots++;
                    if (data.hit) stats.hits++;
                    updateStats();

                    // Show result
                    showBanner(data.result_text, data.hit);

                    firing = false;
                    fireBtn.disabled = false;
                });
            })
            .catch(err => {
                console.error('Fire error:', err);
                showBanner('Error communicating with server', false);
                firing = false;
                fireBtn.disabled = false;
            });
    }

    // -----------------------------------------------------------------------
    // Randomize
    // -----------------------------------------------------------------------

    function doRandomize() {
        fetch('/api/randomize', { method: 'POST' })
            .then(r => r.json())
            .then(data => {
                scene.buildTargets(data);
                hideBanner();
            });
    }

    // -----------------------------------------------------------------------
    // UI Helpers
    // -----------------------------------------------------------------------

    function updateStats() {
        statShots.textContent = stats.shots;
        statHits.textContent = stats.hits;
        const acc = stats.shots > 0 ? Math.round((stats.hits / stats.shots) * 100) : 0;
        statAccuracy.textContent = acc + '%';
    }

    function showBanner(text, isHit) {
        resultBanner.textContent = text;
        resultBanner.className = isHit ? 'show hit' : 'show miss';
        // Auto-hide after 4 seconds
        clearTimeout(resultBanner._hideTimer);
        resultBanner._hideTimer = setTimeout(hideBanner, 4000);
    }

    function hideBanner() {
        resultBanner.className = 'hidden';
    }
})();
