/**
 * LEAA Archery Demo — UI Controller
 *
 * Heuristic mode: Full mouse-driven game
 *   - Click a target to select it (highlight)
 *   - Hold left-click to draw bow (camera zooms to aim POV)
 *   - Release to fire arrow at selected target
 *
 * RL mode: Text command input (future — model not loaded yet)
 */

(function () {
    'use strict';

    let scene;
    let stats = { shots: 0, hits: 0 };
    let firing = false;
    let currentMode = 'heuristic';
    let isMouseAiming = false;
    let selectedTargetId = null;

    // DOM refs
    const commandInput = document.getElementById('command-input');
    const fireBtn = document.getElementById('fire-btn');
    const randomizeBtn = document.getElementById('randomize-btn');
    const resultBanner = document.getElementById('result-banner');
    const statShots = document.getElementById('stat-shots');
    const statHits = document.getElementById('stat-hits');
    const statAccuracy = document.getElementById('stat-accuracy');
    const modeHeuristicBtn = document.getElementById('mode-heuristic');
    const modeRlBtn = document.getElementById('mode-rl');
    const modeLabel = document.getElementById('mode-label');
    const modeStatus = document.getElementById('mode-status');
    const windDirection = document.getElementById('wind-direction');
    const windSpeed = document.getElementById('wind-speed');
    const windSpeedLabel = document.getElementById('wind-speed-label');
    const commandBar = document.getElementById('command-bar');
    const gameHint = document.getElementById('game-hint');

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

        randomizeBtn.addEventListener('click', doRandomize);

        // RL mode text fire
        if (fireBtn) fireBtn.addEventListener('click', doFireText);
        if (commandInput) commandInput.addEventListener('keydown', (e) => {
            if (e.key === 'Enter') doFireText();
        });

        // Mode toggle
        modeHeuristicBtn.addEventListener('click', () => setMode('heuristic'));
        modeRlBtn.addEventListener('click', () => setMode('rl'));

        // Wind controls
        windSpeed.addEventListener('input', updateWind);
        windDirection.addEventListener('change', updateWind);

        // -------------------------------------------------------------------
        // Mouse-driven game (heuristic mode)
        // Listen on document for reliability — check target to ignore UI clicks
        // -------------------------------------------------------------------

        document.addEventListener('mousedown', (e) => {
            if (e.button !== 0) return;
            // Ignore clicks on UI elements
            if (e.target.closest('#wind-panel, #mode-toggle, #hud, #bottom-bar, #command-bar, #game-hint, button, input, select, label')) return;
            if (firing || currentMode !== 'heuristic') return;

            // Click anywhere on the scene to start aiming — no target selection needed
            isMouseAiming = true;
            selectedTargetId = null; // aim-direction based, no pre-selected target
            scene.enterAimMode();
            e.preventDefault();
        });

        // mouseup on document so it fires even if cursor drifts off canvas
        document.addEventListener('mouseup', (e) => {
            if (e.button !== 0 || !isMouseAiming) return;
            isMouseAiming = false;
            doFireGame();
            e.preventDefault();
        });

        // Edge case: mouse leaves window or tab loses focus
        window.addEventListener('blur', () => {
            if (isMouseAiming) {
                isMouseAiming = false;
                scene.exitAimMode();
            }
        });

        document.addEventListener('contextmenu', (e) => {
            if (isMouseAiming) e.preventDefault();
        });

        // Start in heuristic (game) mode
        setMode('heuristic');
    });

    // -----------------------------------------------------------------------
    // Fire — Game mode (mouse click on target)
    // -----------------------------------------------------------------------

    function doFireGame() {
        if (firing) {
            scene.exitAimMode();
            return;
        }

        // Capture aim direction before exiting aim mode (which resets it)
        const aimDir = scene.getAimDirection();
        const aimData = { x: aimDir.x, y: aimDir.y, z: aimDir.z };

        firing = true;
        hideBanner();
        scene.exitAimMode();

        const body = {
            wind: getWindParams(),
            aim_direction: aimData,
        };
        // Include target_id only if one was selected
        if (selectedTargetId) body.target_id = selectedTargetId;

        fetch('/api/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(body),
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showBanner(data.error, false);
                    firing = false;
                    return;
                }

                scene.highlightTarget(data.target_id);

                scene.fireArrow(data.trajectory_points, () => {
                    stats.shots++;
                    if (data.hit) stats.hits++;
                    updateStats();
                    showBanner(data.result_text, data.hit);
                    firing = false;
                }, true);
            })
            .catch(err => {
                console.error('Fire error:', err);
                showBanner('Error communicating with server', false);
                firing = false;
            });
    }

    // -----------------------------------------------------------------------
    // Fire — Text command mode (RL / fallback)
    // -----------------------------------------------------------------------

    function doFireText() {
        if (firing) return;
        const command = commandInput.value.trim();
        if (!command) {
            commandInput.placeholder = 'Enter a command first!';
            return;
        }

        firing = true;
        if (fireBtn) fireBtn.disabled = true;
        hideBanner();

        fetch('/api/fire', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ command, wind: getWindParams() }),
        })
            .then(r => r.json())
            .then(data => {
                if (data.error) {
                    showBanner(data.error, false);
                    firing = false;
                    if (fireBtn) fireBtn.disabled = false;
                    return;
                }

                scene.highlightTarget(data.target_id);

                scene.fireArrow(data.trajectory_points, () => {
                    stats.shots++;
                    if (data.hit) stats.hits++;
                    updateStats();
                    showBanner(data.result_text, data.hit);
                    firing = false;
                    if (fireBtn) fireBtn.disabled = false;
                }, false);
            })
            .catch(err => {
                console.error('Fire error:', err);
                showBanner('Error communicating with server', false);
                firing = false;
                if (fireBtn) fireBtn.disabled = false;
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
                selectedTargetId = null;
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
        clearTimeout(resultBanner._hideTimer);
        resultBanner._hideTimer = setTimeout(hideBanner, 4000);
    }

    function hideBanner() {
        resultBanner.className = 'hidden';
    }

    // -----------------------------------------------------------------------
    // Wind
    // -----------------------------------------------------------------------

    function updateWind() {
        const mph = parseInt(windSpeed.value);
        windSpeedLabel.textContent = mph + ' mph';
        const dir = windDirection.value;
        if (scene) {
            scene.windSpeed = mph;
            scene.windDirection = dir;
        }
    }

    function getWindParams() {
        const mph = parseInt(windSpeed.value);
        const ms = mph * 0.44704;
        const dir = windDirection.value;
        const dirMap = {
            'none':     [0, 0, 0],
            'headwind': [-1, 0, 0],
            'tailwind': [1, 0, 0],
            'left':     [0, 1, 0],
            'right':    [0, -1, 0],
        };
        return { speed: ms, direction: dirMap[dir] || [0, 0, 0] };
    }

    // -----------------------------------------------------------------------
    // Mode Toggle
    // -----------------------------------------------------------------------

    function setMode(mode) {
        currentMode = mode;
        if (mode === 'heuristic') {
            modeHeuristicBtn.classList.add('active');
            modeRlBtn.classList.remove('active');
            modeLabel.textContent = 'Game Mode';
            modeStatus.textContent = 'ACTIVE';
            modeStatus.className = 'mode-status online';

            // Hide text command bar, show game hint
            commandBar.className = 'hidden';
            if (gameHint) gameHint.className = 'show';
            document.body.classList.add('game-mode');
        } else {
            modeRlBtn.classList.add('active');
            modeHeuristicBtn.classList.remove('active');
            modeLabel.textContent = 'Trained RL Policy';
            modeStatus.textContent = 'MODEL NOT LOADED';
            modeStatus.className = 'mode-status offline';

            // Show text command bar, hide game hint
            commandBar.className = '';
            if (gameHint) gameHint.className = 'hidden';
            document.body.classList.remove('game-mode');
        }
    }
})();
