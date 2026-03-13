/**
 * LEAA Archery Demo — Three.js Scene (PBR / Cinematic)
 *
 * Realistic archery range with PBR materials, smooth shading,
 * ACES tone mapping, and anatomically detailed character.
 * Physics coords [x,y,z] are received as Three.js [x,z,y] from the backend.
 */

class ArcheryScene {
    constructor(container) {
        this.container = container;
        this.targets = {};
        this.targetMeshes = {};
        this.highlightRings = {};
        this.flags = {};
        this.arrowMesh = null;
        this.trailLine = null;
        this.particles = [];
        this.animatingArrow = false;
        this.clock = new THREE.Clock();

        this.windSpeed = 0;     // mph, set from UI
        this.windDirection = 'none';
        this.treeFoliage = [];

        // Aim mode state
        this.isAiming = false;
        this.aimStartTime = 0;
        this.aimPower = 0;          // 0-1, builds over time
        this._cameraDefault = { pos: new THREE.Vector3(-4.85, 2.21, 0.41), target: new THREE.Vector3(15.00, 1.30, -0.60) };
        this._cameraAim = { pos: new THREE.Vector3(-1.2, 1.85, 0.65), target: new THREE.Vector3(20.00, 1.50, 0.00) };
        this._cameraLerp = 0;       // 0 = default, 1 = aim POV
        this._cameraLerpDir = 0;    // +1 zooming in, -1 zooming out, 0 idle

        // Mouse-look aiming state
        this._aimYaw = 0;           // horizontal offset from mouse
        this._aimPitch = 0;         // vertical offset from mouse
        this._aimMouseStart = null; // { x, y } when aim mode started

        // GLB model cache
        this._glbModels = {};

        this._initRenderer();
        this._initCamera();
        this._initLights();
        this._buildEnvironment();
        this._buildArcher();
        this._loadGLBProps();
        this.animate();
    }

    // -------------------------------------------------------------------
    // Renderer — PBR pipeline, ACES tone mapping, high-res shadows
    // -------------------------------------------------------------------

    _initRenderer() {
        this.renderer = new THREE.WebGLRenderer({
            antialias: true,
            powerPreference: 'high-performance',
        });
        this.renderer.setSize(window.innerWidth, window.innerHeight);
        this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
        this.renderer.shadowMap.enabled = true;
        this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
        this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
        this.renderer.toneMappingExposure = 1.1;
        this.renderer.outputEncoding = THREE.sRGBEncoding;
        this.renderer.physicallyCorrectLights = true;
        this.container.appendChild(this.renderer.domElement);

        this.scene = new THREE.Scene();

        // Gradient sky via vertex colors
        this.scene.background = new THREE.Color(0x78b9e8);
        this.scene.fog = new THREE.FogExp2(0x9cc8e8, 0.008);

        window.addEventListener('resize', () => {
            this.camera.aspect = window.innerWidth / window.innerHeight;
            this.camera.updateProjectionMatrix();
            this.renderer.setSize(window.innerWidth, window.innerHeight);
        });
    }

    // -------------------------------------------------------------------
    // Camera — cinematic FOV, fixed perspective
    // -------------------------------------------------------------------

    _initCamera() {
        this.camera = new THREE.PerspectiveCamera(
            45, window.innerWidth / window.innerHeight, 0.1, 300
        );
        this.camera.position.set(-4.85, 2.21, 0.41);
        this.camera.lookAt(15.00, 1.30, -0.60);
    }

    // -------------------------------------------------------------------
    // Lighting — 3-point + hemisphere for realistic PBR illumination
    // -------------------------------------------------------------------

    _initLights() {
        // Hemisphere for ambient sky/ground bounce
        const hemi = new THREE.HemisphereLight(0x8dc1f2, 0x3d6b2e, 0.6);
        this.scene.add(hemi);

        // Key light (warm sun)
        const sun = new THREE.DirectionalLight(0xfff0dd, 2.5);
        sun.position.set(25, 35, 15);
        sun.castShadow = true;
        sun.shadow.mapSize.set(2048, 2048);
        sun.shadow.camera.near = 0.5;
        sun.shadow.camera.far = 120;
        sun.shadow.camera.left = -60;
        sun.shadow.camera.right = 60;
        sun.shadow.camera.top = 40;
        sun.shadow.camera.bottom = -40;
        sun.shadow.bias = -0.0005;
        sun.shadow.normalBias = 0.02;
        this.scene.add(sun);

        // Fill light (cool, opposite side)
        const fill = new THREE.DirectionalLight(0x8aaed4, 0.8);
        fill.position.set(-20, 15, -10);
        this.scene.add(fill);

        // Rim/back light
        const rim = new THREE.DirectionalLight(0xffeebb, 0.5);
        rim.position.set(-10, 20, 25);
        this.scene.add(rim);
    }

    // -------------------------------------------------------------------
    // Environment — PBR ground, realistic trees, grass patches, sky dome
    // -------------------------------------------------------------------

    _buildEnvironment() {
        // Sky dome (gradient sphere)
        const skyGeo = new THREE.SphereGeometry(150, 32, 16);
        const skyMat = new THREE.ShaderMaterial({
            uniforms: {
                topColor: { value: new THREE.Color(0x4488cc) },
                bottomColor: { value: new THREE.Color(0xc8dff0) },
                offset: { value: 20 },
                exponent: { value: 0.5 },
            },
            vertexShader: `
                varying vec3 vWorldPosition;
                void main() {
                    vec4 worldPos = modelMatrix * vec4(position, 1.0);
                    vWorldPosition = worldPos.xyz;
                    gl_Position = projectionMatrix * modelViewMatrix * vec4(position, 1.0);
                }
            `,
            fragmentShader: `
                uniform vec3 topColor;
                uniform vec3 bottomColor;
                uniform float offset;
                uniform float exponent;
                varying vec3 vWorldPosition;
                void main() {
                    float h = normalize(vWorldPosition + offset).y;
                    gl_FragColor = vec4(mix(bottomColor, topColor, max(pow(max(h, 0.0), exponent), 0.0)), 1.0);
                }
            `,
            side: THREE.BackSide,
        });
        this.scene.add(new THREE.Mesh(skyGeo, skyMat));

        // Ground — PBR with vertex color variation for realism
        const groundGeo = new THREE.PlaneGeometry(150, 100, 120, 80);
        groundGeo.rotateX(-Math.PI / 2);
        const gv = groundGeo.attributes.position;
        // Vertex color for ground variation
        const colors = new Float32Array(gv.count * 3);
        for (let i = 0; i < gv.count; i++) {
            const x = gv.getX(i), z = gv.getZ(i);
            gv.setY(i,
                Math.sin(x * 0.15) * 0.15 +
                Math.cos(z * 0.2) * 0.1 +
                Math.sin(x * 0.5 + z * 0.4) * 0.06 +
                Math.sin(x * 1.2 + z * 0.8) * 0.02 +
                (Math.random() - 0.5) * 0.02
            );
            // Vary green tones across the ground
            const noise = Math.sin(x * 0.3 + z * 0.2) * 0.08 + Math.random() * 0.04;
            colors[i * 3]     = 0.28 + noise * 0.5;  // R
            colors[i * 3 + 1] = 0.52 + noise;        // G
            colors[i * 3 + 2] = 0.22 + noise * 0.3;  // B
        }
        groundGeo.setAttribute('color', new THREE.BufferAttribute(colors, 3));
        groundGeo.computeVertexNormals();
        const groundMat = new THREE.MeshStandardMaterial({
            vertexColors: true,
            roughness: 0.92,
            metalness: 0.0,
        });
        const ground = new THREE.Mesh(groundGeo, groundMat);
        ground.receiveShadow = true;
        this.scene.add(ground);

        // Trees
        this._addTrees();

        // Fences
        this._addFences();

        // Distance markers
        this._addDistanceMarkers();

        // Rocks scattered around
        this._addRocks();
    }

    _addTrees() {
        this.treeFoliage = []; // track for wind sway

        const barkMat = new THREE.MeshStandardMaterial({
            color: 0x4a3019, roughness: 0.95, metalness: 0.0
        });
        const barkLightMat = new THREE.MeshStandardMaterial({
            color: 0x6b4c2a, roughness: 0.9, metalness: 0.0
        });
        const leafMats = [
            new THREE.MeshStandardMaterial({ color: 0x2a6b2a, roughness: 0.75, metalness: 0 }),
            new THREE.MeshStandardMaterial({ color: 0x1d5c1d, roughness: 0.8, metalness: 0 }),
            new THREE.MeshStandardMaterial({ color: 0x357a35, roughness: 0.7, metalness: 0 }),
            new THREE.MeshStandardMaterial({ color: 0x4a9040, roughness: 0.7, metalness: 0 }),
        ];

        const treePositions = [
            [-8, 0, -15], [-5, 0, -18], [10, 0, -22], [25, 0, -24],
            [40, 0, -20], [55, 0, -22], [60, 0, -16],
            [-8, 0, 16], [-3, 0, 22], [15, 0, 24], [30, 0, 20],
            [45, 0, 22], [55, 0, 18], [65, 0, 14],
            [-12, 0, -8], [-12, 0, 8], [72, 0, -6], [72, 0, 6],
        ];

        treePositions.forEach((pos, idx) => {
            const group = new THREE.Group();
            const treeType = idx % 3; // vary tree shapes
            const h = 3.5 + Math.random() * 4.0;
            const tMat = idx % 2 === 0 ? barkMat : barkLightMat;

            // Trunk — tapered cylinder with slight bend
            const trunkGeo = new THREE.CylinderGeometry(0.08, 0.22, h * 0.5, 10);
            // Slight organic bend
            const tv = trunkGeo.attributes.position;
            for (let i = 0; i < tv.count; i++) {
                const y = tv.getY(i);
                const bendFactor = (y / (h * 0.5)) * 0.15;
                tv.setX(i, tv.getX(i) + bendFactor * Math.sin(idx));
                tv.setZ(i, tv.getZ(i) + bendFactor * Math.cos(idx));
            }
            trunkGeo.computeVertexNormals();
            const trunk = new THREE.Mesh(trunkGeo, tMat);
            trunk.position.y = h * 0.25;
            trunk.castShadow = true;
            trunk.receiveShadow = true;
            group.add(trunk);

            // Branches (small cylinders sprouting from trunk)
            for (let b = 0; b < 3; b++) {
                const branchLen = 0.4 + Math.random() * 0.6;
                const branch = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.02, 0.04, branchLen, 5),
                    tMat
                );
                const branchY = h * 0.25 + h * 0.1 * (b + 1);
                branch.position.set(0, branchY, 0);
                branch.rotation.z = (Math.random() - 0.5) * 1.2 + (b % 2 === 0 ? 0.6 : -0.6);
                branch.rotation.y = Math.random() * Math.PI * 2;
                branch.castShadow = true;
                group.add(branch);
            }

            // Foliage canopy — layered icosahedrons for organic look
            const mat = leafMats[Math.floor(Math.random() * leafMats.length)];
            const canopyGroup = new THREE.Group();

            if (treeType === 0) {
                // Round deciduous tree
                for (let j = 0; j < 6; j++) {
                    const r = (1.2 - j * 0.12) + Math.random() * 0.4;
                    const foliageGeo = new THREE.IcosahedronGeometry(r * 0.5, 1);
                    // Randomize vertices for organic shape
                    const fv = foliageGeo.attributes.position;
                    for (let k = 0; k < fv.count; k++) {
                        fv.setX(k, fv.getX(k) + (Math.random() - 0.5) * 0.15);
                        fv.setY(k, fv.getY(k) + (Math.random() - 0.5) * 0.1);
                        fv.setZ(k, fv.getZ(k) + (Math.random() - 0.5) * 0.15);
                    }
                    foliageGeo.computeVertexNormals();
                    const foliage = new THREE.Mesh(foliageGeo, mat);
                    const angle = (j / 6) * Math.PI * 2;
                    foliage.position.set(
                        Math.cos(angle) * 0.3 * (j < 3 ? 1 : 0.5),
                        h * 0.42 + j * h * 0.08,
                        Math.sin(angle) * 0.3 * (j < 3 ? 1 : 0.5)
                    );
                    foliage.scale.set(1, 0.75 + Math.random() * 0.25, 1);
                    foliage.castShadow = true;
                    canopyGroup.add(foliage);
                }
            } else if (treeType === 1) {
                // Pine / conical tree
                for (let j = 0; j < 5; j++) {
                    const r = (1.6 - j * 0.3);
                    const coneGeo = new THREE.ConeGeometry(r * 0.45, h * 0.18, 8);
                    const fv = coneGeo.attributes.position;
                    for (let k = 0; k < fv.count; k++) {
                        fv.setX(k, fv.getX(k) + (Math.random() - 0.5) * 0.08);
                        fv.setZ(k, fv.getZ(k) + (Math.random() - 0.5) * 0.08);
                    }
                    coneGeo.computeVertexNormals();
                    const cone = new THREE.Mesh(coneGeo, mat);
                    cone.position.y = h * 0.3 + j * h * 0.12;
                    cone.castShadow = true;
                    canopyGroup.add(cone);
                }
            } else {
                // Bushy / oak-like with wide canopy
                for (let j = 0; j < 8; j++) {
                    const r = (0.6 + Math.random() * 0.5);
                    const foliageGeo = new THREE.DodecahedronGeometry(r * 0.45, 1);
                    const fv = foliageGeo.attributes.position;
                    for (let k = 0; k < fv.count; k++) {
                        fv.setX(k, fv.getX(k) + (Math.random() - 0.5) * 0.12);
                        fv.setY(k, fv.getY(k) + (Math.random() - 0.5) * 0.08);
                        fv.setZ(k, fv.getZ(k) + (Math.random() - 0.5) * 0.12);
                    }
                    foliageGeo.computeVertexNormals();
                    const foliage = new THREE.Mesh(foliageGeo, mat);
                    const angle = (j / 8) * Math.PI * 2;
                    const spread = 0.4 + Math.random() * 0.3;
                    foliage.position.set(
                        Math.cos(angle) * spread,
                        h * 0.38 + (j % 3) * h * 0.1,
                        Math.sin(angle) * spread
                    );
                    foliage.castShadow = true;
                    canopyGroup.add(foliage);
                }
            }

            group.add(canopyGroup);
            this.treeFoliage.push({ canopy: canopyGroup, baseY: 0, seed: idx * 1.7 });

            group.position.set(pos[0], pos[1], pos[2]);
            this.scene.add(group);
        });
    }

    _addFences() {
        const fenceMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x8b6914, roughness: 0.8, metalness: 0.05
        });
        for (let side = -1; side <= 1; side += 2) {
            for (let x = -5; x < 65; x += 3.5) {
                const post = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.05, 0.06, 1.3, 6),
                    fenceMat
                );
                post.position.set(x, 0.65, side * 13);
                post.castShadow = true;
                post.receiveShadow = true;
                this.scene.add(post);

                if (x < 62) {
                    for (let rh of [0.45, 0.9]) {
                        const rail = new THREE.Mesh(
                            new THREE.BoxGeometry(3.5, 0.07, 0.05),
                            fenceMat
                        );
                        rail.position.set(x + 1.75, rh, side * 13);
                        rail.castShadow = true;
                        this.scene.add(rail);
                    }
                }
            }
        }
    }

    _addDistanceMarkers() {
        const markerMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0x888866, transparent: true, opacity: 0.4, roughness: 1
        });
        [10, 20, 30, 40, 50].forEach(d => {
            const marker = new THREE.Mesh(
                new THREE.PlaneGeometry(0.5, 2.5),
                markerMat
            );
            marker.rotation.x = -Math.PI / 2;
            marker.position.set(d, 0.02, 0);
            this.scene.add(marker);
        });
    }

    _addRocks() {
        const rockMats = [
            new THREE.MeshStandardMaterial({ color: 0x6a6a6a, roughness: 0.92, metalness: 0.08 }),
            new THREE.MeshStandardMaterial({ color: 0x7a7a72, roughness: 0.88, metalness: 0.05 }),
            new THREE.MeshStandardMaterial({ color: 0x5c5c58, roughness: 0.95, metalness: 0.1 }),
        ];
        const rockPositions = [
            [5, 0, -6], [12, 0, 8], [35, 0, -10], [50, 0, 7],
            [-3, 0, -4], [60, 0, -3], [8, 0, 11], [42, 0, 9],
            [18, 0, -7], [28, 0, 5], [55, 0, -8],
        ];
        rockPositions.forEach((pos, idx) => {
            const rockGroup = new THREE.Group();
            const mat = rockMats[idx % rockMats.length];

            // Main rock — organic dodecahedron with displaced vertices
            const s = 0.15 + Math.random() * 0.4;
            const geo = new THREE.DodecahedronGeometry(s, 2);
            const rv = geo.attributes.position;
            for (let i = 0; i < rv.count; i++) {
                rv.setX(i, rv.getX(i) + (Math.random() - 0.5) * s * 0.2);
                rv.setY(i, rv.getY(i) + (Math.random() - 0.5) * s * 0.15);
                rv.setZ(i, rv.getZ(i) + (Math.random() - 0.5) * s * 0.2);
            }
            geo.computeVertexNormals();
            const rock = new THREE.Mesh(geo, mat);
            rock.scale.set(1 + Math.random() * 0.4, 0.4 + Math.random() * 0.4, 1 + Math.random() * 0.3);
            rock.rotation.set(Math.random() * 0.3, Math.random() * Math.PI, Math.random() * 0.3);
            rock.castShadow = true;
            rock.receiveShadow = true;
            rockGroup.add(rock);

            // Small pebbles around big rock
            for (let p = 0; p < 2 + Math.floor(Math.random() * 3); p++) {
                const ps = 0.03 + Math.random() * 0.08;
                const pebble = new THREE.Mesh(
                    new THREE.DodecahedronGeometry(ps, 1),
                    mat
                );
                pebble.position.set(
                    (Math.random() - 0.5) * s * 2,
                    ps * 0.3,
                    (Math.random() - 0.5) * s * 2
                );
                pebble.scale.y = 0.4 + Math.random() * 0.3;
                pebble.rotation.set(Math.random(), Math.random(), Math.random());
                pebble.castShadow = true;
                rockGroup.add(pebble);
            }

            rockGroup.position.set(pos[0], s * 0.25, pos[2]);
            this.scene.add(rockGroup);
        });
    }

    // -------------------------------------------------------------------
    // GLB Props — archery training grounds, sign, bow, stairs, arrow
    // -------------------------------------------------------------------

    _loadGLBProps() {
        const loader = new THREE.GLTFLoader();
        const base = '/static/models/objects/';

        // Helper: load GLB, auto-compute bounding box, then position/scale
        const loadGLB = (url, onLoaded) => {
            loader.load(url, (gltf) => {
                const model = gltf.scene;
                // Compute actual bounding box so we can scale correctly
                const box = new THREE.Box3().setFromObject(model);
                const size = new THREE.Vector3();
                box.getSize(size);
                console.log(url, 'raw size:', size.x.toFixed(3), size.y.toFixed(3), size.z.toFixed(3));
                onLoaded(model, box, size);
            }, undefined, (err) => {
                console.warn('GLB load failed:', url, err);
            });
        };

        // --- Archery Training Grounds ---
        // Place far behind targets as a decorative backdrop, out of camera's direct field
        loadGLB(base + 'Archery Training Grounds.glb', (model, box, size) => {
            // Scale so it's about 3m tall
            const targetHeight = 3.0;
            const s = targetHeight / Math.max(size.y, 0.001);
            model.scale.set(s, s, s);
            model.position.set(8, 0, -10);  // behind and to the side
            model.rotation.y = Math.PI * 0.5;
            model.traverse(child => {
                if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; }
            });
            this.scene.add(model);
            this._glbModels['trainingGrounds'] = model;
        });

        // --- Wooden Sign ---
        loadGLB(base + 'Wooden Sign.glb', (model, box, size) => {
            const targetHeight = 1.5;
            const s = targetHeight / Math.max(size.y, 0.001);
            model.scale.set(s, s, s);
            model.position.set(-3.5, 0, 3);
            model.rotation.y = Math.PI * 0.3;
            model.traverse(child => {
                if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; }
            });
            this.scene.add(model);
            this._glbModels['sign'] = model;
        });

        // Wooden Bow is now attached to the character's hand bone in _attachBowToHand()
        // No static ground bow needed

        // --- Stairs ---
        loadGLB(base + 'Stairs.glb', (model, box, size) => {
            // Stairs are already ~2.3m, keep roughly that size
            const targetHeight = 2.0;
            const s = targetHeight / Math.max(size.y, 0.001);
            model.scale.set(s, s, s);
            model.position.set(-7, 0, -4);
            model.rotation.y = Math.PI * 0.25;
            model.traverse(child => {
                if (child.isMesh) { child.castShadow = true; child.receiveShadow = true; }
            });
            this.scene.add(model);
            this._glbModels['stairs'] = model;
        });

        // --- Arrow model (cached for fireArrow) ---
        loadGLB(base + 'Arrow.glb', (model, box, size) => {
            // Arrow should be ~0.7m long
            const targetLen = 0.7;
            const s = targetLen / Math.max(size.y, size.z, size.x, 0.001);
            model.scale.set(s, s, s);
            model.traverse(child => {
                if (child.isMesh) { child.castShadow = true; }
            });
            this._glbArrowTemplate = model;
        });
    }

    // -------------------------------------------------------------------
    // Archer — smooth PBR character with anatomical proportions
    // -------------------------------------------------------------------

    _buildArcher() {
        // Prepare the bowGroup placeholder so any legacy reference to this.bowGroup doesn't crash
        this.bowGroup = new THREE.Group();
        this.scene.add(this.bowGroup);

        const loader = new THREE.FBXLoader();

        // Load Character
        loader.load('/static/models/X Bot.fbx', (object) => {
            this.archerGroup = object;

            object.scale.set(0.012, 0.012, 0.012);
            object.position.set(0, 0, 0);
            object.rotation.y = Math.PI * 0.47;   // face straight towards targets

            // Cast shadows
            object.traverse((child) => {
                if (child.isMesh) {
                    child.castShadow = true;
                    child.receiveShadow = true;
                    if (child.material) {
                        child.material.roughness = 0.5;
                        child.material.metalness = 0.1;
                    }
                }
            });

            this.scene.add(object);

            // Find hand bone for bow attachment
            this._findHandBone(object);

            // Setup Animations
            this.mixer = new THREE.AnimationMixer(object);
            this.actions = {};

            // Load Idle Animation
            loader.load('/static/models/standing idle 01.fbx', (anim) => {
                const idleAction = this.mixer.clipAction(anim.animations[0]);
                idleAction.play();
                this.actions['idle'] = idleAction;
                this.activeAction = idleAction;
            });

            // Load Firing Animations (Pre-load for later)
            loader.load('/static/models/standing draw arrow.fbx', (anim) => {
                const drawAction = this.mixer.clipAction(anim.animations[0]);
                drawAction.clampWhenFinished = true;
                this.actions['draw'] = drawAction;
            });

            loader.load('/static/models/standing aim overdraw.fbx', (anim) => {
                const overdrawAction = this.mixer.clipAction(anim.animations[0]);
                this.actions['overdraw'] = overdrawAction;
            });

        }, undefined, (error) => {
            console.error("Error loading FBX:", error);
        });
    }

    _findHandBone(skeleton) {
        // Traverse the skeleton to find the left hand bone (bow hand for archery)
        // Mixamo skeletons use names like "mixamorigLeftHand"
        const handBoneNames = [
            'mixamorigLeftHand', 'LeftHand', 'mixamorig:LeftHand',
            'mixamorigRightHand', 'RightHand', 'mixamorig:RightHand',
        ];
        let handBone = null;
        skeleton.traverse((child) => {
            if (child.isBone) {
                const name = child.name;
                // Prefer left hand (bow hand in archery)
                if (name.toLowerCase().includes('lefthand') && !name.toLowerCase().includes('thumb') && !name.toLowerCase().includes('index')) {
                    handBone = child;
                }
            }
        });
        // Log all bones for debugging
        skeleton.traverse((child) => {
            if (child.isBone) console.log('Bone:', child.name);
        });

        if (handBone) {
            console.log('Found hand bone:', handBone.name);
            this._handBone = handBone;
            this._attachBowToHand();
        } else {
            console.warn('Could not find hand bone in skeleton');
        }
    }

    _attachBowToHand() {
        if (!this._handBone) return;

        const gltfLoader = new THREE.GLTFLoader();
        gltfLoader.load('/static/models/objects/Wooden Bow.glb', (gltf) => {
            const bowModel = gltf.scene;
            const box = new THREE.Box3().setFromObject(bowModel);
            const size = new THREE.Vector3();
            box.getSize(size);

            // Scale bow to ~60cm (0.6 units) to fit hand — the FBX is scaled 0.012
            // so we need the bow in the skeleton's local space (much larger coords)
            const targetHeight = 50; // in skeleton space (0.012 scale → 50 * 0.012 = 0.6m)
            const s = targetHeight / Math.max(size.y, 0.001);
            bowModel.scale.set(s, s, s);

            // Rotate/offset to sit in the hand naturally
            bowModel.rotation.set(Math.PI * 0.5, 0, Math.PI * 0.5);
            bowModel.position.set(0, 0, 0);

            bowModel.traverse(child => {
                if (child.isMesh) child.castShadow = true;
            });

            this._handBone.add(bowModel);
            this._handBowModel = bowModel;
            console.log('Bow attached to hand bone');

            // Remove the static ground bow if it was loaded
            if (this._glbModels['bow']) {
                this.scene.remove(this._glbModels['bow']);
                delete this._glbModels['bow'];
            }
        });
    }

    // -------------------------------------------------------------------
    // Targets
    // -------------------------------------------------------------------

    buildTargets(sceneData) {
        Object.values(this.targetMeshes).forEach(g => this.scene.remove(g));
        Object.values(this.highlightRings).forEach(r => this.scene.remove(r));
        this.targetMeshes = {};
        this.highlightRings = {};
        this.targets = {};
        this.flags = {};

        sceneData.objects.forEach(obj => {
            this.targets[obj.id] = obj;
            const group = this._createTargetMesh(obj);
            group.position.set(obj.position[0], obj.position[1], obj.position[2]);
            this.scene.add(group);
            this.targetMeshes[obj.id] = group;

            const ringGeo = new THREE.RingGeometry(0.7, 0.85, 32);
            const ringMat = new THREE.MeshBasicMaterial({ flatShading: true,
                color: 0xffff00, side: THREE.DoubleSide, transparent: true, opacity: 0,
            });
            const ring = new THREE.Mesh(ringGeo, ringMat);
            ring.rotation.x = -Math.PI / 2;
            ring.position.set(obj.position[0], 0.05, obj.position[2]);
            this.scene.add(ring);
            this.highlightRings[obj.id] = ring;
        });
    }

    _createTargetMesh(obj) {
        const shape = obj.shape || 'barrel';
        const severity = obj.severity || 1;
        const builders = {
            barrel:    () => this._buildBarrel(obj, severity),
            crate:     () => this._buildCrate(obj, severity),
            scarecrow: () => this._buildScarecrow(obj, severity),
            bottle:    () => this._buildBottle(obj, severity),
            lantern:   () => this._buildLantern(obj, severity),
        };
        return (builders[shape] || builders.barrel)();
    }

    // Severity badge — colored ring on ground showing difficulty
    _addSeverityIndicator(group, severity, yBase) {
        const colors = [0x4ade80, 0xa3e635, 0xfacc15, 0xf97316, 0xef4444]; // green→red
        const color = colors[Math.min(severity - 1, 4)];
        const ring = new THREE.Mesh(
            new THREE.RingGeometry(0.5, 0.6, 24),
            new THREE.MeshBasicMaterial({ color, side: THREE.DoubleSide, transparent: true, opacity: 0.5 })
        );
        ring.rotation.x = -Math.PI / 2;
        ring.position.y = yBase + 0.02;
        group.add(ring);
    }

    // --- BARREL (Severity 1 — easy, close, big) ---
    _buildBarrel(obj, severity) {
        const group = new THREE.Group();
        const woodMat = new THREE.MeshStandardMaterial({ color: 0x8b5e3c, roughness: 0.85, metalness: 0.02 });
        const bandMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.4, metalness: 0.6 });

        // Barrel body
        const bodyGeo = new THREE.CylinderGeometry(0.35, 0.38, 0.9, 16);
        const bv = bodyGeo.attributes.position;
        for (let i = 0; i < bv.count; i++) {
            const y = bv.getY(i);
            const bulge = 1 + 0.08 * Math.cos(y * 3.5);
            bv.setX(i, bv.getX(i) * bulge);
            bv.setZ(i, bv.getZ(i) * bulge);
        }
        bodyGeo.computeVertexNormals();
        const body = new THREE.Mesh(bodyGeo, woodMat);
        body.position.y = 0.45;
        body.castShadow = true;
        body.receiveShadow = true;
        group.add(body);

        // Metal bands
        [-0.25, 0, 0.25].forEach(yOff => {
            const band = new THREE.Mesh(
                new THREE.TorusGeometry(0.38, 0.015, 8, 24), bandMat
            );
            band.rotation.x = Math.PI / 2;
            band.position.y = 0.45 + yOff;
            group.add(band);
        });

        // Lid
        const lid = new THREE.Mesh(
            new THREE.CylinderGeometry(0.36, 0.36, 0.04, 16),
            new THREE.MeshStandardMaterial({ color: 0x7a5030, roughness: 0.8 })
        );
        lid.position.y = 0.92;
        group.add(lid);

        // Flag
        this._addFlag(group, obj, 1.2);
        this._addSeverityIndicator(group, severity, 0);
        return group;
    }

    // --- CRATE (Severity 2 — wooden box on the ground) ---
    _buildCrate(obj, severity) {
        const group = new THREE.Group();
        const crateMat = new THREE.MeshStandardMaterial({ color: 0x9e7c4e, roughness: 0.9, metalness: 0.0 });
        const plankMat = new THREE.MeshStandardMaterial({ color: 0x846838, roughness: 0.85 });

        // Main box
        const box = new THREE.Mesh(new THREE.BoxGeometry(0.7, 0.7, 0.7), crateMat);
        box.position.y = 0.35;
        box.castShadow = true;
        box.receiveShadow = true;
        group.add(box);

        // Cross planks on front
        for (let d = -1; d <= 1; d += 2) {
            const plank = new THREE.Mesh(new THREE.BoxGeometry(0.06, 0.8, 0.02), plankMat);
            plank.position.set(0, 0.35, 0.36);
            plank.rotation.z = d * 0.6;
            group.add(plank);
        }

        // Corner edges
        [[-1,-1],[1,-1],[1,1],[-1,1]].forEach(([sx,sz]) => {
            const edge = new THREE.Mesh(
                new THREE.BoxGeometry(0.04, 0.72, 0.04), plankMat
            );
            edge.position.set(sx * 0.34, 0.35, sz * 0.34);
            group.add(edge);
        });

        // Stacked second crate (offset)
        const box2 = new THREE.Mesh(new THREE.BoxGeometry(0.55, 0.5, 0.55), crateMat);
        box2.position.set(0.05, 0.95, -0.05);
        box2.rotation.y = 0.3;
        box2.castShadow = true;
        group.add(box2);

        this._addFlag(group, obj, 1.4);
        this._addSeverityIndicator(group, severity, 0);
        return group;
    }

    // --- SCARECROW (Severity 3 — moving, medium distance) ---
    _buildScarecrow(obj, severity) {
        const group = new THREE.Group();
        const clothMat = new THREE.MeshStandardMaterial({ color: 0x8b6f47, roughness: 0.9 });
        const darkCloth = new THREE.MeshStandardMaterial({ color: 0x4a3828, roughness: 0.85 });
        const stickMat = new THREE.MeshStandardMaterial({ color: 0x5c3a1e, roughness: 0.9 });
        const skinMat = new THREE.MeshStandardMaterial({ color: 0xd4b896, roughness: 0.8 });

        // Main post
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.04, 0.05, 2.2, 6), stickMat);
        post.position.y = 1.1;
        post.castShadow = true;
        group.add(post);

        // Cross beam (arms)
        const arms = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.03, 1.4, 5), stickMat);
        arms.rotation.z = Math.PI / 2;
        arms.position.y = 1.7;
        arms.castShadow = true;
        group.add(arms);

        // Head (burlap sack)
        const head = new THREE.Mesh(new THREE.SphereGeometry(0.18, 10, 8), skinMat);
        head.position.y = 2.15;
        head.scale.y = 1.15;
        head.castShadow = true;
        group.add(head);

        // Hat
        const hatBrim = new THREE.Mesh(new THREE.CylinderGeometry(0.28, 0.28, 0.03, 12), darkCloth);
        hatBrim.position.y = 2.32;
        group.add(hatBrim);
        const hatTop = new THREE.Mesh(new THREE.CylinderGeometry(0.14, 0.16, 0.2, 8), darkCloth);
        hatTop.position.y = 2.43;
        group.add(hatTop);

        // Body / shirt (tapered cylinder)
        const shirt = new THREE.Mesh(
            new THREE.CylinderGeometry(0.2, 0.28, 0.7, 8), clothMat
        );
        shirt.position.y = 1.35;
        shirt.castShadow = true;
        group.add(shirt);

        // Straw poking out at sleeves
        for (let side = -1; side <= 1; side += 2) {
            for (let s = 0; s < 3; s++) {
                const straw = new THREE.Mesh(
                    new THREE.CylinderGeometry(0.008, 0.005, 0.15, 4),
                    new THREE.MeshStandardMaterial({ color: 0xd4b86a, roughness: 0.9 })
                );
                straw.position.set(side * 0.7, 1.7 + (s - 1) * 0.04, (Math.random() - 0.5) * 0.06);
                straw.rotation.z = side * (0.3 + Math.random() * 0.5);
                group.add(straw);
            }
        }

        this._addFlag(group, obj, 2.6);
        this._addSeverityIndicator(group, severity, 0);
        return group;
    }

    // --- BOTTLE (Severity 4 — small, moving, far) ---
    _buildBottle(obj, severity) {
        const group = new THREE.Group();
        const glassMat = new THREE.MeshStandardMaterial({
            color: 0x2d5a27, roughness: 0.15, metalness: 0.1, transparent: true, opacity: 0.85
        });

        // Wooden post to hold the bottle up
        const postMat = new THREE.MeshStandardMaterial({ color: 0x6b4c2a, roughness: 0.9 });
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.03, 0.04, 1.2, 6), postMat);
        post.position.y = 0.6;
        post.castShadow = true;
        group.add(post);

        // Small shelf
        const shelf = new THREE.Mesh(new THREE.BoxGeometry(0.4, 0.03, 0.15), postMat);
        shelf.position.y = 1.2;
        group.add(shelf);

        // Bottle body
        const bodyGeo = new THREE.CylinderGeometry(0.08, 0.1, 0.35, 12);
        const body = new THREE.Mesh(bodyGeo, glassMat);
        body.position.y = 1.4;
        body.castShadow = true;
        group.add(body);

        // Bottle neck
        const neck = new THREE.Mesh(
            new THREE.CylinderGeometry(0.03, 0.06, 0.15, 10), glassMat
        );
        neck.position.y = 1.65;
        group.add(neck);

        // Bottle lip
        const lip = new THREE.Mesh(
            new THREE.TorusGeometry(0.035, 0.008, 6, 12), glassMat
        );
        lip.rotation.x = Math.PI / 2;
        lip.position.y = 1.73;
        group.add(lip);

        // Second bottle slightly offset
        const body2 = new THREE.Mesh(
            new THREE.CylinderGeometry(0.07, 0.09, 0.3, 12),
            new THREE.MeshStandardMaterial({ color: 0x6b3a1a, roughness: 0.2, metalness: 0.05, transparent: true, opacity: 0.8 })
        );
        body2.position.set(0.12, 1.37, 0);
        group.add(body2);
        const neck2 = new THREE.Mesh(
            new THREE.CylinderGeometry(0.025, 0.05, 0.12, 8),
            body2.material
        );
        neck2.position.set(0.12, 1.58, 0);
        group.add(neck2);

        this._addFlag(group, obj, 1.9);
        this._addSeverityIndicator(group, severity, 0);
        return group;
    }

    // --- LANTERN (Severity 5 — tiny, far, hardest) ---
    _buildLantern(obj, severity) {
        const group = new THREE.Group();
        const metalMat = new THREE.MeshStandardMaterial({ color: 0x3a3a3a, roughness: 0.4, metalness: 0.7 });
        const glassMat = new THREE.MeshStandardMaterial({
            color: 0xffdd88, roughness: 0.2, metalness: 0.0, transparent: true, opacity: 0.6,
            emissive: 0xffaa44, emissiveIntensity: 0.3
        });

        // Tall post
        const postMat = new THREE.MeshStandardMaterial({ color: 0x555555, roughness: 0.5, metalness: 0.5 });
        const post = new THREE.Mesh(new THREE.CylinderGeometry(0.025, 0.035, 1.8, 6), postMat);
        post.position.y = 0.9;
        post.castShadow = true;
        group.add(post);

        // Hook at top
        const hook = new THREE.Mesh(
            new THREE.TorusGeometry(0.06, 0.01, 6, 8, Math.PI),
            metalMat
        );
        hook.position.y = 1.85;
        hook.rotation.x = Math.PI;
        group.add(hook);

        // Lantern body frame
        const lanternY = 1.7;
        // Top cap
        const cap = new THREE.Mesh(new THREE.ConeGeometry(0.12, 0.08, 6), metalMat);
        cap.position.y = lanternY + 0.2;
        group.add(cap);

        // Bottom plate
        const plate = new THREE.Mesh(new THREE.CylinderGeometry(0.1, 0.1, 0.02, 6), metalMat);
        plate.position.y = lanternY - 0.15;
        group.add(plate);

        // Glass body (glowing)
        const glassBody = new THREE.Mesh(
            new THREE.CylinderGeometry(0.09, 0.09, 0.3, 8), glassMat
        );
        glassBody.position.y = lanternY + 0.02;
        group.add(glassBody);

        // Metal frame bars
        for (let i = 0; i < 4; i++) {
            const angle = (i / 4) * Math.PI * 2;
            const bar = new THREE.Mesh(
                new THREE.CylinderGeometry(0.008, 0.008, 0.34, 4), metalMat
            );
            bar.position.set(Math.cos(angle) * 0.1, lanternY + 0.02, Math.sin(angle) * 0.1);
            group.add(bar);
        }

        // Point light for glow effect
        const glow = new THREE.PointLight(0xffaa44, 0.5, 3);
        glow.position.y = lanternY;
        group.add(glow);

        this._addFlag(group, obj, 2.1);
        this._addSeverityIndicator(group, severity, 0);
        return group;
    }

    // --- Shared: color flag on pole ---
    _addFlag(group, obj, yPos) {
        const colorMap = {
            red: 0xcc2222, blue: 0x2244cc, yellow: 0xccaa11,
            green: 0x22aa33, white: 0xdddddd,
        };
        const mainColor = colorMap[obj.flag_color] || 0xcc2222;
        const poleMat = new THREE.MeshStandardMaterial({ color: 0x4a2e10, roughness: 0.9 });

        const pole = new THREE.Mesh(new THREE.CylinderGeometry(0.01, 0.01, 0.5, 5), poleMat);
        pole.position.y = yPos;
        group.add(pole);

        const flagGeo = new THREE.PlaneGeometry(0.25, 0.14, 5, 2);
        const fv = flagGeo.attributes.position;
        for (let i = 0; i < fv.count; i++) {
            fv.setZ(i, Math.sin(fv.getX(i) * 10) * 0.015);
        }
        flagGeo.computeVertexNormals();
        const flag = new THREE.Mesh(flagGeo, new THREE.MeshStandardMaterial({
            color: mainColor, side: THREE.DoubleSide, roughness: 0.5
        }));
        flag.position.set(0.14, yPos + 0.18, 0);
        group.add(flag);
        this.flags[obj.id] = flag;
    }

    // -------------------------------------------------------------------
    // Target selection via raycasting
    // -------------------------------------------------------------------

    pickTarget(mouseEvent) {
        // Returns target id nearest to click, or null
        const rect = this.renderer.domElement.getBoundingClientRect();
        const mouse = new THREE.Vector2(
            ((mouseEvent.clientX - rect.left) / rect.width) * 2 - 1,
            -((mouseEvent.clientY - rect.top) / rect.height) * 2 + 1
        );
        const raycaster = new THREE.Raycaster();
        raycaster.setFromCamera(mouse, this.camera);

        // Collect all meshes from target groups
        const meshes = [];
        Object.keys(this.targetMeshes).forEach(id => {
            this.targetMeshes[id].traverse(child => {
                if (child.isMesh) {
                    child._targetId = id;
                    meshes.push(child);
                }
            });
        });

        const hits = raycaster.intersectObjects(meshes, false);
        if (hits.length > 0) {
            return hits[0].object._targetId;
        }

        // Fallback: find closest target to ray
        let bestId = null;
        let bestDist = 5.0; // max pick distance in world units
        const ray = raycaster.ray;
        Object.keys(this.targets).forEach(id => {
            const pos = this.targets[id].position;
            const tPos = new THREE.Vector3(pos[0], pos[1], pos[2]);
            const dist = ray.distanceToPoint(tPos);
            if (dist < bestDist) {
                bestDist = dist;
                bestId = id;
            }
        });
        return bestId;
    }

    selectTarget(targetId) {
        // Visual feedback for selected target
        this._selectedTargetId = targetId;
        // Pulse the highlight ring
        this.highlightTarget(targetId);
    }

    getSelectedTargetId() {
        return this._selectedTargetId || null;
    }

    // -------------------------------------------------------------------
    // Aim Mode — camera zoom + draw animation
    // -------------------------------------------------------------------

    enterAimMode() {
        if (this.isAiming || this.animatingArrow) return;
        this.isAiming = true;
        this.aimStartTime = performance.now();
        this.aimPower = 0;
        this._cameraLerpDir = 1; // zoom in

        // Reset mouse-look offset
        this._aimYaw = 0;
        this._aimPitch = 0;
        this._aimMouseStart = null;

        // Start tracking mouse movement for aiming
        this._onAimMouseMove = (e) => this._handleAimMouseMove(e);
        document.addEventListener('mousemove', this._onAimMouseMove);

        // Play draw animation
        if (this.actions && this.actions['draw']) {
            const draw = this.actions['draw'];
            draw.reset()
                .setEffectiveWeight(1)
                .setEffectiveTimeScale(0.6) // slow draw for dramatic effect
                .setLoop(THREE.LoopOnce, 1);
            draw.clampWhenFinished = true;
            draw.crossFadeFrom(this.activeAction, 0.25).play();
            this.activeAction = draw;
        }

        // Show crosshair + power bar
        document.getElementById('crosshair').className = 'show';
        document.getElementById('aim-power').className = 'show';
        document.body.classList.add('aiming');
    }

    exitAimMode() {
        if (!this.isAiming) return;
        this.isAiming = false;
        this._cameraLerpDir = -1; // zoom back out

        // Stop tracking mouse movement
        if (this._onAimMouseMove) {
            document.removeEventListener('mousemove', this._onAimMouseMove);
            this._onAimMouseMove = null;
        }

        // Reset aim offsets
        this._aimYaw = 0;
        this._aimPitch = 0;

        // Hide crosshair + power bar
        document.getElementById('crosshair').className = 'hidden';
        document.getElementById('aim-power').className = 'hidden';
        document.body.classList.remove('aiming');
    }

    _handleAimMouseMove(e) {
        if (!this.isAiming) return;

        if (!this._aimMouseStart) {
            this._aimMouseStart = { x: e.clientX, y: e.clientY };
            return;
        }

        // Mouse delta from aim start position
        const dx = e.clientX - this._aimMouseStart.x;
        const dy = e.clientY - this._aimMouseStart.y;

        // Sensitivity: pixels to radians (lower = more precise)
        const sensitivity = 0.003;
        const maxYaw = 0.6;    // max ~35 degrees horizontal
        const maxPitch = 0.35; // max ~20 degrees vertical

        this._aimYaw = Math.max(-maxYaw, Math.min(maxYaw, dx * sensitivity));
        this._aimPitch = Math.max(-maxPitch, Math.min(maxPitch, -dy * sensitivity)); // invert Y
    }

    getAimDirection() {
        // Returns the aim direction vector based on current mouse offset
        // Base direction: from aim camera position toward aim target
        const baseDir = new THREE.Vector3().subVectors(this._cameraAim.target, this._cameraAim.pos).normalize();

        // Apply yaw (horizontal) and pitch (vertical) rotations
        const euler = new THREE.Euler(this._aimPitch, -this._aimYaw, 0, 'YXZ');
        baseDir.applyEuler(euler);

        return baseDir;
    }

    _updateAimPower() {
        if (!this.isAiming) return;
        const elapsed = (performance.now() - this.aimStartTime) / 1000; // seconds
        // Power builds over 2.5 seconds, capped at 1
        this.aimPower = Math.min(elapsed / 2.5, 1.0);
        const fill = document.getElementById('power-fill');
        if (fill) fill.style.height = (this.aimPower * 100) + '%';
    }

    highlightTarget(targetId) {
        Object.values(this.highlightRings).forEach(r => { r.material.opacity = 0; });
        const ring = this.highlightRings[targetId];
        if (ring) {
            ring.material.opacity = 0.7;
            ring._pulseTime = 0;
        }
    }

    // -------------------------------------------------------------------
    // Arrow (fired projectile)
    // -------------------------------------------------------------------

    _createArrow() {
        // Use GLB arrow model if loaded, otherwise fall back to procedural
        if (this._glbArrowTemplate) {
            // Wrap in a Group so the inner rotation is preserved when
            // _launchArrowMesh sets quaternion on the outer group
            const wrapper = new THREE.Group();
            const clone = this._glbArrowTemplate.clone();
            // GLB arrow points along Y (vertical) — rotate so it aligns with +X (flight direction)
            // rotation.z = -PI/2 rotates Y-up to X-forward (tip pointing +X)
            clone.rotation.set(0, 0, -Math.PI / 2);
            wrapper.add(clone);
            return wrapper;
        }

        // Procedural fallback
        const group = new THREE.Group();
        const arrowMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0xc8a85c, roughness: 0.5 });
        const metalMat = new THREE.MeshStandardMaterial({ flatShading: true, color: 0x888888, roughness: 0.3, metalness: 0.6 });

        const shaft = new THREE.Mesh(new THREE.CylinderGeometry(0.015, 0.015, 0.65, 8), arrowMat);
        shaft.rotation.z = Math.PI / 2;
        group.add(shaft);

        const tip = new THREE.Mesh(new THREE.ConeGeometry(0.025, 0.08, 6), metalMat);
        tip.rotation.z = -Math.PI / 2;
        tip.position.x = 0.37;
        group.add(tip);

        const fletchMat = new THREE.MeshStandardMaterial({ flatShading: true,
            color: 0xcc3333, side: THREE.DoubleSide, roughness: 0.5
        });
        for (let i = 0; i < 3; i++) {
            const fletch = new THREE.Mesh(new THREE.PlaneGeometry(0.1, 0.04), fletchMat);
            fletch.position.x = -0.28;
            fletch.rotation.x = (i / 3) * Math.PI * 2;
            group.add(fletch);
        }
        return group;
    }

    fireArrow(trajectoryPoints, onComplete, fromAimMode) {
        if (this.animatingArrow) return;
        this.animatingArrow = true;

        if (fromAimMode) {
            // Already in draw pose from aim mode — launch immediately, then return to idle
            this._launchArrowMesh(trajectoryPoints, () => {
                if (this.actions && this.actions['idle']) {
                    this.actions['idle'].reset()
                        .setEffectiveWeight(1)
                        .crossFadeFrom(this.activeAction, 0.5)
                        .play();
                    this.activeAction = this.actions['idle'];
                }
                if (onComplete) onComplete();
            });
        } else if (this.actions && this.actions['draw']) {
            this.actions['draw'].reset()
                .setEffectiveWeight(1)
                .setEffectiveTimeScale(1.5)
                .crossFadeFrom(this.activeAction, 0.3)
                .play();
            this.activeAction = this.actions['draw'];

            setTimeout(() => {
                this._launchArrowMesh(trajectoryPoints, () => {
                    if (this.actions['idle']) {
                        this.actions['idle'].reset()
                            .setEffectiveWeight(1)
                            .crossFadeFrom(this.activeAction, 0.5)
                            .play();
                        this.activeAction = this.actions['idle'];
                    }
                    if (onComplete) onComplete();
                });
            }, 500);
        } else {
            this._launchArrowMesh(trajectoryPoints, onComplete);
        }
    }

    _launchArrowMesh(trajectoryPoints, onComplete) {
        if (this.arrowMesh) this.scene.remove(this.arrowMesh);
        if (this.trailLine) this.scene.remove(this.trailLine);
        // Clean up old trail dots
        if (this._trailDots) {
            this._trailDots.forEach(d => this.scene.remove(d));
        }
        this._trailDots = [];
        this.particles.forEach(p => this.scene.remove(p));
        this.particles = [];

        this.arrowMesh = this._createArrow();
        this.arrowMesh.position.set(
            trajectoryPoints[0][0], trajectoryPoints[0][1], trajectoryPoints[0][2]
        );
        this.scene.add(this.arrowMesh);

        // Black dotted chain trail — we place dot meshes every N points
        const DOT_INTERVAL = 3;         // place a dot every 3 trajectory points
        const MAX_VISIBLE_DOTS = 4;     // only keep last 4 dots visible (fade chain)

        const totalPoints = trajectoryPoints.length;
        const duration = Math.min(totalPoints * 12, 2500);
        let startTime = null;
        let currentIndex = 0;
        let dotCounter = 0;

        const animateStep = (timestamp) => {
            if (!startTime) startTime = timestamp;
            const elapsed = timestamp - startTime;
            const progress = Math.min(elapsed / duration, 1.0);
            const targetIndex = Math.min(Math.floor(progress * totalPoints), totalPoints - 1);

            // Place trail dots as arrow advances
            while (currentIndex <= targetIndex && currentIndex < totalPoints) {
                dotCounter++;
                if (dotCounter % DOT_INTERVAL === 0) {
                    const dpt = trajectoryPoints[currentIndex];
                    const dot = new THREE.Mesh(
                        new THREE.SphereGeometry(0.025, 6, 4),
                        new THREE.MeshBasicMaterial({ color: 0x111111, transparent: true, opacity: 0.9 })
                    );
                    dot.position.set(dpt[0], dpt[1], dpt[2]);
                    dot._birthTime = timestamp;
                    this.scene.add(dot);
                    this._trailDots.push(dot);
                }
                currentIndex++;
            }

            // Fade: only keep the last MAX_VISIBLE_DOTS fully opaque, older ones fade out
            const dotCount = this._trailDots.length;
            for (let i = 0; i < dotCount; i++) {
                const age = dotCount - 1 - i; // 0 = newest
                if (age < MAX_VISIBLE_DOTS) {
                    // Chain fade: newest = 0.9, oldest visible = 0.25
                    this._trailDots[i].material.opacity = 0.9 - (age / MAX_VISIBLE_DOTS) * 0.65;
                    this._trailDots[i].visible = true;
                } else {
                    this._trailDots[i].material.opacity = 0;
                    this._trailDots[i].visible = false;
                }
            }

            // Also connect the visible dots with thin black lines
            if (this.trailLine) this.scene.remove(this.trailLine);
            const visibleDots = this._trailDots.slice(-MAX_VISIBLE_DOTS);
            if (visibleDots.length >= 2) {
                const linePositions = [];
                visibleDots.forEach(d => linePositions.push(d.position.x, d.position.y, d.position.z));
                const lineGeo = new THREE.BufferGeometry();
                lineGeo.setAttribute('position', new THREE.Float32BufferAttribute(linePositions, 3));
                this.trailLine = new THREE.Line(lineGeo, new THREE.LineBasicMaterial({
                    color: 0x111111, transparent: true, opacity: 0.4,
                }));
                this.scene.add(this.trailLine);
            }

            const pt = trajectoryPoints[targetIndex];
            this.arrowMesh.position.set(pt[0], pt[1], pt[2]);

            if (targetIndex < totalPoints - 1) {
                const next = trajectoryPoints[Math.min(targetIndex + 1, totalPoints - 1)];
                const dir = new THREE.Vector3(
                    next[0] - pt[0], next[1] - pt[1], next[2] - pt[2]
                ).normalize();
                if (dir.length() > 0.001) {
                    const axis = new THREE.Vector3(1, 0, 0);
                    this.arrowMesh.quaternion.copy(
                        new THREE.Quaternion().setFromUnitVectors(axis, dir)
                    );
                }
            }

            if (progress < 1.0) {
                requestAnimationFrame(animateStep);
            } else {
                this._spawnImpactParticles(pt);
                // Clean up trail dots after a short delay
                setTimeout(() => {
                    if (this._trailDots) {
                        this._trailDots.forEach(d => this.scene.remove(d));
                        this._trailDots = [];
                    }
                    if (this.trailLine) {
                        this.scene.remove(this.trailLine);
                        this.trailLine = null;
                    }
                }, 800);
                this.animatingArrow = false;
                if (onComplete) onComplete();
            }
        };
        requestAnimationFrame(animateStep);
    }

    _spawnImpactParticles(position) {
        for (let i = 0; i < 15; i++) {
            const geo = new THREE.SphereGeometry(0.02 + Math.random() * 0.04, 6, 4);
            const mat = new THREE.MeshBasicMaterial({ flatShading: true,
                color: new THREE.Color().setHSL(0.06 + Math.random() * 0.08, 0.7, 0.55),
                transparent: true, opacity: 1,
            });
            const p = new THREE.Mesh(geo, mat);
            p.position.set(position[0], position[1], position[2]);
            p._vel = new THREE.Vector3(
                (Math.random() - 0.5) * 4,
                Math.random() * 4 + 1,
                (Math.random() - 0.5) * 4
            );
            p._life = 1.0;
            this.scene.add(p);
            this.particles.push(p);
        }
    }

    // -------------------------------------------------------------------
    // Animation loop
    // -------------------------------------------------------------------

    animate() {
        requestAnimationFrame(() => this.animate());
        const dt = this.clock.getDelta();
        const time = this.clock.getElapsedTime();

        if (this.mixer) {
            this.mixer.update(dt);
        }

        // Camera lerp for aim mode
        if (this._cameraLerpDir !== 0) {
            const lerpSpeed = 2.5; // lerp speed (higher = faster zoom)
            this._cameraLerp += this._cameraLerpDir * dt * lerpSpeed;
            this._cameraLerp = Math.max(0, Math.min(1, this._cameraLerp));

            // Smooth easing (ease in-out)
            const t = this._cameraLerp * this._cameraLerp * (3 - 2 * this._cameraLerp);

            const posX = this._cameraDefault.pos.x + (this._cameraAim.pos.x - this._cameraDefault.pos.x) * t;
            const posY = this._cameraDefault.pos.y + (this._cameraAim.pos.y - this._cameraDefault.pos.y) * t;
            const posZ = this._cameraDefault.pos.z + (this._cameraAim.pos.z - this._cameraDefault.pos.z) * t;
            this.camera.position.set(posX, posY, posZ);

            // Apply mouse-look offset when in aim mode (scale by t so it blends in)
            const aimOffsetYaw = (this._aimYaw || 0) * t;
            const aimOffsetPitch = (this._aimPitch || 0) * t;

            const tgtX = this._cameraDefault.target.x + (this._cameraAim.target.x - this._cameraDefault.target.x) * t;
            const tgtY = this._cameraDefault.target.y + (this._cameraAim.target.y - this._cameraDefault.target.y) * t;
            const tgtZ = this._cameraDefault.target.z + (this._cameraAim.target.z - this._cameraDefault.target.z) * t;

            // Offset the look-at target based on mouse movement
            const lookTarget = new THREE.Vector3(tgtX, tgtY, tgtZ);
            lookTarget.x += aimOffsetYaw * 15;   // horizontal: yaw moves target left/right
            lookTarget.y += aimOffsetPitch * 10;  // vertical: pitch moves target up/down
            lookTarget.z += aimOffsetYaw * -5;    // slight Z offset for depth feel

            this.camera.lookAt(lookTarget);

            // Narrow FOV as we zoom in for scope feel
            this.camera.fov = 45 - 15 * t;
            this.camera.updateProjectionMatrix();

            // Stop lerping when we reach the ends
            if (this._cameraLerp <= 0 || this._cameraLerp >= 1) {
                this._cameraLerpDir = 0;
            }
        }

        // Update aim power while holding
        this._updateAimPower();

        // Moving targets
        Object.keys(this.targets).forEach(id => {
            const obj = this.targets[id];
            if (obj.is_moving && this.targetMeshes[id]) {
                obj.position[0] += obj.velocity[0] * dt;
                obj.position[1] += obj.velocity[1] * dt;
                obj.position[2] += obj.velocity[2] * dt;
                if (Math.abs(obj.position[2]) > 10) {
                    obj.velocity[2] *= -1;
                    obj.position[2] = Math.sign(obj.position[2]) * 10;
                }
                this.targetMeshes[id].position.set(
                    obj.position[0], obj.position[1], obj.position[2]
                );
                if (this.highlightRings[id]) {
                    this.highlightRings[id].position.set(
                        obj.position[0], 0.05, obj.position[2]
                    );
                }
            }
        });

        // Wind sway on trees
        const windStrength = this.windSpeed / 30; // 0-1 normalized
        if (this.treeFoliage) {
            this.treeFoliage.forEach(t => {
                const sway = windStrength * 0.12;
                t.canopy.rotation.z = Math.sin(time * 1.5 + t.seed) * sway;
                t.canopy.rotation.x = Math.cos(time * 1.2 + t.seed * 0.7) * sway * 0.6;
            });
        }

        // Flags — respond to wind
        const flagSway = 0.15 + windStrength * 0.5;
        const flagSpeed = 3 + windStrength * 6;
        Object.values(this.flags).forEach(flag => {
            if (flag) {
                flag.rotation.y = Math.sin(time * flagSpeed + (flag.id || 0)) * flagSway;
                flag.position.x = 0.16 + Math.sin(time * (flagSpeed + 1)) * (0.02 + windStrength * 0.04);
            }
        });



        // Highlight rings
        Object.values(this.highlightRings).forEach(ring => {
            if (ring.material.opacity > 0) {
                ring._pulseTime = (ring._pulseTime || 0) + dt;
                ring.material.opacity = 0.4 + Math.sin(ring._pulseTime * 4) * 0.3;
                ring.scale.setScalar(1 + Math.sin(ring._pulseTime * 3) * 0.1);
            }
        });

        // Particles
        for (let i = this.particles.length - 1; i >= 0; i--) {
            const p = this.particles[i];
            p._life -= dt * 1.5;
            if (p._life <= 0) {
                this.scene.remove(p);
                this.particles.splice(i, 1);
                continue;
            }
            p.position.add(p._vel.clone().multiplyScalar(dt));
            p._vel.y -= 9.8 * dt;
            p.material.opacity = p._life;
            p.scale.setScalar(p._life);
        }

        this.renderer.render(this.scene, this.camera);
    }
}
